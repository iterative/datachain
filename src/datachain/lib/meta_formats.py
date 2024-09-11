import csv
import json
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Callable

import datamodel_code_generator
import jmespath as jsp
from pydantic import BaseModel, ConfigDict, Field, ValidationError  # noqa: F401

from datachain.lib.data_model import DataModel  # noqa: F401
from datachain.lib.file import File


class UserModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


def generate_uuid():
    return uuid.uuid4()  # Generates a random UUID.


# JSON decoder
def load_json_from_string(json_string):
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        print(f"Failed to decode JSON: {json_string} is not formatted correctly.")
        return None


# Validate and reduce JSON
def process_json(data_string, jmespath):
    json_dict = load_json_from_string(data_string)
    if jmespath:
        json_dict = jsp.search(jmespath, json_dict)
    return json_dict


# Print a dynamic datamodel-codegen output from JSON or CSV on stdout
def read_schema(source_file, data_type="csv", expr=None, model_name=None):
    data_string = ""
    # using uiid to get around issue #1617
    if not model_name:
        # comply with Python class names
        uid_str = str(generate_uuid()).replace("-", "")
        model_name = f"Model{data_type}{uid_str}"
    try:
        with source_file.open() as fd:  # CSV can be larger than memory
            if data_type == "csv":
                data_string += fd.readline().replace("\r", "")
                data_string += fd.readline().replace("\r", "")
            elif data_type == "jsonl":
                data_string = fd.readline().replace("\r", "")
            else:
                data_string = fd.read()  # other meta must fit into RAM
    except OSError as e:
        print(f"An unexpected file error occurred: {e}")
        return
    if data_type in ("json", "jsonl"):
        json_object = process_json(data_string, expr)
        if data_type == "json" and isinstance(json_object, list):
            json_object = json_object[0]  # sample the 1st object from JSON array
        if data_type == "jsonl":
            data_type = "json"  # treat json line as plain JSON in auto-schema
        data_string = json.dumps(json_object)

    input_file_types = {i.value: i for i in datamodel_code_generator.InputFileType}
    input_file_type = input_file_types[data_type]
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "model.py"
        datamodel_code_generator.generate(
            data_string,
            input_file_type=input_file_type,
            output=output,
            target_python_version=datamodel_code_generator.PythonVersion.PY_39,
            base_class="datachain.lib.meta_formats.UserModel",
            class_name=model_name,
            additional_imports=["datachain.lib.data_model.DataModel"],
            use_standard_collections=True,
        )
        epilogue = f"""
DataModel.register({model_name})
spec = {model_name}
"""
        return output.read_text() + epilogue


#
# UDF mapper which calls chain in the setup to infer the dynamic schema
#
def read_meta(  # noqa: C901
    spec=None,
    schema_from=None,
    meta_type="json",
    jmespath=None,
    print_schema=False,
    model_name=None,
    nrows=None,
) -> Callable:
    from datachain.lib.dc import DataChain

    if schema_from:
        chain = (
            DataChain.from_storage(schema_from, type="text")
            .limit(1)
            .map(  # dummy column created (#1615)
                meta_schema=lambda file: read_schema(
                    file, data_type=meta_type, expr=jmespath, model_name=model_name
                ),
                output=str,
            )
        )
        (model_output,) = chain.collect("meta_schema")
        if print_schema:
            print(f"{model_output}")
        # Below 'spec' should be a dynamically converted DataModel from Pydantic
        if not spec:
            gl = globals()
            exec(model_output, gl)  # type: ignore[arg-type] # noqa: S102
            spec = gl["spec"]

    if not (spec) and not (schema_from):
        raise ValueError(
            "Must provide a static schema in spec: or metadata sample in schema_from:"
        )

    #
    # UDF mapper parsing a JSON or CSV file using schema spec
    #

    def parse_data(
        file: File,
        data_model=spec,
        meta_type=meta_type,
        jmespath=jmespath,
        nrows=nrows,
    ) -> Iterator[spec]:
        def validator(json_object: dict, nrow=0) -> spec:
            json_string = json.dumps(json_object)
            try:
                data_instance = data_model.model_validate_json(json_string)
                yield data_instance
            except ValidationError as e:
                print(f"Validation error occurred in row {nrow} file {file.name}:", e)

        if meta_type == "csv":
            with (
                file.open() as fd
            ):  # TODO: if schema is statically given, should allow CSV without headers
                reader = csv.DictReader(fd)
                for row in reader:  # CSV can be larger than memory
                    yield from validator(row)

        if meta_type == "json":
            try:
                with file.open() as fd:  # JSON must fit into RAM
                    data_string = fd.read()
            except OSError as e:
                print(f"An unexpected file error occurred in file {file.name}: {e}")
            json_object = process_json(data_string, jmespath)
            if not isinstance(json_object, list):
                yield from validator(json_object)

            else:
                nrow = 0
                for json_dict in json_object:
                    nrow = nrow + 1
                    if nrows is not None and nrow > nrows:
                        return
                    yield from validator(json_dict, nrow)

        if meta_type == "jsonl":
            try:
                nrow = 0
                with file.open() as fd:
                    data_string = fd.readline().replace("\r", "")
                    while data_string:
                        nrow = nrow + 1
                        if nrows is not None and nrow > nrows:
                            return
                        json_object = process_json(data_string, jmespath)
                        data_string = fd.readline()
                        yield from validator(json_object, nrow)
            except OSError as e:
                print(f"An unexpected file error occurred in file {file.name}: {e}")

    return parse_data
