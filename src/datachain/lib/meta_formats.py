# pip install datamodel-code-generator
# pip install jmespath
#
import csv
import io
import json
import subprocess
import sys
import uuid
from collections.abc import Iterator
from typing import Any, Callable

import jmespath as jsp
from pydantic import ValidationError

from datachain.lib.feature_utils import pydantic_to_feature  # noqa: F401
from datachain.lib.file import File

# from datachain.lib.dc import C, DataChain


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
        uid_str = str(generate_uuid()).replace(
            "-", ""
        )  # comply with Python class names
        model_name = f"Model{data_type}{uid_str}"
    try:
        with source_file.open() as fd:  # CSV can be larger than memory
            if data_type == "csv":
                data_string += fd.readline().decode("utf-8", "ignore").replace("\r", "")
                data_string += fd.readline().decode("utf-8", "ignore").replace("\r", "")
            elif data_type == "jsonl":
                data_string = fd.readline().decode("utf-8", "ignore").replace("\r", "")
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
    command = [
        "datamodel-codegen",
        "--input-file-type",
        data_type,
        "--class-name",
        model_name,
    ]
    try:
        result = subprocess.run(  # noqa: S603
            command,
            input=data_string,
            text=True,
            capture_output=True,
            check=True,
        )
        model_output = (
            result.stdout
        )  # This will contain the output from datamodel-codegen
    except subprocess.CalledProcessError as e:
        model_output = f"An error occurred in datamodel-codegen: {e.stderr}"
    print(f"{model_output}")
    print("\n" + f"spec=pydantic_to_feature({model_name})" + "\n")
    return model_output


#
# UDF mapper which calls chain in the setup to infer the dynamic schema
#
def read_meta(  # noqa: C901
    spec=None,
    schema_from=None,
    meta_type="json",
    jmespath=None,
    show_schema=False,
    model_name=None,
) -> Callable:
    from datachain.lib.dc import DataChain

    # ugly hack: datachain is run redirecting printed outputs to a variable
    if schema_from:
        captured_output = io.StringIO()
        current_stdout = sys.stdout
        sys.stdout = captured_output
        try:
            chain = (
                DataChain.from_storage(schema_from)
                .limit(1)
                .map(  # dummy column created (#1615)
                    meta_schema=lambda file: read_schema(
                        file, data_type=meta_type, expr=jmespath, model_name=model_name
                    ),
                    output=str,
                )
            )
            # dummy executor (#1616)
            chain.save()
        finally:
            sys.stdout = current_stdout
        model_output = captured_output.getvalue()
        captured_output.close()

        if show_schema:
            print(f"{model_output}")
        # Below 'spec' should be a dynamically converted Feature from Pydantic datamodel
        if not spec:
            local_vars: dict[str, Any] = {}
            exec(model_output, globals(), local_vars)  # noqa: S102
            spec = local_vars["spec"]

    if not (spec) and not (schema_from):
        raise ValueError(
            "Must provide a static schema in spec: or metadata sample in schema_from:"
        )

    #
    # UDF mapper parsing a JSON or CSV file using schema spec
    #

    def parse_data(
        file: File,
        DataModel=spec,  # noqa: N803
        meta_type=meta_type,
        jmespath=jmespath,
    ) -> Iterator[spec]:
        def validator(json_object: dict) -> spec:
            json_string = json.dumps(json_object)
            try:
                data_instance = DataModel.model_validate_json(json_string)
                yield data_instance
            except ValidationError as e:
                print(f"Validation error occurred in file {file.name}:", e)

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
                for json_dict in json_object:
                    yield from validator(json_dict)

        if meta_type == "jsonl":
            try:
                with file.open() as fd:
                    data_string = fd.readline().replace("\r", "")
                    while data_string:
                        json_object = process_json(data_string, jmespath)
                        data_string = fd.readline()
                        yield from validator(json_object)
            except OSError as e:
                print(f"An unexpected file error occurred in file {file.name}: {e}")

    return parse_data
