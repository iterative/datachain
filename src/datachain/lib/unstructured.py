from unstructured.partition.auto import partition
from unstructured.staging.base import convert_to_dataframe


def partition_object(file):
    with file.open() as raw:
        elements = partition(file=raw, metadata_filename=file.name)
    title = str(elements[0])
    text = "\n\n".join([str(el) for el in elements])
    df = convert_to_dataframe(elements)
    return (df.to_json(), title, text, "")
