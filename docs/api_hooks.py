# ruff: noqa: INP001
def on_pre_build(**kwargs):
    import os

    import requests

    # Skip if files already exist
    if os.path.exists("docs/openapi.json") and os.path.exists(
        "docs/studio/api/index.md"
    ):
        print("API docs already exist, skipping generation")
        return

    # Download OpenAPI spec
    response = requests.get(
        "https://studio.datachain.ai/api/openapi.json",
        timeout=30,
    )

    # Write to file
    print("Writing OpenAPI spec to docs/openapi.json")
    with open("docs/openapi.json", "w") as f:
        f.write(response.text)

    # Generate API docs using widdershins
    print("Generating API docs using widdershins")
    cmd = [
        "npx widdershins",
        "docs/openapi.json",
        "-o docs/studio/api/index.md",
        "--language_tabs 'python:Python'",
        "--language_tabs 'shell:curl'",
        "--expandBody true",
        "--summary",
        "--shallowSchemas",
        "--omitBody",
        "--resolve",
        "--httpsnippet",
        "-u docs/templates",
        "--omitHeader",
    ]
    os.system(" ".join(cmd))  # noqa: S605
