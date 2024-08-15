import glob
import os
import subprocess
import sys
from typing import Optional

import pytest

get_started_examples = [
    filename
    for filename in glob.glob("examples/get_started/**/*.py", recursive=True)
    # torch-loader will not finish within an hour on Linux runner
    if "torch" not in filename or os.environ.get("RUNNER_OS") != "Linux"
]

llm_and_nlp_examples = [
    filename
    for filename in glob.glob("examples/llm_and_nlp/**/*.py", recursive=True)
    # no anthropic token
    if "claude" not in filename
]

multimodal_examples = [
    filename
    for filename in glob.glob("examples/multimodal/**/*.py", recursive=True)
    # no OpenAI token
    if "openai" not in filename
]

computer_vision_examples = [
    filename
    for filename in glob.glob("examples/computer_vision/**/*.py", recursive=True)
    # fashion product images tutorial out of scope
    # and hf download painfully slow
    if "image_desc" not in filename and "fashion_product_images" not in filename
]


def smoke_test(example: str, env: Optional[dict] = None):
    try:
        completed_process = subprocess.run(  # noqa: S603
            [sys.executable, example],
            env={**os.environ, **(env or {})},
            capture_output=True,
            cwd=os.path.abspath(os.path.join(__file__, "..", "..", "..")),
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Example failed: {example}")
        print()
        print()
        print("stdout:")
        print(e.stdout.decode("utf-8"))
        print()
        print()
        print("stderr:")
        print(e.stderr.decode("utf-8"))
        pytest.fail("subprocess returned a non-zero exit code")

    assert completed_process.stdout
    assert completed_process.stderr


@pytest.mark.examples
@pytest.mark.get_started
@pytest.mark.parametrize("example", get_started_examples)
def test_get_started_examples(example):
    smoke_test(example, {"NUM_EPOCHS": "1"})


@pytest.mark.examples
@pytest.mark.llm_and_nlp
@pytest.mark.parametrize("example", llm_and_nlp_examples)
def test_llm_and_nlp_examples(example):
    smoke_test(example)


@pytest.mark.examples
@pytest.mark.multimodal
@pytest.mark.parametrize("example", multimodal_examples)
def test_multimodal(example):
    smoke_test(
        example,
        {
            "IMAGE_TARS": "gs://datachain-demo/datacomp-small/shards/00001286.tar",
            "PARQUET_METADATA": "gs://datachain-demo/datacomp-small/metadata/036d6b9ae87a00e738f8fc554130b65b.parquet",
            "NPZ_METADATA": "gs://datachain-demo/datacomp-small/metadata/036d6b9ae87a00e738f8fc554130b65b.npz",
        },
    )


@pytest.mark.examples
@pytest.mark.computer_vision
@pytest.mark.parametrize("example", computer_vision_examples)
def test_computer_vision_examples(example):
    smoke_test(example)
