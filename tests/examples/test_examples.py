import glob
import os
import subprocess
import sys

import pytest

NO_EXAMPLES = "no examples found"


def can_import_unstructured():
    try:
        import unstructured  # noqa: F401

        return True
    except ImportError:
        return False


get_started_examples = [
    filename
    for filename in glob.glob("examples/get_started/**/*.py", recursive=True)
    if "torch" not in filename or os.environ.get("RUNNER_OS") != "Linux"
]

llm_and_nlp_examples = [
    filename
    for filename in glob.glob("examples/llm_and_nlp/**/*.py", recursive=True)
    # no anthropic token
    if "claude" not in filename
    and ("unstructured" not in filename or can_import_unstructured())
] or [NO_EXAMPLES]

multimodal_examples = [
    filename
    for filename in glob.glob("examples/multimodal/**/*.py", recursive=True)
    # no OpenAI token
    # and hf download painfully slow
    if "openai" not in filename and "hf" not in filename
]

computer_vision_examples = [
    filename
    for filename in glob.glob("examples/computer_vision/**/*.py", recursive=True)
    # fashion product images tutorial out of scope
    # and hf download painfully slow
    if "image_desc" not in filename and "fashion_product_images" not in filename
]


def smoke_test(example: str):
    if example == NO_EXAMPLES:
        return

    completed_process = subprocess.run(  # noqa: S603
        [sys.executable, example],
        capture_output=True,
        cwd=os.path.abspath(os.path.join(__file__, "..", "..", "..")),
        check=True,
    )

    assert completed_process.stdout
    assert completed_process.stderr


@pytest.mark.examples
@pytest.mark.parametrize("example", get_started_examples)
def test_get_started_examples(example):
    smoke_test(example)


@pytest.mark.examples
@pytest.mark.parametrize("example", llm_and_nlp_examples)
def test_llm_and_nlp_examples(example):
    smoke_test(example)


@pytest.mark.examples
@pytest.mark.parametrize("example", multimodal_examples)
def test_multimodal(example):
    smoke_test(example)


@pytest.mark.examples
@pytest.mark.parametrize("example", computer_vision_examples)
def test_computer_vision_examples(example):
    smoke_test(example)
