# pip install torch
# NOTE: also need to install ffmpeg binary
import torch

from datachain.lib.dc import C, DataChain
from datachain.lib.hf_pipeline import Helper

image_source = "gs://dvcx-datalakes/dogs-and-cats/"
audio_source = "gs://dvcx-datalakes/speech-emotion-recognition-dataset/"
text_source = "gs://dvcx-datalakes/NLP/cnn/stories"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


if __name__ == "__main__":
    print("** HuggingFace pipeline helper model zoo demo **")
    print("\nZero-shot object detection and classification:")
    results = (
        DataChain.from_storage(
            image_source,
            anon=True,
            type="image",
        )
        .filter(C("name").glob("*.jpg"))
        .limit(1)
        .map(
            Helper(
                model="google/owlv2-base-patch16",
                device=device,
                candidate_labels=["cat", "dog", "squirrel", "unknown"],
            ),
            params=["file"],
            output={"model_output": dict, "error": str},
        )
        .select("file.source", "file.parent", "file.name", "model_output", "error")
        .results()
    )
    print(*results, sep="\n")

    print("\nNot-safe-for-work image detection:")
    results = (
        DataChain.from_storage(
            image_source,
            anon=True,
            type="image",
        )
        .filter(C("name").glob("*.jpg"))
        .limit(1)
        .map(
            Helper(
                model="Falconsai/nsfw_image_detection",
                device=device,
            ),
            params=["file"],
            output={"model_output": dict, "error": str},
        )
        .select("file.source", "file.parent", "file.name", "model_output", "error")
        .results()
    )
    print(*results, sep="\n")

    print("\nAudio emotion classification:")
    results = (
        DataChain.from_storage(
            audio_source,
            anon=True,
            type="binary",
        )
        .filter(C("name").glob("*.wav"))
        .limit(1)
        .map(
            Helper(
                model="Krithika-p/my_awesome_emotions_model",
                device=device,
            ),
            params=["file"],
            output={"model_output": dict, "error": str},
        )
        .select("file.source", "file.parent", "file.name", "model_output", "error")
        .results()
    )
    print(*results, sep="\n")
    print("\nLong text summarization:")
    results = (
        DataChain.from_storage(
            text_source,
            anon=True,
            type="text",
        )
        .filter(C("name").glob("*.story"))
        .limit(1)
        .map(
            Helper(
                model="pszemraj/led-large-book-summary",
                device=device,
                max_length=150,
            ),
            params=["file"],
            output={"model_output": dict, "error": str},
        )
        .select("file.source", "file.parent", "file.name", "model_output", "error")
        .results()
    )
    print(*results, sep="\n")
