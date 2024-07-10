# pip install torch
import torch

from datachain.lib.hf_pipeline import ImageHelper, RawHelper, TextHelper
from datachain.query import C, DatasetQuery

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
        DatasetQuery(
            image_source,
            anon=True,
        )
        .filter(C.name.glob("*.jpg"))
        .limit(1)
        .add_signals(
            ImageHelper(
                model="google/owlv2-base-patch16",
                device=device,
                candidate_labels=["cat", "dog", "squirrel", "unknown"],
            ),
            parallel=False,
        )
        .select("source", "parent", "name", "model_output", "error")
        .results()
    )
    print(*results, sep="\n")

    print("\nNot-safe-for-work image detection:")
    results = (
        DatasetQuery(
            image_source,
            anon=True,
        )
        .filter(C.name.glob("*.jpg"))
        .limit(1)
        .add_signals(
            ImageHelper(
                model="Falconsai/nsfw_image_detection",
                device=device,
            ),
            parallel=False,
        )
        .select("source", "parent", "name", "model_output", "error")
        .results()
    )
    print(*results, sep="\n")

    print("\nAudio emotion classification:")
    results = (
        DatasetQuery(
            audio_source,
            anon=True,
        )
        .filter(C.name.glob("*.wav"))
        .limit(1)
        .add_signals(
            RawHelper(
                model="Krithika-p/my_awesome_emotions_model",
                device=device,
            ),
            parallel=False,
        )
        .select("source", "parent", "name", "model_output", "error")
        .results()
    )
    print(*results, sep="\n")
    print("\nLong text summarization:")
    results = (
        DatasetQuery(
            text_source,
            anon=True,
        )
        .filter(C.name.glob("*.story"))
        .limit(1)
        .add_signals(
            TextHelper(
                model="pszemraj/led-large-book-summary",
                device=device,
                max_length=150,
            ),
            parallel=False,
        )
        .select("source", "parent", "name", "model_output", "error")
        .results()
    )
    print(*results, sep="\n")
