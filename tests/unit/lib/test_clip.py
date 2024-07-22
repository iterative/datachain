import pytest
from PIL import Image

from datachain.lib.clip import similarity_scores

IMAGES = [Image.new(mode="RGB", size=(64, 64)), Image.new(mode="RGB", size=(32, 32))]
TEXTS = ["text1", "text2"]


@pytest.mark.parametrize(
    "images",
    [None, Image.new(mode="RGB", size=(64, 64)), IMAGES],
)
@pytest.mark.parametrize("text", [None, "text", TEXTS])
@pytest.mark.parametrize("prob", [True, False])
@pytest.mark.parametrize("image_to_text", [True, False])
def test_similarity_scores(fake_clip_model, images, text, prob, image_to_text):
    model, preprocess, tokenizer = fake_clip_model
    if not (images or text):
        with pytest.raises(ValueError):
            scores = similarity_scores(
                images, text, model, preprocess, tokenizer, prob, image_to_text
            )
    else:
        scores = similarity_scores(
            images, text, model, preprocess, tokenizer, prob, image_to_text
        )
        assert isinstance(scores, list)
        if not images:
            image_to_text = False
        elif not text:
            image_to_text = True
        if image_to_text:
            if isinstance(images, list):
                assert len(scores) == len(images)
            else:
                assert len(scores) == 1
        elif not image_to_text:
            if isinstance(text, list):
                assert len(scores) == len(text)
            else:
                assert len(scores) == 1
        if prob:
            for score in scores:
                assert sum(score) == pytest.approx(1)


def test_similarity_scores_hf(fake_hf_model):
    model, processor = fake_hf_model

    scores = similarity_scores(
        IMAGES, TEXTS, model, processor.image_processor, processor.tokenizer
    )
    assert isinstance(scores, list)
    assert len(scores) == len(IMAGES)
