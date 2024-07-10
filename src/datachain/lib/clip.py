import inspect
from typing import Any, Callable, Literal, Union

from datachain.lib.image import convert_images
from datachain.lib.text import convert_text

try:
    import torch
    from PIL import Image
    from transformers.modeling_utils import PreTrainedModel
except ImportError as exc:
    raise ImportError(
        "Missing dependencies for computer vision:\n"
        "To install run:\n\n"
        "  pip install 'datachain[cv]'\n"
    ) from exc


def _get_encoder(model: Any, type: Literal["image", "text"]) -> Callable:
    # Check for transformers CLIPModel
    method_name = f"get_{type}_features"
    if isinstance(model, PreTrainedModel) and (
        hasattr(model, method_name) and inspect.ismethod(getattr(model, method_name))
    ):
        method = getattr(model, method_name)
        return lambda x: method(torch.tensor(x))

    # Check for model from clip or open_clip library
    method_name = f"encode_{type}"
    if hasattr(model, method_name) and inspect.ismethod(getattr(model, method_name)):
        return getattr(model, method_name)

    raise ValueError(
        f"Error encoding {type}: "
        "'model' must be a CLIP model from clip, open_clip, or transformers library."
    )


def similarity_scores(
    images: Union[None, Image.Image, list[Image.Image]],
    text: Union[None, str, list[str]],
    model: Any,
    preprocess: Callable,
    tokenizer: Callable,
    prob: bool = False,
    image_to_text: bool = True,
) -> list[list[float]]:
    """
    Calculate CLIP similarity scores between one or more images and/or text.

    Args:
        images: Images to use as inputs.
        text: Text to use as inputs.
        model: Model from clip or open_clip packages.
        preprocess: Image preprocessor to apply.
        tokenizer: Text tokenizer.
        prob: Compute softmax probabilities.
        image_to_text: Whether to compute for image-to-text or text-to-image. Ignored if
            only one of images or text provided.


    Examples
    --------

    using https://github.com/openai/CLIP
    >>> import clip
    >>> model, preprocess = clip.load("ViT-B/32")
    >>> similarity_scores(img, "cat", model, preprocess, clip.tokenize)
    [[21.813]]

    using https://github.com/mlfoundations/open_clip
    >>> import open_clip
    >>> model, _, preprocess = open_clip.create_model_and_transforms(
    ...     "ViT-B-32", pretrained="laion2b_s34b_b79k"
    ... )
    >>> tokenizer = open_clip.get_tokenizer("ViT-B-32")
    >>> similarity_scores(img, "cat", model, preprocess, tokenizer)
    [[21.813]]

    using https://huggingface.co/docs/transformers/en/model_doc/clip
    >>> from transformers import CLIPProcessor, CLIPModel
    >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    >>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    >>> scores = similarity_scores(
    ...     img, "cat", model, processor.image_processor, processor.tokenizer
    ... )
    [[21.813]]

    image -> list of text
    >>> similarity_scores(img, ["cat", "dog"], model, preprocess, tokenizer)
    [[21.813, 35.313]]

    list of images -> text
    >>> similarity_scores([img1, img2], "cat", model, preprocess, tokenizer)
    [[21.813], [83.123]]

    list of images -> list of text
    >>> similarity_scores([img1, img2], ["cat", "dog"], model, preprocess, tokenizer)
    [[21.813, 35.313], [83.123, 34.843]]

    list of images -> list of images
    >>> similarity_scores([img1, img2], None, model, preprocess, tokenizer)
    [[94.189, 37.092]]

    list of text -> list of text
    >>> similarity_scores(None, ["cat", "dog"], model, preprocess, tokenizer)
    [[67.334, 23.588]]

    text -> list of images
    >>> similarity_scores([img1, img2], "cat", ..., image_to_text=False)
    [[19.708, 19.842]]

    show scores as softmax probabilities
    >>> similarity_scores(img, ["cat", "dog"], ..., prob=True)
    [[0.423, 0.577]]
    """

    with torch.no_grad():
        if images is not None:
            encoder = _get_encoder(model, "image")
            image_features = convert_images(
                images, transform=preprocess, encoder=encoder
            )
            image_features /= image_features.norm(dim=-1, keepdim=True)  # type: ignore[union-attr]

        if text is not None:
            encoder = _get_encoder(model, "text")
            text_features = convert_text(text, tokenizer, encoder=encoder)
            text_features /= text_features.norm(dim=-1, keepdim=True)  # type: ignore[union-attr]

        if images is not None and text is not None:
            if image_to_text:
                logits = 100.0 * image_features @ text_features.T  # type: ignore[operator,union-attr]
            else:
                logits = 100.0 * text_features @ image_features.T  # type: ignore[operator,union-attr]
        elif images is not None:
            logits = 100.0 * image_features @ image_features.T  # type: ignore[operator,union-attr]
        elif text is not None:
            logits = 100.0 * text_features @ text_features.T  # type: ignore[operator,union-attr]
        else:
            raise ValueError(
                "Error calculating CLIP similarity - "
                "provide at least one of images or text"
            )

        if prob:
            scores = logits.softmax(dim=1)
        else:
            scores = logits

        return scores.tolist()
