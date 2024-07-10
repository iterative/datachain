from typing import TYPE_CHECKING, Any, Callable, Optional, Union

if TYPE_CHECKING:
    import torch


def convert_text(
    text: Union[str, list[str]],
    tokenizer: Optional[Callable] = None,
    tokenizer_kwargs: Optional[dict[str, Any]] = None,
    encoder: Optional[Callable] = None,
) -> Union[str, list[str], "torch.Tensor"]:
    """
    Tokenize and otherwise transform text.

    Args:
        text (str): Text to convert.
        tokenizer (Callable): Tokenizer to use to tokenize objects.
        tokenizer_kwargs (dict): Additional kwargs to pass when calling tokenizer.
        encoder (Callable): Encode text using model.
    """
    if not tokenizer:
        return text

    if isinstance(text, str):
        text = [text]

    if tokenizer_kwargs:
        res = tokenizer(text, **tokenizer_kwargs)
    else:
        res = tokenizer(text)
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        tokens = (
            res.input_ids if isinstance(tokenizer, PreTrainedTokenizerBase) else res
        )
    except ImportError:
        tokens = res

    if not encoder:
        return tokens

    try:
        import torch
    except ImportError:
        "Missing dependency 'torch' needed to encode text."

    return encoder(torch.tensor(tokens))
