from typing import Any, Callable, Optional, Union

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def convert_text(
    text: Union[str, list[str]],
    tokenizer: Optional[Callable] = None,
    tokenizer_kwargs: Optional[dict[str, Any]] = None,
    encoder: Optional[Callable] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Union[str, list[str], torch.Tensor]:
    """
    Tokenize and otherwise transform text.

    Args:
        text (str): Text to convert.
        tokenizer (Callable): Tokenizer to use to tokenize objects.
        tokenizer_kwargs (dict): Additional kwargs to pass when calling tokenizer.
        encoder (Callable): Encode text using model.
        device (str or torch.device): Device to use.
    """
    if not tokenizer:
        return text

    if isinstance(text, str):
        text = [text]

    if tokenizer_kwargs:
        res = tokenizer(text, **tokenizer_kwargs)
    else:
        res = tokenizer(text)

    tokens = res.input_ids if isinstance(tokenizer, PreTrainedTokenizerBase) else res
    tokens = torch.as_tensor(tokens).clone().detach()
    if device:
        tokens = tokens.to(device)

    if not encoder:
        return tokens

    return encoder(tokens)
