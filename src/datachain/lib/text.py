from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from datachain.lib.file import File

if TYPE_CHECKING:
    import torch

    from datachain.catalog import Catalog


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


class TextFile(File):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._stream = None

    def _set_stream(self, catalog: "Catalog", caching_enabled: bool = False) -> None:
        super()._set_stream(catalog, caching_enabled)
        self._stream.set_mode("r")
