# Torch

Use `pip install datachain[torch]` and then import from `datachain.torch` to use the
[PyTorch](https://pytorch.org/) functionality.
[`DataChain.to_pytorch`](datachain.md#datachain.lib.dc.DataChain.to_pytorch) converts a
chain into a PyTorch `Dataset` for downstream tasks like model training or inference.
The classes and methods below help manipulate data from the chain for PyTorch.

::: datachain.lib.clip.clip_similarity_scores

::: datachain.lib.image.convert_image

::: datachain.lib.image.convert_images

::: datachain.lib.text.convert_text

::: datachain.lib.pytorch.label_to_int
