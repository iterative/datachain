import os
from typing import Callable, Optional

import anthropic
from anthropic.types.message import Message

from datachain.lib.file import File

default_model_name = "claude-3-haiku-20240307"
DEFAULT_OUTPUT_TOKENS = 1024


def claude_processor(
    prompt: str,
    messages: Optional[list] = None,
    model: str = "claude-3-haiku-20240307",
    api_key: Optional[str] = "",
    max_retries: int = 5,
    temperature: float = 0.9,
    max_tokens: int = 1024,
    **kwargs,
) -> Callable:
    if not messages:
        messages = []
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    def claude_func(file) -> Message:
        data = file.get_value() if isinstance(file, File) else file
        client = anthropic.Anthropic(api_key=api_key, max_retries=max_retries)
        return client.messages.create(
            model=model,
            system=prompt,
            messages=[{"role": "user", "content": data}, *messages],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    return claude_func
