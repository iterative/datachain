import os
from typing import Callable, Literal, Optional

import anthropic

from datachain.lib.feature import Feature
from datachain.lib.file import File

default_model_name = "claude-3-haiku-20240307"
DEFAULT_OUTPUT_TOKENS = 1024

# This classes can be auto-generated:
# >> from anthropic.types.message import Message
# >> ClaudeMessage = pydantic_to_feature(Message)
# However, auto-generated pydentic classes do not work in multithreading mode.


class UsageFr(Feature):
    input_tokens: int = 0
    output_tokens: int = 0


class TextBlockFr(Feature):
    text: str = ""
    type: str = "text"


class ClaudeMessage(Feature):
    id: str = ""
    content: list[TextBlockFr]
    model: str = ""
    role: str = ""
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence"]] = None
    stop_sequence: Optional[str] = None
    type: Literal["message"] = "message"
    usage: UsageFr = UsageFr()


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

    def claude_func(file) -> ClaudeMessage:
        try:
            data = file.get_value() if isinstance(file, File) else file
            client = anthropic.Anthropic(api_key=api_key, max_retries=max_retries)
            response = client.messages.create(
                model=model,
                system=prompt,
                messages=[{"role": "user", "content": data}, *messages],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            return ClaudeMessage(**response.model_dump())
        except Exception:  # noqa: BLE001
            return ClaudeMessage(content=[])

    return claude_func
