import os

import anthropic
from anthropic.types import Message

from datachain import Column, DataChain, File

DATA = "gs://datachain-demo/chatbot-KiT"
MODEL = "claude-3-opus-20240229"
PROMPT = """Summarise the dialog in a sentence"""
TEMPERATURE = 0.9
DEFAULT_OUTPUT_TOKENS = 1024

API_KEY = os.environ.get("ANTHROPIC_API_KEY")

chain = (
    DataChain.from_storage(DATA, type="text")
    .filter(Column("file.name").glob("*.txt"))
    .limit(5)
    .settings(parallel=4, cache=True)
    .setup(client=lambda: anthropic.Anthropic(api_key=API_KEY))
    .map(
        claude=lambda client, file: client.messages.create(
            model=MODEL,
            system=PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": file.read() if isinstance(file, File) else file,
                },
            ],
            temperature=TEMPERATURE,
            max_tokens=DEFAULT_OUTPUT_TOKENS,
        ),
        output=Message,
    )
)

chain.show()
