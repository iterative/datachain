import json
import os
import sys

import anthropic
from anthropic.types import Message
from pydantic import BaseModel

from datachain import Column, DataChain, File

DATA = "gs://datachain-demo/chatbot-KiT"
MODEL = "claude-3-opus-20240229"
PROMPT = """Consider the dialogue between the 'user' and the 'bot'. \
The 'user' is a human trying to find the best mobile plan. \
The 'bot' is a chatbot designed to query the user and offer the \
best  solution. The dialog is successful if the 'bot' is able to \
gather the information and offer a plan, or inform the user that \
such plan does not exist. The dialog is not successful if the \
conversation ends early or the 'user' requests additional functions \
the 'bot' cannot perform. Read the dialogue below and rate it 'Success' \
if it is successful, and 'Failure' if not. After that, provide \
one-sentence explanation of the reasons for this rating. Use only \
JSON object as output with the keys 'status', and 'explanation'.
"""
TEMPERATURE = 0.9
DEFAULT_OUTPUT_TOKENS = 1024

API_KEY = os.environ.get("ANTHROPIC_API_KEY")

if not API_KEY:
    print("This example requires an Anthropic API key")
    print("Add your key using the ANTHROPIC_API_KEY environment variable.")
    sys.exit(0)


class Rating(BaseModel):
    status: str = ""
    explanation: str = ""


chain = (
    DataChain.from_storage(DATA, type="text")
    .filter(Column("file.path").glob("*.txt"))
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
    .map(
        rating=lambda claude: Rating(
            **(json.loads(claude.content[0].text) if claude.content else {})
        ),
        output=Rating,
    )
)

chain = chain.settings(parallel=13).mutate(
    x=Column("file.path"),
    y=Column("rating.status"),
    price=Column("claude.usage.output_tokens") * 0.0072,
)

# chain.print_schema()

chain.show()
