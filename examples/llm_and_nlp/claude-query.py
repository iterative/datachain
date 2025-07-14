import os
import sys

import anthropic
from pydantic import BaseModel

import datachain as dc
from datachain import C, File

DATA = "gs://datachain-demo/chatbot-KiT"
MODEL = "claude-3-5-haiku-latest"
TEMPERATURE = 0.9
DEFAULT_OUTPUT_TOKENS = 1024

PROMPT = """Consider the dialogue between the 'user' and the 'bot'. The 'user' is a
 human trying to find the best mobile plan. The 'bot' is a chatbot designed to query
 the user and offer the best  solution. The dialog is successful if the 'bot' is able to
 gather the information and offer a plan, or inform the user that such plan does not
 exist. The dialog is not successful if the conversation ends early or the 'user'
 requests additional functions the 'bot' cannot perform. Read the dialogue below and
 rate it 'Success' if it is successful, and 'Failure' if not. After that, provide
 one-sentence explanation of the reasons for this rating. Use only JSON object as output
 with the keys 'status', and 'explanation'.
"""

API_KEY = os.environ.get("ANTHROPIC_API_KEY")

if not API_KEY:
    print("This example requires an Anthropic API key")
    print("Add your key using the ANTHROPIC_API_KEY environment variable.")
    sys.exit(0)


class Rating(BaseModel):
    status: str = ""
    explanation: str = ""


def rate(client: anthropic.Anthropic, file: File) -> Rating:
    content = file.read()
    response = client.messages.create(
        model=MODEL,
        max_tokens=DEFAULT_OUTPUT_TOKENS,
        system=PROMPT,
        temperature=TEMPERATURE,
        messages=[
            {"role": "user", "content": f"{content}"},
        ],
    )

    first_block = response.content[0]
    if first_block.type == "text":
        return Rating.model_validate_json(first_block.text)
    raise ValueError(f"Unexpected content block type: {first_block.type}")


(
    dc.read_storage(DATA, type="text", anon=True)
    .filter(C("file.path").glob("*.txt"))
    .limit(4)
    .settings(parallel=2, cache=True)
    .setup(client=lambda: anthropic.Anthropic(api_key=API_KEY))
    .map(rating=rate)
    .show()
)
