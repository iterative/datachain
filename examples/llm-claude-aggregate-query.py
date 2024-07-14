import os

import anthropic
import pandas as pd
from anthropic.types import Message

from datachain import Column
from datachain.lib.dc import C, DataChain
from datachain.sql.functions import path

DATA = "gs://dvcx-datalakes/chatbot-public"
MODEL = "claude-3-opus-20240229"
PROMPT = """Consider the following dialogues between the 'user' and the 'bot' separated\
 by '===='. The 'user' is a human trying to find the best mobile plan. The 'bot' is a \
chatbot designed to query the user and offer the best solution. The dialog is \
successful if the 'bot' is able to gather the information and offer a plan, or inform \
the user that such plan does not exist. The dialog is not successful if the \
conversation ends early or the 'user' requests additional functions the 'bot' \
cannot perform. Read the dialogues and classify them into a fixed number of concise \
failure reasons covering most failure cases. Present output as JSON list of reason \
strings and nothing else.
"""

TEMPERATURE = 0.9
DEFAULT_OUTPUT_TOKENS = 1024

API_KEY = os.environ.get("ANTHROPIC_API_KEY")


chain = (
    DataChain.from_storage(DATA, type="text")
    .filter(Column("file.name").glob("*.txt"))
    .limit(5)
    .settings(parallel=4, cache=True)
    .agg(
        dialogues=lambda file: ["\n=====\n".join(f.get_value() for f in file)],
        output=str,
        partition_by=path.file_ext(C.name),
    )
    .setup(client=lambda: anthropic.Anthropic(api_key=API_KEY))
    .map(
        claude=lambda client, dialogues: client.messages.create(
            model=MODEL,
            system=PROMPT,
            messages=[
                {"role": "user", "content": dialogues},
            ],
            temperature=TEMPERATURE,
            max_tokens=DEFAULT_OUTPUT_TOKENS,
        ),
        output=Message,
    )
    .map(
        res=lambda claude: claude.content[0].text if claude.content else [],
        output=str,
    )
)

with pd.option_context("display.max_columns", None):
    df = chain.to_pandas()
    print(df)
