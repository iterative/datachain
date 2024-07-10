import pandas as pd

from datachain.lib.claude import claude_processor
from datachain.lib.dc import C, DataChain
from datachain.sql.functions import path

SOURCE = "gs://dvcx-datalakes/chatbot-public"
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


chain = (
    DataChain.from_storage(SOURCE, is_text=True)
    .filter(C.name.glob("*.txt"))
    .limit(5)
    .agg(
        dialogues=lambda file: ["\n=====\n".join(f.get_value() for f in file)],
        output=str,
        partition_by=path.file_ext(C.name),
    )
    .map(claude=claude_processor(prompt=PROMPT, model=MODEL), params="dialogues")
    .map(
        res=lambda claude: [str(claude.content[0].text) if claude.content else ""],
        output=str,
    )
)

df = chain.to_pandas()

with pd.option_context("display.max_columns", None):
    print(df)
