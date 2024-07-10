import pandas as pd

from datachain.lib.claude import claude_processor
from datachain.lib.dc import C, DataChain

SOURCE = "gs://dvcx-datalakes/chatbot-public"
MODEL = "claude-3-opus-20240229"
PROMPT = """Summarise the dialog in a sentence"""


chain = (
    DataChain.from_storage(SOURCE, is_text=True)
    .filter(C.name.glob("*.txt"))
    .limit(5)
    .map(claude=claude_processor(prompt=PROMPT, model=MODEL))
)

df = chain.to_pandas()

with pd.option_context("display.max_columns", None):
    print(df)
