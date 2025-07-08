import os

from huggingface_hub import InferenceClient
from requests import HTTPError

import datachain as dc

PROMPT = """
Was this dialog successful? Put result as a single word: Success or Failure.
Explain the reason in a few words.
"""


class DialogEval(dc.DataModel):
    result: str
    reason: str


# DataChain function to evaluate dialog.
# DataChain is using types for inputs, results to automatically infer schema.
def eval_dialog(
    client: InferenceClient,
    user_input: str,
    bot_response: str,
) -> DialogEval:
    try:
        completion = client.chat_completion(
            model="meta-llama/Llama-3.3-70B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": f"{PROMPT}\n\nUser: {user_input}\nBot: {bot_response}",
                },
            ],
            response_format={"type": "json", "value": DialogEval.model_json_schema()},
        )
    except HTTPError as e:
        return DialogEval(
            result="Error",
            reason=f"Error while interacting with the Hugging Face API. {e}",
        )

    message = completion.choices[0].message
    try:
        return DialogEval.model_validate_json(message.content)
    except ValueError:
        return DialogEval(result="Error", reason="Failed to parse response.")


# Run HF inference in parallel for each example.
# Get result as Pydantic model that DataChain can understand and serialize it.
# Save to HF as Parquet. Dataset can be previewed here:
# https://huggingface.co/datasets/dvcorg/test-datachain-llm-eval/viewer
(
    dc.read_csv(
        "hf://datasets/infinite-dataset-hub/MobilePlanAssistant/data.csv", source=False
    )
    .settings(parallel=True)
    .setup(
        client=lambda: InferenceClient(
            provider="hf-inference", api_key=os.environ["HF_TOKEN"]
        )
    )
    .map(response=eval_dialog)
    .to_parquet("hf://datasets/dvcorg/test-datachain-llm-eval/data.parquet")
)

# Read it back to filter and show.
# It restores the Pydantic model from Parquet under the hood.
(
    dc.read_parquet(
        "hf://datasets/dvcorg/test-datachain-llm-eval/data.parquet", source=False
    )
    .filter(dc.C("response.result") == "Failure")
    .show(3)
)
