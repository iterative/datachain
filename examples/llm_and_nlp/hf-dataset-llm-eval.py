from huggingface_hub import InferenceClient

from datachain import C, DataChain, DataModel

PROMPT = """
Was this dialog successful? Put result as a single word: Success or Failure.
Explain the reason in a few words.
"""


class DialogEval(DataModel):
    result: str
    reason: str


def eval_dialog(user_input: str, bot_response: str) -> DialogEval:
    client = InferenceClient("meta-llama/Llama-3.1-70B-Instruct")

    completion = client.chat_completion(
        messages=[
            {
                "role": "user",
                "content": f"{PROMPT}\n\nUser: {user_input}\nBot: {bot_response}",
            },
        ],
        response_format={"type": "json", "value": DialogEval.model_json_schema()},
    )

    message = completion.choices[0].message
    try:
        return DialogEval.model_validate_json(message.content)
    except ValueError:
        return DialogEval(result="Error", reason="Failed to parse response.")


# Run OpenAI in parallel for each example
# Get result as Pydantic model that DataChain can understand and serialize
# Save to HF as Parquet
(
    DataChain.from_csv(
        "hf://datasets/infinite-dataset-hub/MobilePlanAssistant/data.csv"
    )
    .settings(parallel=10)
    .map(response=eval_dialog)
    .to_parquet("hf://datasets/dvcorg/test-datachain-llm-eval/data.parquet")
)

# Read it back to filter and show
(
    DataChain.from_parquet(
        "hf://datasets/dvcorg/test-datachain-llm-eval/data.parquet", source=False
    )
    .filter(C("response.result") == "Failure")
    .show(3)
)
