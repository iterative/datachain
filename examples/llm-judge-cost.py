# TODO: check is setup() works
# TODO: check if collect_one() is renamed to collect()

import os
import json

from datachain.lib.feature import Feature
from datachain.lib.dc import Column, DataChain
#from datachain.lib.feature_utils import pydantic_to_feature

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from typing import Callable, Literal, Optional

import anthropic
from anthropic.types.message import Message


#source = "gs://datachain-demo/chatbot-KiT/"
PROMPT = "Was this dialog successful? Describe the 'result' as 'Yes' or 'No' in JSON format and print nothing else"

#mistral_model = "mistral-large-latest"
mistral_model = "open-mixtral-8x22b"
mistral_api_key = os.environ["MISTRAL_API_KEY"]

claude_model = "claude-3-5-sonnet-20240620"
claude_api_key = os.environ["ANTHROPIC_API_KEY"]

## define the Mistral data model ###
class Usage(Feature):
    prompt_tokens: int = 0
    completion_tokens: int = 0

class MyChatMessage(Feature):
    role: str = ""
    content: str = ""

class CompletionResponseChoice(Feature):
    message: MyChatMessage = MyChatMessage()

class MistralModel(Feature):
    id: str = ""
    choices: list[CompletionResponseChoice]
    usage: Usage = Usage()

### define the Claude data model ###
class UsageFr(Feature):
    input_tokens: int = 0
    output_tokens: int = 0


class TextBlockFr(Feature):
    text: str = ""
    type: str = "text"


class ClaudeMessage(Feature):
    id: str = ""
    content: list[TextBlockFr]
    type: str = "message"
    usage: UsageFr = UsageFr()


def mistral_api_response(mistral_client, content, prompt=PROMPT):
    response = mistral_client.chat(
                             model=mistral_model,
                             response_format={"type": "json_object"},
                             messages= [
                               ChatMessage(role="system", content=f"{prompt}"), 
                               ChatMessage(role="user", content=f"{content}") 
			     ]
               )
    return MistralModel(**response.dict())

def claude_api_response(claude_client, content, prompt=PROMPT):
    response = claude_client.messages.create(
                             model=claude_model,
                             max_tokens=1024,
                             system=prompt,
                             messages= [{"role":"user", "content": f"{content}"},]
               )
    return ClaudeMessage(**response.dict())

# Twitter GIF starts here

chain = (
    DataChain
    .from_storage("gs://datachain-demo/chatbot-KiT/")
    .settings(parallel=4, cache=True)
    .filter(Column("file.name").glob("*.txt"))
    #.setup(mistral_client = lambda: MistralClient(api_key=mistral_api_key))
    #.setup(claude_client = lambda: anthropic.Anthropic(api_key=claude_api_key))
    #.map(mistral=lambda file: mistral_api_response(mistral_client, file.get_value(), prompt=PROMPT), output=MistralModel)
    #.map(claude=lambda file: claude_api_response(claude_client, file.get_value(), prompt=PROMPT), output=ClaudeMessage)
    .map(mistral=lambda file: mistral_api_response(MistralClient(api_key=mistral_api_key), file.get_value(), prompt=PROMPT), output=MistralModel)
    .map(claude=lambda file: claude_api_response(anthropic.Anthropic(api_key=claude_api_key), file.get_value(), prompt=PROMPT), output=ClaudeMessage)
    .save("llm-claude-mistral")
)

# price plot
import matplotlib.pyplot as plt

m_input = 0.000002
m_output = 0.000006
c_input = 0.000003 
c_output = 0.000015

mistral_cost = chain.sum("mistral.usage.prompt_tokens")*m_input + chain.sum("mistral.usage.completion_tokens")*m_output
claude_cost = chain.sum("claude.usage.input_tokens")*c_input + chain.sum("claude.usage.output_tokens")*c_output

plt.bar(["Open Mixtral 8x22b", "Claude Sonnet-3.5"], [mistral_cost, claude_cost])
plt.title(f"Cost of {chain.count()}x LLM evaluations by Mistral vs Anthropic API, $")
plt.show()


# confusion matrix plot
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

new_chain = (
    DataChain.from_dataset("llm-claude-mistral")
    .map(mistral_rating=lambda mistral: json.loads( mistral.choices[0].message.content)["result"])
    .map(claude_rating=lambda claude: json.loads(claude.content[0].text)["result"])
    .save()
)

y_mistral = new_chain.collect_one("mistral_rating")
y_claude = new_chain.collect_one("claude_rating")

cm = confusion_matrix(y_claude, y_mistral)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
disp.plot(cmap=plt.cm.Blues)

plt.title('LLM judge confusion Matrix')
plt.xlabel('Mixtral 8x22b')
plt.ylabel('Claude Sonnet-3.5')

plt.show()


