import os

from openai import OpenAI

from datachain.query import C, DatasetQuery

query = "What is the best way to do topic modeling?"

k_rag = 5
source_name = "Community"
system_name = "System"

ds = DatasetQuery(name="rag-query").order_by(C.distance).limit(k_rag)

df = ds.to_pandas().sort_values(by="distance", ascending=True)
content = "\n\n".join([f"{source_name}: {page}" for page in df["page"].to_list()])

openai_api_key = os.environ["OPENAI_API_KEY"]
assert openai_api_key.startswith("sk-")
client = OpenAI(api_key=openai_api_key)
del openai_api_key

msg = f"""
{system_name}: {query.strip('?')} according to the {source_name}?

{content}
"""

print(msg)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": msg.strip(),
        }
    ],
    model="gpt-4",
    temperature=0,
)
response = chat_completion.choices[0].message.content

print("\n\nLLM-RAG Response:\n")
print(response.strip())

ds.limit(k_rag)
