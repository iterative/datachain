import os

from langchain_community.embeddings import OpenAIEmbeddings
from scipy.spatial import distance

from datachain.query import C, DatasetQuery, udf
from datachain.sql.functions.array import cosine_distance
from datachain.sql.types import Float

query = "What is the best way to do topic modeling?"

openai_api_key = os.environ["OPENAI_API_KEY"]
assert openai_api_key.startswith("sk-")
openai = OpenAIEmbeddings(openai_api_key=openai_api_key)
del openai_api_key

(embed_query,) = openai.embed_documents([query])


@udf(
    params=("embed",),
    output={"distance": Float},
)
def distance_to_example(embed, embed0=embed_query):
    dist = distance.cosine(embed, embed0)
    return (dist,)


DatasetQuery(name="pdf-bib").mutate(distance=cosine_distance(C.embed, embed_query))
