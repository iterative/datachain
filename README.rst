|PyPI| |Python Version| |Codecov| |Tests|

.. |PyPI| image:: https://img.shields.io/pypi/v/datachain.svg
   :target: https://pypi.org/project/datachain/
   :alt: PyPI
.. |Python Version| image:: https://img.shields.io/pypi/pyversions/datachain
   :target: https://pypi.org/project/datachain
   :alt: Python Version
.. |Codecov| image:: https://codecov.io/gh/iterative/datachain/graph/badge.svg?token=byliXGGyGB
   :target: https://codecov.io/gh/iterative/datachain
   :alt: Codecov
.. |Tests| image:: https://github.com/iterative/datachain/workflows/Tests/badge.svg
   :target: https://github.com/iterative/datachain/actions?workflow=Tests
   :alt: Tests

AI 🔗 DataChain
----------------

DataChain is an open-source Python library for processing and curating unstructured
data at scale.

🤖 AI-Driven Data Curation: Use local ML models, LLM APIs calls to enrich your data.

🚀 GenAI Dataset scale: Handle 10s of milions of files or file snippets.

🐍 Python-friendly: Use strictly typed `Pydantic`_ objects instead of JSON.


To ensure efficiency, Datachain supports parallel processing, parallel data
downloads, and out-of-memory computing. It excels at optimizing batch operations.
While most GenAI tools focus on online applications and realtime, DataChain is designed
for offline data processing, data curation and ETL.

The typical use cases are Computer Vision data curation, LLM analytics
and validation.


.. code:: console

   $ pip install datachain

|Flowchart|

Quick Start
-----------

Basic evaluation
================

We will evaluate chatbot dialogs stored as text files in Google Cloud Storage
- 50 files total in the example.
These dialogs involve users looking for better wireless plans chatting with bot.
Our goal is to identify successful dialogs.

The data used in the examples is publicly available. Please feel free to run this code.

First, we'll use a simple sentiment analysis model. Please install transformers.

.. code:: shell

    pip install transformers

The code below downloads files the cloud, applies function
`is_positive_dialogue_ending()` to each. All files with a positive sentiment
are copied to local directory `output/`.

.. code:: py

    from transformers import pipeline
    from datachain import DataChain, Column

    classifier = pipeline("sentiment-analysis", device="cpu",
                    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

    def is_positive_dialogue_ending(file) -> bool:
        dialogue_ending = file.read()[-512:]
        return classifier(dialogue_ending)[0]["label"] == "POSITIVE"

    chain = (
       DataChain.from_storage("gs://datachain-demo/chatbot-KiT/",
                              object_name="file", type="text")
       .settings(parallel=8, cache=True)
       .map(is_positive=is_positive_dialogue_ending)
       .save("file_response")
    )

    positive_chain = chain.filter(Column("is_positive") == True)
    positive_chain.export_files("./output1")

    print(f"{positive_chain.count()} files were exported")



13 files were exported

.. code:: shell

    $ ls output/datachain-demo/chatbot-KiT/
    15.txt 20.txt 24.txt 27.txt 28.txt 29.txt 33.txt 37.txt 38.txt 43.txt ...
    $ ls output/datachain-demo/chatbot-KiT/ | wc -l
    13


LLM judging LLMs dialogs
==========================

Finding good dialogs using an LLM can be more efficient. In this example,
we use Mistral with a free API. Please install the package and get a free
Mistral API key at https://console.mistral.ai

.. code:: shell

    $ pip install mistralai
    $ export MISTRAL_API_KEY=_your_key_

Below is a similar code example, but this time using an LLM to evaluate the dialogs.
Note, only 4 threads were used in this example `parallel=4` due to a limitation of
the free LLM service.

.. code:: py

    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage
    from datachain import File, DataChain, Column

    PROMPT = "Was this dialog successful? Answer in a single word: Success or Failure."

    def eval_dialogue(file: File) -> bool:
         client = MistralClient()
         response = client.chat(
             model="open-mixtral-8x22b",
             messages=[ChatMessage(role="system", content=PROMPT),
                       ChatMessage(role="user", content=file.read())])
         result = response.choices[0].message.content
         return result.lower().startswith("success")

    chain = (
       DataChain.from_storage("gs://datachain-demo/chatbot-KiT/", object_name="file")
       .settings(parallel=4, cache=True)
       .map(is_success=eval_dialogue)
       .save("mistral_files")
    )

    successful_chain = chain.filter(Column("is_success") == True)
    successful_chain.export_files("./output_mistral")

    print(f"{successful_chain.count()} files were exported")


With the current prompt, we found 31 files considered successful dialogs:

.. code:: shell

    $ ls output_mistral/datachain-demo/chatbot-KiT/
    1.txt  15.txt 18.txt 2.txt  22.txt 25.txt 28.txt 33.txt 37.txt 4.txt  41.txt ...
    $ ls output_mistral/datachain-demo/chatbot-KiT/ | wc -l
    31



Serializing Python-objects
==========================

LLM responses contain valuable information for analytics, such as tokens used and the
model. Preserving this information can be beneficial.

Instead of extracting this information from the Mistral data structure (class
`ChatCompletionResponse`), we serialize the entire Python object to the internal DB.


.. code:: py

    from mistralai.client import MistralClient
    from mistralai.models.chat_completion import ChatMessage, ChatCompletionResponse
    from datachain import File, DataChain, Column

    PROMPT = "Was this dialog successful? Answer in a single word: Success or Failure."

    def eval_dialog(file: File) -> ChatCompletionResponse:
         client = MistralClient()
         return client.chat(
             model="open-mixtral-8x22b",
             messages=[ChatMessage(role="system", content=PROMPT),
                       ChatMessage(role="user", content=file.read())])

    chain = (
       DataChain.from_storage("gs://datachain-demo/chatbot-KiT/", object_name="file")
       .settings(parallel=4, cache=True)
       .map(response=eval_dialog)
       .map(status=lambda response: response.choices[0].message.content.lower()[:7])
       .save("response")
    )

    chain.select("file.name", "status", "response.usage").show(5)

    success_rate = chain.filter(Column("status") == "success").count() / chain.count()
    print(f"{100*success_rate:.1f}% dialogs were successful")

Output:

.. code:: shell

         file   status      response     response          response
         name                  usage        usage             usage
                       prompt_tokens total_tokens completion_tokens
    0   1.txt  success           547          548                 1
    1  10.txt  failure          3576         3578                 2
    2  11.txt  failure           626          628                 2
    3  12.txt  failure          1144         1182                38
    4  13.txt  success          1100         1101                 1

    [Limited by 5 rows]
    64.0% dialogs were successful


Complex Python data structures
=============================================

In the previous examples, a few dataset were saved in the embedded database
(`SQLite`_ in directory `.datachain`).
These datasets are versioned, and can be accessed using
`DataChain.from_dataset("dataset_name")`.

.. code:: py

    chain = DataChain.from_dataset("response")

    # Iterating one-by-one: out of memory
    for file, response in chain.limit(5).collect("file", "response"):
        # You work with Python objects
        assert isinstance(response, ChatCompletionResponse)

        status = response.choices[0].message.content[:7]
        tokens = response.usage.total_tokens
        print(f"{file.get_uri()}: {status}, file size: {file.size}, tokens: {tokens}")

Output:

.. code:: shell

    gs://datachain-demo/chatbot-KiT/1.txt: Success, file size: 1776, tokens: 548
    gs://datachain-demo/chatbot-KiT/10.txt: Failure, file size: 11576, tokens: 3578
    gs://datachain-demo/chatbot-KiT/11.txt: Failure, file size: 2045, tokens: 628
    gs://datachain-demo/chatbot-KiT/12.txt: Failure, file size: 3833, tokens: 1207
    gs://datachain-demo/chatbot-KiT/13.txt: Success, file size: 3657, tokens: 1101


Vectorized analytics over Python objects
========================================

Some operations can be efficiently run inside the DB without deserializing Python objects.
Let's calculate the cost of using LLM APIs in a vectorized way.
Mistral calls cost $2 per 1M input tokens and $6 per 1M output tokens:

.. code:: py

    chain = DataChain.from_dataset("mistral_dataset")

    cost = chain.sum("response.usage.prompt_tokens")*0.000002 \
               + chain.sum("response.usage.completion_tokens")*0.000006
    print(f"Spent ${cost:.2f} on {chain.count()} calls")

Output:

.. code:: shell

    Spent $0.08 on 50 calls


PyTorch data loader
===================

Chain results can be exported or passed directly to PyTorch dataloader.
For example, if we are interested in passing image and a label based on file
name suffix, the following code will do it:

.. code:: py

    from torch.utils.data import DataLoader
    from transformers import CLIPProcessor

    from datachain import C, DataChain

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    chain = (
        DataChain.from_storage("gs://datachain-demo/dogs-and-cats/", type="image")
        .map(label=lambda name: name.split(".")[0], params=["file.name"])
        .select("file", "label").to_pytorch(
            transform=processor.image_processor,
            tokenizer=processor.tokenizer,
        )
    )
    loader = DataLoader(chain, batch_size=1)


Tutorials
---------

* `Getting Started`_
* `Multimodal <examples/multimodal/clip_fine_tuning.ipynb>`_ (try in `Colab <https://colab.research.google.com/github/iterative/datachain/blob/main/examples/multimodal/clip_fine_tuning.ipynb>`__)

Contributions
-------------

Contributions are very welcome.
To learn more, see the `Contributor Guide`_.


Community and Support
---------------------

* `Docs <https://datachain.dvc.ai/>`_
* `File an issue`_ if you encounter any problems
* `Discord Chat <https://dvc.org/chat>`_
* `Email <mailto:support@dvc.org>`_
* `Twitter <https://twitter.com/DVCorg>`_


.. _PyPI: https://pypi.org/
.. _file an issue: https://github.com/iterative/datachain/issues
.. github-only
.. _Contributor Guide: CONTRIBUTING.rst
.. _Pydantic: https://github.com/pydantic/pydantic
.. _SQLite: https://www.sqlite.org/
.. _Getting Started: https://datachain.dvc.ai/
.. |Flowchart| image:: https://github.com/iterative/datachain/docs/assets/datachain-flowchart.png?branch=main
   :alt: DataChain FlowChart
