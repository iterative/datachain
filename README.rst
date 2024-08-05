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
.. |Tests| image:: https://github.com/iterative/datachain/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/iterative/datachain/actions/workflows/tests.yml
   :alt: Tests

AI 🔗 DataChain
----------------

DataChain is a data-frame library designed for AI-specific scenarios. It helps ML and
AI engineers build a metadata layer on top of unstructured files and analyze data using
this layer.

📂 **Raw File Processing.** Process raw files (images, video, text, PDFs) directly from
   storage (S3, GCP, Azure, Local), version and update datasets.

🌟 **Metadata layer.** Build a metadata layer on top of files using structured sources
   like CSV, Parquet, and JSON files.

⭐ **Metadata enrichment.** Enhance the metadata layer with outputs from local ML
   model inferences and LLM calls.

🛠️ **Data Transformation.** Transform metadata using traditional methods like
   filtering, grouping, joining, and others.

🐍 **User-friendly interface.** Operate efficiently with familiar Python objects and
   object fields, eliminating the need for SQL.


.. code:: console

   $ pip install datachain


Data Structures
===============

DataChain introduces expressive data structures tailored for AI-specific workload:

- **Dataset:** Preserves the file-references and meta-information. Takes care of Python
  object serialization, dataset versioning and difference. Operations on dataset:

    - **Transformations:** traditional data-frame or SQL operations such as filtering,
    grouping, joining.
    - **Enrichments:** mapping, aggregating and generating using customer’s Python
    code. This is needed to work with ML inference and LLM calls.

- **Chain** is a sequence of operations on datasets. Chain executes operations in lazy
  mode - only when needed.

DataChain name comes from these major data structures: dataset and chaining.


What’s new in DataChain?
========================

The project combines multiple ideas from different areas in order to simplify AI
use-cases and at the same time to fit it into traditional data infrastructure.

- **Python-Native for AI.** Utilizes Python instead of SQL for data manipulation as the
  native language for AI. It’s powered by `Pydantic`_ data models.
- **Separation of CPU-GPU workloads.** Distinguishes CPU-heavy transformations (filter,
  group_by, join) from GPU heavy enrichments (ML-inference or LLM calls). That’s mostly
  needed for distributed computations.
- **Resuming data processing** (in development). Introduces idempotent operations,
  allowing data processing to resume from the last successful process file/record/batch
  if it fails due to issues like failed LLM calls, ML inference or file download.

Additional relatively new ideas:

- **Functional style data processing.** Using a functional/chaining approach to data
  processing rather than declarative SQL, inspired by R-dplyr and some Python libraries.
- **Data Versioning.** Treats raw files in cloud storage as the source of truth for data
  and implements data versioning, extending ideas from DVC (developed by the same team).


What DataChain is NOT?
======================

- **Not a database** (Postgres, MySQL). Instead, it uses databases under the hood:
  `SQLite`_ in open-source and ClickHouse and other data warehouses for the commercial
  version.
- **Not a data processing tool / data warehouse** (Spark, Snowflake, Big Query) since
  it delegates heavy data transformations to underlying data warehouses and focuses on
  AI specific data enrichments and orchestrating all the pieces together.


Quick Start
-----------

Data curation with a local model
=================================

We will evaluate chatbot dialogs stored as text files in Google Cloud Storage
- 50 files total in this example.
These dialogs involve users chatting with a bot while looking for better wireless plans.
Our goal is to identify the successful dialogs.

The data used in the examples is `publicly available`_. The sample code is designed to run on a local machine.

First, we'll show batch inference with a simple sentiment model using the `transformers` library:

.. code:: shell

    pip install transformers

The code below downloads files the cloud, and applies a user-defined function
to each one of them. All files with a positive sentiment
detected are then copied to the local directory.

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
    positive_chain.export_files("./output")

    print(f"{positive_chain.count()} files were exported")



13 files were exported

.. code:: shell

    $ ls output/datachain-demo/chatbot-KiT/
    15.txt 20.txt 24.txt 27.txt 28.txt 29.txt 33.txt 37.txt 38.txt 43.txt ...
    $ ls output/datachain-demo/chatbot-KiT/ | wc -l
    13


LLM judging chatbots
=============================

LLMs can work as efficient universal classifiers. In the example below,
we employ a free API from Mistral to judge the chatbot performance. Please get a free
Mistral API key at https://console.mistral.ai

.. code:: shell

    $ pip install mistralai
    $ export MISTRAL_API_KEY=_your_key_

DataChain can parallelize API calls; the free Mistral tier supports up to 4 requests at the same time.

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


With the instruction above, the Mistral model considers 31/50 files to hold the successful dialogues:

.. code:: shell

    $ ls output_mistral/datachain-demo/chatbot-KiT/
    1.txt  15.txt 18.txt 2.txt  22.txt 25.txt 28.txt 33.txt 37.txt 4.txt  41.txt ...
    $ ls output_mistral/datachain-demo/chatbot-KiT/ | wc -l
    31



Serializing Python-objects
==========================

LLM responses may contain valuable information for analytics – such as the number of tokens used, or the
model performance parameters.

Instead of extracting this information from the Mistral response data structure (class
`ChatCompletionResponse`), DataChain can serialize the entire LLM response to the internal DB:


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


Iterating over Python data structures
=============================================

In the previous examples, datasets were saved in the embedded database
(`SQLite`_ in folder `.datachain` of the working directory).
These datasets were automatically versioned, and can be accessed using
`DataChain.from_dataset("dataset_name")`.

Here is how to retrieve a saved dataset and iterate over the objects:

.. code:: py

    chain = DataChain.from_dataset("response")

    # Iterating one-by-one: support out-of-memory workflow
    for file, response in chain.limit(5).collect("file", "response"):
        # verify the collected Python objects
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

Some operations can run inside the DB without deserialization.
For instance, let's calculate the total cost of using the LLM APIs, assuming the Mixtral call costs $2 per 1M input tokens and $6 per 1M output tokens:

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
.. _publicly available: https://radar.kit.edu/radar/en/dataset/FdJmclKpjHzLfExE.ExpBot%2B-%2BA%2Bdataset%2Bof%2B79%2Bdialogs%2Bwith%2Ban%2Bexperimental%2Bcustomer%2Bservice%2Bchatbot
.. _SQLite: https://www.sqlite.org/
.. _Getting Started: https://datachain.dvc.ai/
.. |Flowchart| image:: https://github.com/iterative/datachain/blob/main/docs/assets/flowchart.png?raw=true
   :alt: DataChain FlowChart
