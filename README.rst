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

🤖 AI-Driven Data Curation: Use local ML models, LLM APIs calls to enreach your data.

🚀 GenAI Dataset scale: Handle 10s of milions of files or file snippets.

🐍 Python-friendly: Python objects instead of JSON to represent annotations


Datachain enables parallel processing of multiple data files or samples.
It can chain different operations such as filtering, aggregation and merging datasets.
Resulting datasets can be saved, versioned, and extracted as files or converted
to a PyTorch data loader.

Datachain can serialize Python objects (via `Pydantic`_) to an embedded
`SQLite`_ databased. It efficiently deserializes Python object or run vectorized
analytical query in the DB without deserialization.

The typical use cases are data curation, LLM analytics and validation, image
segmentation, pose detection, and GenAI alignment.
DataChain excels at optimizing batch operations, such as parallelizing synchronous API
calls or leveraging heavy batch processing tasks.

.. code:: console

   $ pip install datachain

Quick Start
-----------

Basic file filtering
====================

Find files with text dialogs that contains keyword "Thank you".
The files should be downloaded and processed in parallel with data caching:

.. code:: py

    from datachain import DataChain, Column

    chain = (
       DataChain.from_storage("gs://datachain-demo/chatbot-KiT/",
                              object_name="file", type="text")
       .settings(parallel=12, cache=True)
       .map(is_good=lambda file: "thank you" in file.read().lower(),
            output={"is_good": bool})
       .save("file_response")
    )

    chain.filter(Column("is_good") == True).export_files("./output")

4 files were found:

.. code: shell

    $ ls output/datachain-demo/chatbot-KiT/
    15.txt 28.txt 29.txt 39.txt


LLM judging LLMs dialogues
==========================

Finding good dialogues using an LLM can be more efficient. In this example,
we use Mistral with a free API. Please install the package and get a free
Mistral API key at https://console.mistral.ai

.. code:: shell

    $ pip install mistralai
    $ export MISTRAL_API_KEY=_your_key_


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
       .map(is_good=eval_dialogue)
       .save("mistral_files")
    )

    chain.filter(Column("is_good") == True).export_files("./output_mistral")


With the current prompt, we found 31 files considered successful dialogues:

.. code: shell

    $ ls output_mistral/datachain-demo/chatbot-KiT/
    1.txt  15.txt 18.txt 2.txt  22.txt 25.txt 28.txt 33.txt 37.txt 4.txt  41.txt ...
    $ ls output_mistral/datachain-demo/chatbot-KiT/ | wc -l
    32

Note: Only 4 threads were used in this example `parallel=4` due to a limitation of
the free LLM service.


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

    def eval_dialogue(file: File) -> ChatCompletionResponse:
         client = MistralClient()
         return client.chat(
             model="open-mixtral-8x22b",
             messages=[ChatMessage(role="system", content=PROMPT),
                       ChatMessage(role="user", content=file.read())])

    chain = (
       DataChain.from_storage("gs://datachain-demo/chatbot-KiT/", object_name="file")
       .settings(parallel=4, cache=True)
       .map(response=eval_dialogue)
       .save("response")
    )

    good_files = (
        chain
        .map(status=lambda response: response.choices[0].message.content.lower())
        .filter(Column("status") == "success")
    )

    good_files.export_files("./output_responses")

Output:

.. code:: shell

    $ ls output_responses/datachain-demo/chatbot-KiT/
    1.txt  15.txt 18.txt 2.txt  22.txt 25.txt 28.txt 33.txt 37.txt 4.txt  41.txt ...
    $ ls output_responses/datachain-demo/chatbot-KiT/ | wc -l
    32


Deserializing Python-objects
============================


..code:: py

    chain = DataChain.from_dataset("response")

    # Iterating one-by-one: out of memory
    for file, response in chain.limit(5).collect("file", "response"):
        # You work with Python objects
        assert isinstance(response, ChatCompletionResponse)

        status = response.choices[0].message.content[:7]
        tokens = response.usage.total_tokens
        print(f"{file.get_uri()}: {status}, file size: {file.size}, tokens: {tokens}")

Output:

.code:: shell

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

..code:: py

    chain = DataChain.from_dataset("mistral_dataset")

    cost = chain.sum("response.usage.prompt_tokens")*0.000002 \
               + chain.sum("response.usage.completion_tokens")*0.000006
    print(f"Spent ${cost:.2f} on {chain.count()} calls")

Output:

..code:: shell

    Spent $0.08 on 50 calls


Passing data to pytorch for training
====================================

Chain results can be exported or passed directly to Pytorch dataloader.
For example, if we are interested in passing three columns to training,
the following Pytorch code will do it:

.. code:: py

      ds = train.select("file", "caption_choices", "label_ind").to_pytorch(
          transform=preprocess,
          tokenizer=clip.tokenize,
      )

      loader = DataLoader(ds, batch_size=2)
      optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
      train(loader, model, optimizer)


Tutorials
---------

* `Getting Started`_
* `Multimodal <examples/multimodal/clip_fine_tuning.ipynb>`_ (try in `Colab <https://colab.research.google.com/github/iterative/datachain/blob/main/examples/multimodal/clip_fine_tuning.ipynb>`__)
* `Computer Vision <examples/computer_vision/fashion_product_images/1-quick-start.ipynb>`_ (try in `Colab <https://colab.research.google.com/github/iterative/datachain/blob/main/examples/computer_vision/fashion_product_images/1-quick-start.ipynb>`__)

Contributions
-------------

Contributions are very welcome.
To learn more, see the `Contributor Guide`_.


Community and Support
---------------------

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
