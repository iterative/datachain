================
|logo| DataChain
================

|PyPI| |Python Version| |Codecov| |Tests|

.. |logo| image:: docs/assets/datachain.svg
   :height: 24
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

DataChain is a Python-based AI-data warehouse for transforming and analyzing unstructured
data like images, audio, videos, text and PDFs. It integrates with external storage
(e.g. S3) to process data efficiently without data duplication and manages metadata
in an internal database for easy and efficient querying.


Use Cases
=========

1. **ETL.** Pythonic framework for describing and running unstructured data transformations
   and enrichments, applying models to data, including LLMs.
2. **Analytics.** DataChain dataset is a table that combines all the information about data
   objects in one place + it provides dataframe-like API and vectorized engine to do analytics
   on these tables at scale.
3. **Versioning.** DataChain doesn't store, require moving or copying data (unlike DVC).
   Perfect use case is a bucket with thousands or millions of images, videos, audio, PDFs.

Getting Started
===============

Visit `Quick Start <https://docs.datachain.ai/quick-start>`_ and `Docs <https://docs.datachain.ai/>`_
to get started with `DataChain` and learn more.

.. code:: bash

        pip install datachain


Example: download subset of files based on metadata
---------------------------------------------------

Sometimes users only need to download a specific subset of files from cloud storage,
rather than the entire dataset.
For example, you could use a JSON file's metadata to download just cat images with
high confidence scores.


.. code:: py

    from datachain import Column, DataChain

    meta = DataChain.from_json("gs://datachain-demo/dogs-and-cats/*json", object_name="meta", anon=True)
    images = DataChain.from_storage("gs://datachain-demo/dogs-and-cats/*jpg", anon=True)

    images_id = images.map(id=lambda file: file.path.split('.')[-2])
    annotated = images_id.merge(meta, on="id", right_on="meta.id")

    likely_cats = annotated.filter((Column("meta.inference.confidence") > 0.93) \
                                   & (Column("meta.inference.class_") == "cat"))
    likely_cats.export_files("high-confidence-cats/", signal="file")


Example: LLM based text-file evaluation
---------------------------------------

In this example, we evaluate chatbot conversations stored in text files
using LLM based evaluation.

.. code:: shell

    $ pip install mistralai # Requires version >=1.0.0
    $ export MISTRAL_API_KEY=_your_key_

Python code:

.. code:: py

    from mistralai import Mistral
    from datachain import File, DataChain, Column

    PROMPT = "Was this dialog successful? Answer in a single word: Success or Failure."

    def eval_dialogue(file: File) -> bool:
         client = Mistral()
         response = client.chat.complete(
             model="open-mixtral-8x22b",
             messages=[{"role": "system", "content": PROMPT},
                       {"role": "user", "content": file.read()}])
         result = response.choices[0].message.content
         return result.lower().startswith("success")

    chain = (
       DataChain.from_storage("gs://datachain-demo/chatbot-KiT/", object_name="file", anon=True)
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


Key Features
============

📂 **Multimodal Dataset Versioning.**
   - Version unstructured data without moving or creating data copies, by supporting
     references to S3, GCP, Azure, and local file systems.
   - Multimodal data support: images, video, text, PDFs, JSONs, CSVs, parquet, etc.
   - Unite files and metadata together into persistent, versioned, columnar datasets.

🐍 **Python-friendly.**
   - Operate on Python objects and object fields: float scores, strings, matrixes,
     LLM response objects.
   - Run Python code in a high-scale, terabytes size datasets, with built-in
     parallelization and memory-efficient computing — no SQL or Spark required.

🧠 **Data Enrichment and Processing.**
   - Generate metadata using local AI models and LLM APIs.
   - Filter, join, and group datasets by metadata. Search by vector embeddings.
   - High-performance vectorized operations on Python objects: sum, count, avg, etc.
   - Pass datasets to Pytorch and Tensorflow, or export them back into storage.


Contributing
============

Contributions are very welcome. To learn more, see the `Contributor Guide`_.


Community and Support
=====================

* `Docs <https://docs.datachain.ai/>`_
* `File an issue`_ if you encounter any problems
* `Discord Chat <https://dvc.org/chat>`_
* `Email <mailto:support@dvc.org>`_
* `Twitter <https://twitter.com/DVCorg>`_


DataChain Studio Platform
=========================

`DataChain Studio`_ is a proprietary solution for teams that offers:

- **Centralized dataset registry** to manage data, code and
  dependencies in one place.
- **Data Lineage** for data sources as well as derivative dataset.
- **UI for Multimodal Data** like images, videos, and PDFs.
- **Scalable Compute** to handle large datasets (100M+ files) and in-house
  AI model inference.
- **Access control** including SSO and team based collaboration.

.. _PyPI: https://pypi.org/
.. _file an issue: https://github.com/iterative/datachain/issues
.. github-only
.. _Contributor Guide: https://docs.datachain.ai/contributing
.. _Pydantic: https://github.com/pydantic/pydantic
.. _publicly available: https://radar.kit.edu/radar/en/dataset/FdJmclKpjHzLfExE.ExpBot%2B-%2BA%2Bdataset%2Bof%2B79%2Bdialogs%2Bwith%2Ban%2Bexperimental%2Bcustomer%2Bservice%2Bchatbot
.. _SQLite: https://www.sqlite.org/
.. _Getting Started: https://docs.datachain.ai/
.. _DataChain Studio: https://studio.datachain.ai/
