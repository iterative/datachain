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
   objects in one place + it provides dataframe-like API and vecrorized engine to do analytics
   on these tables at scale.
3. **Versioning.** DataChain doesn't store, require moving or copying data (unlike DVC).
   Perfect use case is a bucket with thousands or millions of images, videos, audio, PDFs.

Getting Started
===============

Visit `Quick Start <https://docs.datachain.ai/quick-start>`_ and `Docs <https://docs.datachain.ai/>`_
to get started with `DataChain` and learn more.

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

- **Centralized dataset registry** to manage data, code and dependency
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
