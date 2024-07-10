|PyPI| |Python Version| |Codecov| |Tests| |License|

.. |PyPI| image:: https://img.shields.io/pypi/v/datachain.svg
   :target: https://pypi.org/project/datachain/
   :alt: PyPI
.. |Python Version| image:: https://img.shields.io/pypi/pyversions/datachain
   :target: https://pypi.org/project/datachain
   :alt: Python Version
.. |Codecov| image:: https://codecov.io/gh/iterative/dvcx/branch/main/graph/badge.svg?token=VSCP2T9R5X
   :target: https://app.codecov.io/gh/iterative/dvcx
   :alt: Codecov
.. |Tests| image:: https://github.com/iterative/dvcx/workflows/Tests/badge.svg
   :target: https://github.com/iterative/dvcx/actions?workflow=Tests
   :alt: Tests
.. |License| image:: https://img.shields.io/pypi/l/datachain
   :target: https://opensource.org/licenses/Apache-2.0
   :alt: License

AI 🔗 DataChain
----------------

DataChain is an open-source Python data processing library for wrangling unstructured AI data at scale.

It enables batch LLM API calls and local language and vision AI model inferences to run in parallel over many samples as chained operations resolving to table-like datasets. These datasets can be saved, versioned, and sent directly to PyTorch and TensorFlow for training. DataChain employs rigorous `Pydantic`_ data structures, promoting better data processing practices and enabling vectorized analytical operations normally found in databases.

The DataChain fills the gap between dataframe libraries, data warehouses, and Python-based multimodal AI applications. Our primary use cases include massive data curation, LLM analytics and validation, batch image segmentation and pose detection, GenAI data alignment, etc.

.. code:: console

   $ pip install datachain

Basic operation
---------------

DataChain is built by composing wrangling operations.

For example, it can be instructed to read files from the cloud, map them onto a modern AI service returning a Python object, parallelize API calls, save the result as a dataset, and export a column:

.. code:: py

         import os
         import datachain as dc

         from anthropic.types.message import Message
         ClaudeModel = dc.pydantic_to_feature(Message)
         PROMPT = "summarize this book in less than 200 words"
         service = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
         source = "gs://datachain-demo/mybooks/"

         chain = dc.DataChain(source)                          \
                       .filter(File.name.glob("*.txt"))        \
                       .settings(parallel=4)                   \
                       .map(                                   \
         		              claude = lambda file:                                         \
         				        ClaudeModel(**service.messages.create(                        \
                                                  model="claude-3-haiku-20240307",         \
                                                  system=PROMPT,                           \
                                                  messages=[{"role": "user",               \
                                                             "content": file.get_value()}] \
         								),  \
         							).model_dump()  \
         							)               \
         							.save("mydataset")

         dc.DataChain("mydataset").export("./", "claude.response") # export summaries

Dataset persistence
-------------------

In the example above, the chain resolves to a saved dataset  “mydataset”.  DataChain datasets are immutable and versioned. A saved dataset version can be used as a data source:

.. code:: py

   ds = dc.DataChain("mydataset", version = 1)

Note that DataChain represents file samples as pointers into their respective storage locations. This means a newly created dataset version does not duplicate files in storage, and storage remains the single source of truth for the original samples

Vectorized analytics
---------------------
Since datasets are internally represented as tables, analytical queries can be vectorized:

.. code:: py

         rate = ds.filter(chain.response == "Success").count() / chain.count() # ??
         print(f"API class success rate: {100*rate:.2f}%")
         >> 74.68%

         price_input = 0.25
         price_output = 1.25
         price=(ds.sum(C.claude.usage.input_tokens)*price_input \
                + ds.sum(C.claude.usage.output_tokens)*price_output)/1_000_000
         print(f"Cost of API calls: ${price:.2f}")
         >> Cost of API calls: $1.42


Importing metadata
------------------------

It is common for AI data to come together with metadata (annotations, classes, etc).
DataChain understands many metadata formats, and can connect data samples in storage with external metadata (e.g. CSV columns) to form a single dataset:

.. code:: py

         from dc import parse_csv

         files = dc.DataChain("gs://datachain-demo/myimages/")
         metadata = dc.DataChain("gs://datachain-demo/myimagesmetadata.csv") \
                        .gen(meta=parse_csv)  # TBD, also dependent on dropping file
         dataset = chain1.merge(chain2, on = "file.name", right_on="name"])

         print(dataset.select("file.name", "class", "prob").limit(5).to_pandas())
         ....
         ....
         ....
         ....
         ....

Nested annotations (like JSON) can be unrolled into rows and columns in the way that best fits the application. For example, the MS COCO dataset includes JSON annotations detailing segmentations. To build a dataset consisting of all segmented objects in all COCO images:

.. code:: py

      image_files = dc.DataChain("gs://datachain-demo/coco/images/")
      image_meta  = dc.DataChain("gs://datachain-demo/coco.json")  \
                     .gen(meta=parse_json, key="images")       # list of images
      images = image_files.merge(image_meta, on = "file.name", right_on="file_name")
      objects_meta = dc.DataChain("gs://datachain-demo/coco.json") \
                     .gen(meta=parse_json, key="annotations")  # annotated objects

      objects = image.full_merge(objects_meta, on = "id", right_on = "image_id")

Generating metadata
---------------------

A typical step in data curation is to create features from data samples for future selection. DataChain represents the newly created metadata as columns, which makes it easy to create new features and filter on them:

.. code:: py

      from fashion_clip.fashion_clip import FashionCLIP
      from sqlalchemy import JSON
      from tabulate import tabulate

      from datachain.lib.param import Image
      from datachain.query import C, DatasetQuery, udf


      @udf(
          params=(Image(),),
          output={"fclip": JSON},
          method="fashion_clip",
          batch=10,
      )
      class MyFashionClip:
          def __init__(self):
              self.fclip = FashionCLIP("fashion-clip")

          def fashion_clip(self, inputs):
              embeddings = self.fclip.encode_images(
                  [input[0] for input in inputs], batch_size=1
              )
              return [(json.dumps(emb),) for emb in embeddings.tolist()]

      chain = dc.DataChain("gs://datachain-demo/zalando/images/").filter(
              C.name.glob("*.jpg")
          ).limit(5).add_signals(MyFashionClip).save("zalando_hd_emb")

      test_image = "cs://datachain-demo/zalando/test/banner.jpg"
      test_embedding = MyFashionClip.fashion_clip.encode_images(Image(test_image))

      best_matches = chain.filter(similarity_search(test_embeding)).limit(5)

      print best_matches.to_result()


Delta updates
-------------

DataChain is capable of “delta updates” – that is, batch-processing only the newly added data samples. For example, let us copy some images into a local folder and run a chain to generate captions with a locally served captioning model from HuggingFace:

.. code:: console

      > mkdir demo-images/
      > datachain cp gs://datachain-demo/images/ /tmp/demo-images


.. code:: py

         import torch

         from datachain.lib.hf_image_to_text import LLaVAdescribe
         from datachain.query import C, DatasetQuery

         source = "/tmp/demo-images"

         if torch.cuda.is_available():
             device = "cuda"
         else:
             device = "cpu"

         if __name__ == "__main__":
             results = (
                 DatasetQuery(
                     source,
                     anon=True,
                 )
                 .filter(C.name.glob("*.jpg"))
                 .add_signals(
                     LLaVAdescribe(
                         device=device,
                         model=model,
                     ),
                     parallel=False,
                 )
                 .save("annotated-images")
             )

Now let us add few more more images to the same folder:

.. code:: console

         > datachain cp gs://datachain-demo/extra-images/ /tmp/demo-images

and calculate updates only for the delta:

.. code:: py

      processed = dc.DataChain("annotated-images")
      delta = dc.dataChain("/tmp/demo-images").subtract(processed)

Passing data to training
------------------------

Datasets can be exported to CSV or webdataset formats. However, a much better way to pass data to training which avoids data copies and re-sharding is  to wrap a DataChain dataset into a PyTorch class, and let the library take care of file downloads and caching under the hood:

.. code:: py

         ds = dc.DataChain("gs://datachain-demo/name-labeled/images/")
                        .filter(C.name.glob("*.jpg"))
                        .map(lambda name: (name[:3],), output={"label": str}, parallel=4)
             )

         train_loader = DataLoader(
                 ds.to_pytorch(
                     ImageReader(),
                     LabelReader("label", classes=CLASSES),
                     transform=transform,
                 ),
                 batch_size=16,
                 parallel=2,
             )

Tutorials
------------------

* `Computer Vision <examples/computer_vision/fashion_product_images/1-quick-start.ipynb>`_ (try in `Colab <https://colab.research.google.com/github/iterative/dvcx/blob/main/examples/computer_vision/fashion_product_images/1-quick-start.ipynb>`__)
* `Multimodal <examples/multimodal/clip_fine_tuning.ipynb>`_ (try in `Colab <https://colab.research.google.com/github/iterative/dvclive/blob/main/examples/multimodal/clip_fine_tuning.ipynb>`__)

💻  More examples
------------------

* Curating images to train a custom CLIP model without re-sharding the Webdataset files
* Batch-transforming and indexing images to create a searchable merchandise catalog
* Evaluating an LLM application at scale
* Ranking the LLM retrieval strategies
* Delta updates in batch processing

Contributions
--------------------

Contributions are very welcome.
To learn more, see the `Contributor Guide`_.


License
-------

Distributed under the terms of the `Apache 2.0 license`_,
*DataChain* is free and open source software.


Issues
------

If you encounter any problems,
please `file an issue`_ along with a detailed description.


.. _Apache 2.0 license: https://opensource.org/licenses/Apache-2.0
.. _PyPI: https://pypi.org/
.. _file an issue: https://github.com/iterative/dvcx/issues
.. _pip: https://pip.pypa.io/
.. github-only
.. _Contributor Guide: CONTRIBUTING.rst
.. _Pydantic: https://github.com/pydantic/pydantic
