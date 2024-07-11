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

AI 🔗 DataChain
----------------

DataChain is an open-source Python data processing library for wrangling unstructured AI data at scale.

Datachain enables multimodal API calls and local AI inferences to run in parallel over many samples as chained operations. The resulting datasets can be saved, versioned, and sent directly to PyTorch and TensorFlow for training. Datachain can persist features of Python objects returned by AI models, and enables vectorized analytical operations over them.

The typical use cases are data curation, LLM analytics and validation, image segmentation, pose detection, and GenAI alignment. Datachain is especially helpful if batch operations can be optimized – for instance, when synchronous API calls can be parallelized  or where an LLM API offers batch processing.

.. code:: console

   $ pip install datachain

Operation basics
----------------

DataChain is built by composing wrangling operations.

For example, let us consider a dataset from Karlsruhe Institute of Technology detailing dialogs between users and customer service chatbots. We can use the chain to read data from the cloud, map it onto the parallel API calls for LLM evaluation, and organize the output into a dataset :

.. code:: py

      # pip install mistralai
      # this example requires a free Mistral API key, get yours at https://console.mistral.ai
      # add the key to your shell environment: $ export MISTRAL_API_KEY= your key

      import os
      import pandas as pd
      from datachain.lib.feature import Feature
      from mistralai.client import MistralClient
      from mistralai.models.chat_completion import ChatMessage
      from datachain.lib.dc import Column, DataChain
      
      source = "gs://datachain-demo/chatbot-KiT/"
      PROMPT = "Was this bot dialog successful? Describe the 'result' as 'Yes' or 'No' in a short JSON"
      
      model = "mistral-large-latest"
      api_key = os.environ["MISTRAL_API_KEY"]
      
      chain = (
          DataChain.from_storage(source)
          .limit(5)
          .settings(cache=True, parallel=5)
          .map(
              mistral_response=lambda file: MistralClient(api_key=api_key)
              .chat(
                  model=model,
                  response_format={"type": "json_object"},
                  messages=[
                      ChatMessage(role="user", content=f"{PROMPT}: {file.get_value()}")
                  ],
              ).choices[0].message.content,
          )
          .save()
      )
      
      try:
         print(chain.select("mistral_response").results())
      except Exception as e:
         print(f"do you have the right Mistral API key? {e}")	
      
      -> 
      [('{"result": "Yes"}',), ('{"result": "No"}',), ... , ('{"result": "Yes"}',)]

Now we have parallel-processed an LLM API-based query over cloud data and persisted the results. 

Vectorized analytics 
--------------------

Datachain internally represents datasets as tables, so analytical queries on the chain are automatically vectorized:

.. code:: py

      failed_dialogs = chain.filter(Column("mistral_response") == '{"result": "No"}')
      success_rate = failed_dialogs.count() / chain.count() 
      print(f"Chatbot dialog success rate: {100*success_rate:.2f}%")
      
      -> 
      "40.00%" (results may vary)

Note that DataChain represents file samples as pointers into their respective storage locations. This means a newly created dataset version does not duplicate files in storage, and storage remains the single source of truth for the original samples

Handling Python objects 
-----------------------
In addition to storing primitive Python data types, chain is also capable of using data models.

For example, instead of collecting just a text response from Mistral API, we might be interested in more fields of the Mistral response object. For this task, we can define a Pydantic-like model and populate it from the API replies:

.. code:: py

      import os
      from datachain.lib.feature import Feature
      from datachain.lib.dc import Column, DataChain 
      from mistralai.client import MistralClient
      from mistralai.models.chat_completion import ChatMessage
      
      source = "gs://datachain-demo/chatbot-KiT/"     
      PROMPT = "Was this dialog successful? Describe the 'result' as 'Yes' or 'No' in a short JSON"
      
      model = "mistral-large-latest"
      api_key = os.environ["MISTRAL_API_KEY"]
      
      ## define the data model ###
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


      ## Populate model instances ###
      chain = (
          DataChain.from_storage(source)
          .limit(5)
          .settings(cache=True, parallel=5)
          .map(
              mistral_response=lambda file: MistralModel(
                  **MistralClient(api_key=api_key)
                  .chat(
                      model=model,
                      response_format={"type": "json_object"},
                      messages=[
                          ChatMessage(role="user", content=f"{PROMPT}: {file.get_value()}")
                      ],
                  ).dict()
              ),
              output=MistralModel,
          )
          .save("dialog-eval")
      )

After the chain execution, we can collect the objects:

.. code:: py

      responses = chain.collect_one("mistral_response")
      for object in responses:
         print(type(object))
         -> 
         <class '__main__.MistralModel'>
         <class '__main__.MistralModel'>
         <class '__main__.MistralModel'>
         <class '__main__.MistralModel'>
         <class '__main__.MistralModel'>

      print(responses[0].usage.prompt_tokens)
         -> 
         610

Dataset persistence
--------------------

The “save” operation makes chain dataset persistent in the current (working) directory of the query. A hidden folder .datachain/ holds the records. A persistent dataset can be accessed later to start a derivative chain:

.. code:: py

         DataChain.from_dataset("dialog-eval").limit(2).save("dialog-eval")

Persistent datasets are immutable and automatically versioned. Versions can be listed from shell:

.. code:: shell

      $ datachain ls-datasets      
      
      dialog-rate (v1)
      dialog-rate (v2)

By default, when a persistent dataset is loaded, the latest version is fetched but another version can be requested:

.. code:: py

      ds = DataChain.from_dataset("dialog-eval", version = 1) 

Chain optimization and execution
--------------------------------

Datachain avoids redundant operations. Execution is triggered only when a downstream operation requests the processed results. However, it would be inefficient to run, say, LLM queries again every time you just want to collect several objects. 

“Save” operation nails execution results and automatically refers to them every time the downstream functions ask for data. Saving without an explicit name generates an auto-named dataset which serves the same purpose.


Matching data with metadata
----------------------------
It is common for AI data to come with pre-computed metadata (annotations, classes, etc).

DataChain library understands common metadata formats (JSON, CSV and parquet), and can unite data samples from storage with side-loaded metadata. The schema for metadata can be set explicitly or be inferred.

Here is an example of reading a CSV file where schema is heuristically derived from the header:

.. code:: py

      from datachain.lib.dc import DataChain 
      
      uri="gs://datachain-demo/chatbot-csv/"  
      csv_dataset = DataChain.from_csv(uri)
      
      print(csv_dataset.to_pandas())

Reading metadata from JSON format is a more complicated scenario because a JSON-annotated dataset typically references data samples (e.g. images) in annotation arrays somewhere within JSON files.

Here is an example from MS COCO “captions” JSON which employs separate sections for image meta and captions:

.. code:: json


      {
        "images": [
          {
            "license": 4,
            "file_name": "000000397133.jpg",
            "coco_url": "http://images.cocodataset.org/val2017/000000397133.jpg",
            "height": 427,
            "width": 640,
            "date_captured": "2013-11-14 17:02:52",
            "flickr_url": "http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg",
            "id": 397133
          },
          ...
        ],
        "annotations": [
          {
            "image_id"	:	"179765",
            "id"	:	38,
            "caption"	:	"A black Honda motorcycle parked in front of a garage."
          },
          ...
        ],
        ...
      }

To deal with this layout, we can take the following steps:

1. Generate a dataset of raw image files from storage
2. Generate a meta-information dataset from the JSON section “images”
3. Join these datasets via the matching id keys

.. code:: python


   from datachain.lib.dc import DataChain 
   
   image_uri="gs://datachain-demo/coco2017/images/val/"
   coco_json="gs://datachain-demo/coco2017/annotations_captions" 
   
   images = DataChain.from_storage(image_uri)
   meta = DataChain.from_json(coco_json, jmespath = "images")
                   
   images_with_meta = images.merge(meta, on="file.name", right_on="images.file_name")                 
   
   
   print(images_with_meta.limit(1).results())


Passing data to training
------------------------

Chain results can be exported or passed directly to Pytorch dataloader. For example, if we are interested in passing three columns to training, the following Pytorch code will do it:

.. code:: py

      ds = train.select("file", "caption_choices", "label_ind").to_pytorch(
          transform=preprocess,
          tokenizer=clip.tokenize,
      )
      
      loader = DataLoader(ds, batch_size=2)
      optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
      train(loader, model, optimizer)

Tutorials
------------------

* `Multimodal <examples/multimodal/clip_fine_tuning.ipynb>`_ (try in `Colab <https://colab.research.google.com/github/iterative/dvclive/blob/main/examples/multimodal/clip_fine_tuning.ipynb>`__)

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
