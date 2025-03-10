---
title: Examples
---

# Examples

## DataChain Basics

!!! example "DataChain Basics"

    Datachain is built by composing wrangling operations.

    For example, let us consider the New Yorker Cartoon caption contest dataset, where cartoons are matched against the potential titles. Let us imagine we want to augment this dataset with synthetic scene descriptions coming from an AI model. The below code takes images from the cloud, and applies PaliGemma model to caption the first five of them and put the results in the column “scene”:

    ```python
    from datachain import Column, DataChain, File # (1)!
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration # (2)!

    images = DataChain.from_storage("gs://datachain-demo/newyorker_caption_contest/images", type="image")

    model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-mix-224")
    processor = AutoProcessor.from_pretrained("google/paligemma-3b-mix-224")

    def process(file: File) -> str:
      image=file.read().convert("RGB")
      inputs = processor(text="caption", images=image, return_tensors="pt")
      generate_ids = model.generate(**inputs, max_new_tokens=100)
      return processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    chain = (
          images.limit(5)
          .settings(cache=True)
          .map(scene=lambda file: process(file), output = str)
          .save()
    )
    ```

    1. `pip install datachain`
    2. `pip install transformers`

    Here is how we can view the results in a plot:

    ```python
    import matplotlib.pyplot as plt
    import re
    from textwrap import wrap

    def trim_text(text):
        match = re.search(r'[A-Z][^.]*\.', text)
        return match.group(0) if match else ''

    images = chain.collect("file")
    captions = chain.collect("scene")
    _ , axes = plt.subplots(1, len(captions), figsize=(15, 5))

    for ax, img, caption in zip(axes, images, captions):
        ax.imshow(img.read(),cmap='gray')
        ax.axis('off')
        wrapped_caption = "\n".join(wrap(trim_text(caption), 30))
        ax.set_title(wrapped_caption, fontsize=6)

    plt.show()
    ```

    ![Untitled](assets/captioned_cartoons.png)

If interested to see more examples, please check out the [tutorials](tutorials.md).

### Handling Python objects

In addition to storing primitive Python data types like strings, DataChain is also capable of using data models.

For example, most LLMs return objects that carry additional fields. If provider offers a Pydantic model for their LLM, Datachain can use it as a schema.

In the below example, we are calling a Mixtral 8x22b model to judge the “service chatbot” dataset from [Karlsruhe Institute of Technology](https://radar.kit.edu/radar/en/dataset/FdJmclKpjHzLfExE.ExpBot%2B-%2BA%2Bdataset%2Bof%2B79%2Bdialogs%2Bwith%2Ban%2Bexperimental%2Bcustomer%2Bservice%2Bchatbot) and saving the results as Mistral’s *ChatCompletionResponse* objects:

```python
# pip install mistralai
# this example requires a free Mistral API key, get yours at https://console.mistral.ai
# $ export MISTRAL_API_KEY='your key'

import os
from datachain import Column, DataChain, DataModel, Feature
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from mistralai.models.chat_completion import ChatCompletionResponse as MistralModel

prompt = "Was this dialog successful? Describe the 'result' as 'Yes' or 'No' in a short JSON"
api_key = os.environ["MISTRAL_API_KEY"]

## register the data model ###
DataModel.register(MistralModel)

chain = (
    DataChain
    .from_storage("gs://datachain-demo/chatbot-KiT/", type="text")
    .filter(Column("file.name").glob("*.txt"))
    .limit(5)
    .settings(parallel=4, cache=True)
    .map(
       mistral=lambda file: MistralClient(api_key=api_key).chat(
                                          model="open-mixtral-8x22b",
                                          response_format={"type": "json_object"},
                                          messages= [
                                             ChatMessage(role="system", content=f"{prompt}"),
                                             ChatMessage(role="user", content=f"{file.read()}")
                                          ]
                            ),
       output=MistralModel
    )
    .save("dialog-rating")
)

**iter = chain.collect("mistral")
**print(*map(lambda chat_response: chat_response.choices[0].message.content, iter))
```

```
{"result": "Yes"} {"result": "No"} {"result": "No"} {"result": "Yes"}
```

If you are interested in more LLM evaluation examples for DataChain, please follow this tutorial:

[https://github.com/iterative/datachain-examples/blob/main/llm/llm_chatbot_evaluation.ipynb](https://github.com/iterative/datachain-examples/blob/main/llm/llm_chatbot_evaluation.ipynb) [Google Colab](https://colab.research.google.com/github/iterative/datachain-examples/blob/main/llm/llm_chatbot_evaluation.ipynb)

### Vectorized analytics

Datachain internally represents datasets as tables, so analytical queries on the chain are automatically vectorized:

```python
# continued from the previous example

mistral_cost = chain.sum("mistral.usage.prompt_tokens")*0.000002 + \
               chain.sum("mistral.usage.completion_tokens")*0.000006

print(f"The cost of {chain.count()} calls to Mixtral 8x22b : ${mistral_cost:.4f}")
```

```
The cost of 5 calls to Mixtral 8x22b : $0.0142
```

### Dataset persistence

The “save” operation makes chain dataset persistent in the current (working) directory of the query. A hidden folder `.datachain/` holds the records. A persistent dataset can be accessed later to start a derivative chain:

```python
DataChain.from_dataset("rating").limit(2).save("dialog-rating")
```

Persistent datasets are immutable and automatically versioned. Here is how to access the dataset registry:

```python
mydatasets = DataChain.datasets()
for ds in mydatasets.collect("dataset"):
    print(f"{ds.name}@v{ds.version}")

```

```
Processed: 1 rows [00:00, 777.59 rows/s]
Generated: 14 rows [00:00, 11279.34 rows/s]
dialog-rating@v1
dialog-rating@v2
```

By default, when a saved dataset is loaded, the latest version is fetched but another version can be requested:

```python
ds = DataChain.from_dataset("dialog-rating", version = 1)
```

### Chain execution, optimization and parallelism

Datachain avoids redundant operations. Execution is triggered only when a downstream operation requests the processed results. However, it would be inefficient to run, say, LLM queries again every time you just want to collect several objects from the chain.

“Save” operation nails execution results and automatically refers to them every time the downstream functions ask for data. Saving without an explicit name generates an auto-named dataset which serves the same purpose.

Datachain natively supports parallelism in execution. If an API or a local model supports parallel requests, the `settings` operator can split the load across multiple workers (see the [code example above](#handling-python-objects))

### Reading external metadata

It is common for AI data to come with pre-computed metadata (annotations, classes, etc).

DataChain library understands common annotation formats (JSON, CSV, webdataset and parquet), and can unite data samples from storage with side-loaded metadata. The schema for metadata can be set explicitly or be inferred.

Here is an example of reading a simple CSV file where schema is heuristically derived from the header:

```python
from datachain import DataChain

uri="gs://datachain-demo/chatbot-csv/"
csv_dataset = DataChain.from_csv(uri)

print(csv_dataset.to_pandas())
```

Reading metadata from JSON format is a more complicated scenario because a JSON-annotated dataset typically references data samples in blocks within JSON files.

Here is an example from MS COCO “captions” JSON which employs separate sections for image meta and captions:

```json
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
```

Note how complicated the setup is. Every image is references by the name, and the metadata for this file is keyed by the “id” field. This same field is references later in the “annotations” array, which is present in JSON files describing captions and the detected instances. The categories for the instances are stored in the “categories” array.

However, Datachain can easily parse the entire COCO structure via several reading and merging operators:

```python

from datachain import Column, DataChain

images_uri="gs://datachain-demo/coco2017/images/val/"
captions_uri="gs://datachain-demo/coco2017/annotations/captions_val2017.json"

images = DataChain.from_storage(images_uri)
meta = DataChain.from_json(captions_uri, jmespath = "images")
captions = DataChain.from_json(captions_uri, jmespath = "annotations")

images_meta = images.merge(meta, on="file.name", right_on="images.file_name")
captioned_images = images_meta.merge(captions, on="images.id", right_on="annotations.image_id")
```

The resulting dataset has image entries as files decorated with all the metadata and captions:

```python
images_with_dogs = captioned_images.filter(Column("annotations.caption").glob("*dog*"))
images_with_dogs.select("annotations", "file.name").show()
```

```
   captions captions                                           captions              file
   image_id       id                                            caption              name
0     17029   778902         a dog jumping to catch a frisbee in a yard  000000017029.jpg
1     17029   779838   A dog jumping to catch a red frisbee in a garden  000000017029.jpg
2     17029   781941  The dog is catching the Frisbee in mid air in ...  000000017029.jpg
3     17029   782283      A dog catches a frisbee outside in the grass.  000000017029.jpg
4     17029   783543              A dog leaping to catch a red frisbee.  000000017029.jpg
5     18193   278544  A woman in green sweater eating a hotdog by ca...  000000018193.jpg
...

[Limited by 20 rows]
```
For in-depth review of working with JSON metadata, please follow this tutorial:

[GitHub](https://github.com/iterative/datachain-examples/blob/main/formats/json-metadata-tutorial.ipynb) or [Google Colab](https://colab.research.google.com/github/iterative/datachain-examples/blob/main/formats/json-metadata-tutorial.ipynb)

### Passing data to training

Chain results can be exported or passed directly to Pytorch dataloader. For example, if we are interested in passing three columns to training, the following Pytorch code will do it:

```python

ds = train.select("file", "caption_choices", "label_ind").to_pytorch(
    transform=preprocess,
    tokenizer=clip.tokenize,
)

loader = DataLoader(ds, batch_size=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
train(loader, model, optimizer)
```

See a larger example for CLIP fine-tuning here:

[GitHub](https://github.com/iterative/datachain-examples/blob/main/multimodal/clip_fine_tuning.ipynb) or [Google Colab](https://colab.research.google.com/github/iterative/datachain-examples/blob/main/multimodal/clip_fine_tuning.ipynb)
