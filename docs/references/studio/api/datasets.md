
### List Datasets

**Description**: Retrieve a list of datasets with optional prefix filtering.

**Endpoint**: `/datachain/datasets`

**Method**: `GET`

**Authentication**: Requires `DATASETS` scope from [`datachain auth login`](../../../commands/auth/login.md)

**Parameters**:

- `prefix` (query, optional): Filter datasets by name prefix
- `team_name` (query, required): Team identifier (automatically added by client)

**Example Request**:
```bash
curl -X GET "https://studio.datachain.ai/api/datachain/datasets?prefix=training&team_name=my-team" \
  -H "Authorization: token YOUR_STUDIO_TOKEN" \
  -H "Content-Type: application/json"
```

**Example Response**:
```json
[
  {
    "attrs": ["training"],
    "created_at": "2025-06-11T14:22:31.123456+00:00",
    "created_by_id": 2,
    "description": "Training dataset for computer vision",
    "id": 1853,
    "name": "training-images-v2",
    "project": {
      "created_at": "2025-06-20T04:34:24.083394+00:00",
      "created_by_id": null,
      "descr": "Computer Vision Project",
      "id": 2,
      "name": "cv-project",
      "namespace": {
        "created_at": "2025-06-20T04:34:24.074155+00:00",
        "created_by_id": null,
        "descr": "CV Team Namespace",
        "id": 2,
        "name": "cv-team",
        "team_id": 1,
        "uuid": "b73d73f9-420a-5314-0994-7859bd067c19"
      },
      "uuid": "ddd29091-4578-55f5-c1d2-77651d997f2"
    },
    "team_id": 1,
    "versions": [
      {
        "created_at": "2025-06-11T14:22:31.234567+00:00",
        "created_by_id": 2,
        "dataset_id": 1853,
        "error_message": "",
        "error_stack": "",
        "finished_at": "2025-06-11T14:23:45.345678+00:00",
        "id": 4150,
        "job_id": null,
        "num_objects": 1000,
        "query_script": "",
        "size": 5000000,
        "status": 4,
        "uuid": "628e849f-b66c-5647-b849-278d35df94d4",
        "version": "2.0.0"
      }
    ],
    "warehouse_id": null
  }
]
```

**Using Studio Client**:
```python
from datachain.remote.studio import StudioClient

client = StudioClient(team="my-team")
response = client.ls_datasets(prefix="training")

if response.ok:
    for dataset in response.data:
        print(f"Dataset: {dataset['name']}")
```

---

### List Dataset Contents

**Description**: List contents of a specific dataset path.

**Endpoint**: `/datachain/ls`

**Method**: `POST`

**Content Type**: `application/msgpack`

**Authentication**: Requires `DATASETS` scope from [`datachain auth login`](../../../commands/auth/login.md)

**Parameters**:

- `source` (body, required): Path to list contents for
- `team_name` (body, required): Team identifier (automatically added by client)

**Example Request**:
```bash
curl -X POST "https://studio.datachain.ai/api/datachain/ls" \
  -H "Authorization: token YOUR_STUDIO_TOKEN" \
  -H "Content-Type: application/msgpack" \
  -d '{
    "source": "s3://my-bucket/dataset/images",
    "team_name": "my-team"
  }'
```

**Example Response**:
```json
{
  "data": [
    {
      "path_str": "s3://my-bucket/data/image1.jpg",
      "name": "image1.jpg",
      "is_latest": true,
      "last_modified": "2024-01-01T12:00:00Z",
      "dir_type": 0,
      "size": 1048576
    },
    {
      "path_str": "s3://my-bucket/data/folder/",
      "name": "folder",
      "is_latest": true,
      "last_modified": "2024-01-01T12:00:00Z",
      "dir_type": 1,
      "size": null
    },
    {
      "path": "s3://my-bucket/data/document.pdf",
      "name": "document.pdf",
      "is_latest": true,
      "last_modified": "2024-01-01T11:30:00Z",
      "dir_type": 0,
      "size": 2097152
    }
  ]
}
```

**Using Studio Client**:
```python
from datachain.remote.studio import StudioClient

client = StudioClient(team="my-team")
for path, response in client.ls(["s3://my-bucket/dataset/images"]):
    if response.ok:
        for item in response.data:
            print(f"Item: {item['path']}, Size: {item['size']}")
```

---

### Get Dataset Information

**Description**: Retrieve detailed information about a specific dataset.

**Endpoint**: `/datachain/datasets/info`

**Method**: `GET`

**Authentication**: Requires `DATASETS` scope from [`datachain auth login`](../../../commands/auth/login.md)

**Parameters**:

- `namespace` (query, required): Dataset namespace
- `project` (query, required): Project name
- `name` (query, required): Dataset name
- `team_name` (query, required): Team identifier (automatically added by client)

**Example Request**:
```bash
curl -X GET "https://studio.datachain.ai/api/datachain/datasets/info?namespace=default&project=computer_vision&name=training_images&team_name=my-team" \
  -H "Authorization: token YOUR_STUDIO_TOKEN" \
  -H "Content-Type: application/json"
```

**Example Response**:
```json
{
  "id": 1974,
  "name": "amrit-datachain-test-2",
  "description": "Test dataset for playground files and nested structures",
  "attrs": [],
  "created_at": "2025-07-07T08:56:37.571075+00:00",
  "finished_at": "2025-07-07T09:56:37.571075+00:00",
  "error_message": "",
  "error_stack": "",
  "script_output": "",
  "sources": "",
  "query_script": "",
  "status": 4,
  "team_id": 1,
  "warehouse_id": null,
  "created_by_id": 1,
  "schema": {
    "sys__id": {
      "type": "UInt64"
    },
    "sys__rand": {
      "type": "UInt64"
    },
    "file__source": {
      "type": "String"
    },
    "file__path": {
      "type": "String"
    },
    "file__size": {
      "type": "Int64"
    },
    "file__version": {
      "type": "String"
    },
    "file__etag": {
      "type": "String"
    },
    "file__is_latest": {
      "type": "Boolean"
    },
    "file__last_modified": {
      "type": "DateTime"
    },
    "file__location": {
      "type": "JSON"
    }
  },
  "feature_schema": {
    "file": "File@v1"
  },
  "versions": [
    {
      "id": 4827,
      "uuid": "e8a57253-7584-492e-becd-81c46a8cd35c",
      "version": "1.0.0",
      "status": 4,
      "created_at": "2025-08-08T12:53:30.369675+00:00",
      "finished_at": "2025-08-08T12:53:30.446309+00:00",
      "num_objects": 12,
      "size": 153,
      "error_message": "",
      "error_stack": "",
      "script_output": "",
      "job_id": "dda65efc-d0f7-4ae9-902f-ba081da52b3d"
    }
  ],
  "project": {
    "id": 10886,
    "uuid": "96444b21-34f8-49d2-89f5-a9db7a6fc94c",
    "name": "default",
    "descr": "Default project for testing",
    "created_at": "2025-07-07T08:56:37.571075+00:00",
    "created_by_id": 1,
    "namespace": {
      "id": 10885,
      "uuid": "6bce9f26-69ab-4907-b1da-3de743ac3836",
      "name": "@amritghimire",
      "descr": "Personal namespace for testing",
      "created_at": "2025-07-07T08:56:37.568605+00:00",
      "team_id": 1,
      "created_by_id": 1
    }
  }
}
```

**Using Studio Client**:
```python
from datachain.remote.studio import StudioClient

client = StudioClient(team="my-team")
response = client.dataset_info(
    namespace="default",
    project="computer_vision",
    name="training_images"
)

if response.ok:
    dataset = response.data
    print(f"Dataset: {dataset['name']}")
    print(f"Versions: {len(dataset['versions'])}")
```

---
