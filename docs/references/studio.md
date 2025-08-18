# Studio REST API

DataChain Studio provides a comprehensive REST API for programmatically managing datasets, jobs, and storage operations. All API endpoints require authentication and are scoped to specific teams.

## Authentication

All API endpoints require authentication via a Studio token. The token must be included in the `Authorization` header.

- **Token**: Set via [`datachain auth login`](../commands/auth/login.md) or environment variable `DATACHAIN_STUDIO_TOKEN`
- **Team**: Set via [`datachain auth team`](../commands/auth/team.md) or environment variable `DATACHAIN_STUDIO_TEAM`
- **Base URL**: `https://studio.datachain.ai/api`

### Required Headers
```http
Authorization: token YOUR_STUDIO_TOKEN
Content-Type: application/json
```

## API Endpoints

### List Datasets

**Description**: Retrieve a list of datasets with optional prefix filtering.

**Endpoint**: `/datachain/datasets`

**Method**: `GET`

**Authentication**: Requires `DATASETS` scope from [`datachain auth login`](../commands/auth/login.md)

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

**Authentication**: Requires `DATASETS` scope from [`datachain auth login`](../commands/auth/login.md)

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

**Authentication**: Requires `DATASETS` scope from [`datachain auth login`](../commands/auth/login.md)

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

### Upload file

**Description**: Upload a file to Studio to use in Studio Job.

**Endpoint**: `/datachain/upload-file`

**Method**: `POST`

**Authentication**: Requires `JOB` scope from [`datachain auth login`](../commands/auth/login.md)

**Parameters**:
- `file_content`: (body, required): base64 encoded value of file content encoded with utf-8
- `file_name`: (body, required): Name of the file
- `team_name` (body, required): Team identifier (automatically added by client)


**Example Request**:
```bash
curl -X POST "https://studio.datachain.ai/api/datachain/upload-file" \
  -H "Authorization: token YOUR_STUDIO_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
        "file_content": "ZmlsZSBjb250ZW50",
        "file_name": "file.txt",
        "team_name": "team_name",
    }'
```

**Example Response**:
```json
{"blob": {"id": 1}}
```

### Create Job

**Description**: Submit a new job for execution in Studio.

**Endpoint**: `/datachain/job`

**Method**: `POST`

**Authentication**: Requires `EXPERIMENTS` scope from [`datachain auth login`](../commands/auth/login.md)

**Parameters**:
- `query` (body, required): Query string or script content
- `query_type` (body, required): Type of query (e.g., "PYTHON", "SHELL")
- `environment` (body, optional): Environment configuration eg. ENVIRONMENT_NAME=1
- `workers` (body, optional): Number of worker processes
- `query_name` (body, optional): Name for the job
- `files` (body, optional): List of file paths to include.
- `python_version` (body, optional): Python version to use
- `requirements` (body, optional): Python requirements file content
- `repository` (body, optional): Git repository URL
- `priority` (body, optional): Job priority level (0-5, lower is higher priority)
- `compute_cluster_name` (body, optional): Target compute cluster
- `start_after` (body, optional): Start time for delayed execution
- `cron_expression` (body, optional): Cron expression for recurring jobs
- `credentials_name` (body, optional): Credentials identifier
- `team_name` (body, required): Team identifier (automatically added by client)

Note that compute_cluster_name and compute_cluster_id are mutually exclusive.
Check documentations from [`datachain job run`] on more information about parameters above.
You can get the file id from above (Upload file) endpoint.

**Example Request**:
```bash
curl -X POST "https://studio.datachain.ai/api/datachain/job" \
  -H "Authorization: token YOUR_STUDIO_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
        "query": "print(1)",
        "query_type": "PYTHON",
        "environment": "ENV_FROM_FILE=1\nENV_FROM_ARGS=1",
        "workers": 2,
        "query_name": "example_query.py",
        "files": ["1"],
        "python_version": "3.12",
        "requirements": "pyjokes\nstupidity",
        "team_name": "team_name",
        "repository": "https://github.com/iterative/datachain",
        "priority": 5,
        "compute_cluster_name": "default",
        "start_after": null,
        "cron_expression": null,
        "credentials_name": "my-credentials",
    }'
```

**Example Response**:
```json
{
    "job": {
        "id": "a184a035-793a-4d04-8ba5-a2153fe72182",
        "url": "https://studio.datachain.ai/team/Iterative/datasets/jobs/a184a035-793a-4d04-8ba5-a2153fe72182"
    }
}
```

**Using Studio Client**:
```python
from datachain.remote.studio import StudioClient

client = StudioClient(team="my-team")
response = client.create_job(
    query="SELECT * FROM my_dataset LIMIT 100",
    query_type="sql",
    workers=2,
    query_name="Data Analysis Job",
    priority=3
)

if response.ok:
    job_id = response.data['id']
    print(f"Job created with ID: {job_id}")
```

---

### List Jobs

**Description**: Retrieve a list of jobs with optional status filtering.

**Endpoint**: `/datachain/jobs`

**Method**: `GET`

**Authentication**: Requires `EXPERIMENTS` scope from [`datachain auth login`](../commands/auth/login.md)

**Parameters**:
- `status` (query, optional): Filter by job status
- `limit` (query, optional): Maximum number of jobs to return (default: 20)
- `team_name` (query, required): Team identifier (automatically added by client)

**Example Request**:
```bash
curl -X GET "https://studio.datachain.ai/api/datachain/jobs?status=running&limit=10&team_name=my-team" \
  -H "Authorization: token YOUR_STUDIO_TOKEN" \
  -H "Content-Type: application/json"
```

**Example Response**:
```json
{
  "jobs": [
    {
      "created_at": "2025-08-18T09:31:11.896956+00:00",
      "created_by": "amritghimire",
      "environment": {},
      "error_message": "",
      "error_stack": "",
      "exit_code": null,
      "finished_at": null,
      "id": "a184a035-793a-4d04-8ba5-a2153fe72182",
      "name": "test.py",
      "python_version": null,
      "query": "",
      "query_type": 1,
      "repository": "",
      "requirements": "",
      "status": "CREATED",
      "team": "Iterative",
      "url": "https://studio.datachain.ai/team/Iterative/datasets/jobs/a184a035-793a-4d04-8ba5-a2153fe72182",
      "workers": 0
    }
  ]
}
```

**Using Studio Client**:
```python
from datachain.remote.studio import StudioClient

client = StudioClient(team="my-team")
response = client.get_jobs(status="running", limit=10)

if response.ok:
    for job in response.data['jobs']:
        print(f"Job {job['id']}: {job['status']} - {job['name']}%")
```

---

### Cancel Job

**Description**: Cancel a running or queued job.

**Endpoint**: `/datachain/job/{job_id}/cancel`

**Method**: `POST`

**Authentication**: Requires `EXPERIMENTS` scope from [`datachain auth login`](../commands/auth/login.md)

**Parameters**:
- `job_id` (path, required): Job identifier
- `team_name` (body, required): Team identifier (automatically added by client)

**Example Request**:
```bash
curl -X POST "https://studio.datachain.ai/api/datachain/job/job_12345/cancel" \
  -H "Authorization: token YOUR_STUDIO_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "team_name": "my-team"
  }'
```

**Example Response**:
```json
{"message": "Successfully canceled"}
```

**Using Studio Client**:
```python
from datachain.remote.studio import StudioClient

client = StudioClient(team="my-team")
response = client.cancel_job("job_12345")

if response.ok:
    print(f"Job cancelled")
```

---

### Get Compute Clusters

**Description**: Retrieve available compute clusters for job execution.

**Endpoint**: `/datachain/clusters`

**Method**: `GET`

**Authentication**: Requires `EXPERIMENTS` scope from [`datachain auth login`](../commands/auth/login.md)

**Parameters**:
- `team_name` (query, required): Team identifier (automatically added by client)

**Example Request**:
```bash
curl -X GET "https://studio.datachain.ai/api/datachain/clusters?team_name=my-team" \
  -H "Authorization: token YOUR_STUDIO_TOKEN" \
  -H "Content-Type: application/json"
```

**Example Response**:
```json
{
  "clusters": [
    {
      "id": 3,
      "name": "cluster",
      "status": "ACTIVE",
      "cloud_provider": "AWS",
      "cloud_credentials": "aws creds",
      "is_active": true,
      "default": true,
      "max_workers": 4
    }
  ]
}
```

**Using Studio Client**:
```python
from datachain.remote.studio import StudioClient

client = StudioClient(team="my-team")
response = client.get_clusters()

if response.ok:
    for cluster in response.data['clusters']:
        print(f"Cluster {cluster['id']}: {cluster['name']} ({cluster['status']})")
```

---

### Follow Job Logs (WebSocket)

**Description**: Stream real-time job logs via WebSocket connection.

**Endpoint**: `/logs/follow/`

**Method**: `WebSocket`

**Authentication**: Requires `EXPERIMENTS` scope from [`datachain auth login`](../commands/auth/login.md)

**Parameters**:
- `job_id` (query, required): Job identifier
- `team_name` (query, required): Team identifier

**Example WebSocket Connection**:
```bash
# Using wscat or similar WebSocket client
wscat -c "wss://studio.datachain.ai/api/logs/follow/?job_id=job_12345&team_name=my-team" \
  -H "Authorization: token YOUR_STUDIO_TOKEN"
```

**Example WebSocket Messages**:
```json
// Log message
{
  "type": "log",
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "message": "Starting data processing job",
  "worker_id": "worker_1"
}

// Job status update
{
  "type": "status",
  "timestamp": "2024-01-15T10:31:00Z",
  "status": "running",
  "progress": 25,
  "message": "Processing batch 1 of 4"
}
```

**Using Studio Client**:
```python
import asyncio
from datachain.remote.studio import StudioClient

async def monitor_logs():
    client = StudioClient(team="my-team")
    async for log_data in client.tail_job_logs("a184a035-793a-4d04-8ba5-a2153fe72182"):
        if "logs" in message:
            for log in message["logs"]:
                print(log["message"], end="")
        elif "job" in message:
            latest_status = message["job"]["status"]
            print(f"\n>>>> Job is now in {latest_status} status.")

# Run the async function
asyncio.run(monitor_logs())
```

## Dependencies

The Studio client requires the following optional dependencies:
- `msgpack`: For efficient data serialization
- `requests`: For HTTP communication
- `websockets`: For real-time log streaming
