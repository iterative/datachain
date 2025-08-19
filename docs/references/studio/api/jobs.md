
### Upload file

**Description**: Upload a file to Studio to use in Studio Job.

**Endpoint**: `/datachain/upload-file`

**Method**: `POST`

**Authentication**: Requires `JOB` scope from [`datachain auth login`](../../../commands/auth/login.md)

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

**Using Studio Client**:
```python
from datachain.remote.studio import StudioClient

client = StudioClient(team="my-team")

with open("file_name.txt", "rb") as f:
  content = f.read()
  response = client.upload_file(content, "file_name.txt")
  file_id = response.data.get("blob", {}).get("id")
```

### Create Job

**Description**: Submit a new job for execution in Studio.

**Endpoint**: `/datachain/job`

**Method**: `POST`

**Authentication**: Requires `EXPERIMENTS` scope from [`datachain auth login`](../../../commands/auth/login.md)

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

Check documentation from [`datachain job run`] for more information about the parameters above.
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

**Authentication**: Requires `EXPERIMENTS` scope from [`datachain auth login`](../../../commands/auth/login.md)

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
        print(f"Job {job['id']}: {job['status']} - {job['name']}")
```

---

### Cancel Job

**Description**: Cancel a running or queued job.

**Endpoint**: `/datachain/job/{job_id}/cancel`

**Method**: `POST`

**Authentication**: Requires `EXPERIMENTS` scope from [`datachain auth login`](../../../commands/auth/login.md)

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

**Authentication**: Requires `EXPERIMENTS` scope from [`datachain auth login`](../../../commands/auth/login.md)

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

**Authentication**: Requires `EXPERIMENTS` scope from [`datachain auth login`](../../../commands/auth/login.md)

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
