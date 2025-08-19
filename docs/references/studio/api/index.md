# Studio REST API

DataChain Studio provides a comprehensive REST API for programmatically managing datasets, jobs, and storage operations. All API endpoints require authentication and are scoped to specific teams.

## Authentication

All API endpoints require authentication via a Studio token. The token must be included in the `Authorization` header.

- **Token**: Set via [`datachain auth login`](../../../commands/auth/login.md) or environment variable `DATACHAIN_STUDIO_TOKEN`
- **Team**: Set via [`datachain auth team`](../../../commands/auth/team.md) or environment variable `DATACHAIN_STUDIO_TEAM`
- **Base URL**: `https://studio.datachain.ai/api`

### Required Headers
```http
Authorization: token YOUR_STUDIO_TOKEN
Content-Type: application/json
```

## API Endpoints
The following APIs are available:

- [List datasets](../../../references/studio/api/datasets.md#list-datasets)
- [List Dataset Contents](../../../references/studio/api/datasets.md#list-dataset-contents)
- [Get Dataset Information](../../../references/studio/api/datasets.md#get-dataset-information)
- [Upload file](../../../references/studio/api/jobs.md#upload-file)
- [Create Job](../../../references/studio/api/jobs.md#create-job)
- [List Job](../../../references/studio/api/jobs.md#list-jobs)
- [Cancel Job](../../../references/studio/api/jobs.md#cancel-job)
- [Get compute clusters](../../../references/studio/api/jobs.md#get-compute-clusters)
- [Follow Job Logs](../../../references/studio/api/jobs.md#follow-job-logs-websocket)
