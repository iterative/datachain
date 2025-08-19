## Studio Client

You can use Studio Client to interact with Iterative Studio. Similar to [Rest API](./../../references/studio/api/index.md), this also require authentication scoped to specific team.


## Authentication

All method calls require authentication via a Studio token. The token must be included in the `Authorization` header.

- **Token**: Set via [`datachain auth login`](../../commands/auth/login.md) or environment variable `DATACHAIN_STUDIO_TOKEN`
- **Team**: Set via [`datachain auth team`](../../commands/auth/team.md) or environment variable `DATACHAIN_STUDIO_TEAM`

::: datachain.remote.studio.StudioClient
