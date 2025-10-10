# DataChain Studio

DataChain Studio is a web application that enables Machine Learning and Data teams to seamlessly

- [Run and track jobs](user-guide/jobs/index.md)
- [Manage datasets](user-guide/datasets/index.md)
- [Collaborate on data projects](user-guide/team-collaboration.md)

It works on top of [DataChain](https://datachain.ai/) and [Git](https://git-scm.com/), maintaining Git as the single-source-of-truth for your data, jobs and datasets.

Sign in to DataChain Studio using your GitHub.com, GitLab.com, or Bitbucket.org account, or with your email address. Explore the demo projects and datasets, and [let us know](user-guide/troubleshooting.md#support) if you need any help getting started.

## Why DataChain Studio?

- Simplify data processing job tracking, visualization, and collaboration on top of Git.
- Keep your code, data and processing connected at all times.
- Apply your existing software engineering stack for data teams.
- Build a comprehensive data processing platform for transparency and discovery across all your projects.
- Because your platform is built on top of Git, you can use [GitOps](https://www.gitops.tech/) for deployment and automation.

## Getting Started

New to DataChain Studio? Start with these guides:

- **[User Guide](user-guide/index.md)** - Learn how to use DataChain Studio features
- **[API Reference](api/index.md)** - Integrate with Studio programmatically
- **[Webhooks](webhooks.md)** - Set up event notifications
- **[Self-hosting](self-hosting/index.md)** - Deploy your own Studio instance

## Key Features

### Dataset Management
- Track and version your datasets
- Visualize data processing pipelines
- Share datasets across teams

### Job Processing
- Run data processing jobs in the cloud
- Monitor job progress and logs
- Schedule recurring data processing tasks

### Team Collaboration
- Share projects with team members
- Control access with role-based permissions
- Integrate with Git workflows

### API Integration
- RESTful API for programmatic access
- Webhook notifications for automation
- Command-line tools for developers

## Architecture

DataChain Studio is designed to work with your existing data infrastructure:

- **Git Integration**: Native support for GitHub, GitLab, and Bitbucket
- **Cloud Storage**: Works with S3, GCS, Azure Blob, and other cloud storage
- **Flexible Deployment**: Available as SaaS or self-hosted
- **API-First**: Everything accessible via REST API and CLI

Visit [studio.datachain.ai](https://studio.datachain.ai) to get started, or learn about [self-hosting](self-hosting/index.md) for enterprise deployments.
