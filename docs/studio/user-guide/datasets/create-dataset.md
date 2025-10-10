# Create a Dataset

In this section, you will learn how to:

- [Connect to a Git repository and add a dataset](#connect-to-a-git-repository-and-add-a-dataset)
- [Create multiple datasets from a single Git repository](#create-multiple-datasets-from-a-single-git-repository)
- [Create datasets shared across a team](#create-datasets-shared-across-a-team)

## Connect to a Git repository and add a dataset

To add a new dataset, follow these steps:

1. Sign in to DataChain Studio using your GitHub.com, GitLab.com, or Bitbucket.org account, or with your email address.

2. Click on `Add a Dataset`. All the organizations that you have access to will be listed.

   !!! info
       If you do not see your desired organizations or Git repositories, make sure that [the connection to your Git server has been set up](../account-management.md#git-connections).

       To connect to your GitHub repositories, you must install the DataChain Studio GitHub app. Refer to the section on [GitHub app installation](../git-connections/github-app.md) for more details.

       To connect to repositories on your self-hosted GitLab server, you must first add a connection to this server and create a team. Refer to the section on [self-hosted GitLab server support](../git-connections/custom-gitlab-server.md) for more details.

3. Open the organization whose repository you want to connect to. You can also use the search bar to directly look for a repository.

4. Click on the Git repository that you want to connect to.

5. In the `Dataset settings` page that opens up, you can edit the dataset name, directory and visibility (public accessibility). These settings can also be edited after the dataset has been created.

   !!! info
       If your DataChain code is in a sub-directory of a repository, make sure to specify the correct directory path. DataChain Studio will look for `datachain` configuration and code in this directory.

6. Click on `Create Dataset`.

That's it! DataChain Studio will now create the dataset. If there are any DataChain jobs or configurations in your repository, they will be automatically detected and displayed.

## Create multiple datasets from a single Git repository

You can create multiple datasets from a single Git repository. This is useful when:

- You have different data processing workflows in different directories
- You want to separate development and production datasets
- You have different teams working on different parts of the same repository

To create multiple datasets:

1. Follow the same steps as above for creating a dataset
2. In the dataset settings, specify a different name and/or directory path
3. Configure the dataset parameters as needed for this specific workflow

Each dataset will track its own jobs, data, and configurations independently.

## Create datasets shared across a team

To create datasets that can be accessed by your entire team:

1. Create a team in DataChain Studio (if you haven't already)
2. When creating the dataset, make sure you're working within the team context
3. Set the appropriate visibility and access permissions for the dataset
4. Team members will be able to view and contribute to the dataset based on their role permissions

Team datasets enable:

- Shared access to data processing workflows
- Collaborative development and review
- Centralized job monitoring and management
- Consistent data quality standards across the team

## Dataset Configuration

After creating a dataset, you can configure:

- **Data sources**: Connect to cloud storage, databases, or file systems
- **Processing parameters**: Set default parameters for DataChain jobs
- **Access control**: Manage who can view and modify the dataset
- **Notifications**: Set up alerts for job completion or failures
- **Metadata**: Add descriptions, tags, and documentation

## Next Steps

Once your dataset is created:

1. [Explore your dataset](explore-datasets.md) to understand its structure and content
2. [Run processing jobs](../jobs/create-and-run.md) to transform and analyze your data
3. [Share your dataset](share-dataset.md) with team members
4. [Visualize results](visualize-and-compare.md) to gain insights from your data
