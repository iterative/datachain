# Git Connections

DataChain Studio integrates seamlessly with Git repositories to manage your data processing code, track changes, and enable collaboration.

## Overview

Git connections in DataChain Studio enable:
- **Source Code Management**: Store DataChain scripts and configurations in Git
- **Version Control**: Track changes to your data processing pipelines
- **Collaboration**: Share code and collaborate with team members
- **Automated Workflows**: Trigger jobs based on Git events

## Supported Git Providers

DataChain Studio supports integration with major Git hosting providers:

### GitHub
- **GitHub.com**: Public and private repositories
- **GitHub Enterprise**: Self-hosted GitHub instances
- **GitHub App**: Dedicated app integration for enhanced security

[Learn more about GitHub integration →](github-app.md)

### GitLab
- **GitLab.com**: SaaS GitLab service
- **Self-hosted GitLab**: Custom GitLab installations
- **OAuth Integration**: Secure authentication flow

[Learn more about GitLab integration →](custom-gitlab-server.md)

### Bitbucket
- **Bitbucket Cloud**: Atlassian's cloud service
- **OAuth Integration**: Secure authentication and access
- **Repository Access**: Public and private repository support

## Setting Up Git Connections

### Initial Setup

1. **Navigate to Settings**: Go to Account Settings → Git Connections
2. **Choose Provider**: Select your Git hosting provider
3. **Authorize Access**: Complete OAuth authorization flow
4. **Configure Permissions**: Set repository access permissions

### GitHub Setup

For GitHub repositories:
1. Install the DataChain Studio GitHub App
2. Configure organization and repository access
3. Grant necessary permissions for repository access
4. Test connection with a sample repository

### GitLab Setup

For GitLab repositories:
1. Create OAuth application in GitLab
2. Configure redirect URLs and scopes
3. Add GitLab connection in Studio settings
4. Authorize application access

### Bitbucket Setup

For Bitbucket repositories:
1. Create OAuth consumer in Bitbucket
2. Configure callback URLs and permissions
3. Add Bitbucket connection in Studio
4. Complete authorization flow

## Repository Configuration

### DataChain Repository Structure

Organize your repository for optimal Studio integration:

```
my-datachain-project/
├── .datachain/
│   └── config.yaml          # DataChain configuration
├── scripts/
│   ├── process_data.py      # Main processing script
│   ├── feature_extraction.py
│   └── data_validation.py
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
└── .github/
    └── workflows/
        └── datachain.yml    # GitHub Actions workflow
```

### Configuration Files

#### DataChain Configuration (`.datachain/config.yaml`)
```yaml
version: 1
datasets:
  customer_data:
    source: "s3://company-data/customers/"
    description: "Customer transaction data"
    
jobs:
  feature_extraction:
    script: "scripts/feature_extraction.py"
    resources:
      cpu: 2
      memory: "8GB"
    schedule: "0 2 * * *"  # Daily at 2 AM
```

#### Requirements File (`requirements.txt`)
```
datachain>=0.1.0
pandas>=1.3.0
scikit-learn>=1.0.0
boto3>=1.24.0
```

## Git Workflows

### Development Workflow

Standard Git workflow for DataChain development:

1. **Clone Repository**
   ```bash
   git clone https://github.com/company/data-processing
   cd data-processing
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/new-algorithm
   ```

3. **Develop and Test**
   ```bash
   # Develop your DataChain script
   # Test locally
   datachain run scripts/new_algorithm.py --local
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "Add new customer segmentation algorithm"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/new-algorithm
   # Create pull request via Git provider UI
   ```

6. **Studio Integration**
   - Studio automatically detects new commits
   - Jobs can be triggered on PR creation or merge
   - Results are linked back to commits

### Continuous Integration

Integrate Studio with CI/CD pipelines:

#### GitHub Actions Example
```yaml
# .github/workflows/datachain.yml
name: DataChain CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        python -m pytest tests/
    
    - name: Submit DataChain job
      if: github.ref == 'refs/heads/main'
      run: |
        datachain job run scripts/process_data.py \
          --token ${{ secrets.DATACHAIN_TOKEN }}
```

### Branch-based Jobs

Configure different behavior for different branches:

```yaml
# Branch-specific configurations
branches:
  main:
    auto_deploy: true
    resources:
      cpu: 4
      memory: "16GB"
    
  develop:
    auto_test: true
    resources:
      cpu: 2
      memory: "8GB"
    
  feature/*:
    validation_only: true
    resources:
      cpu: 1
      memory: "4GB"
```

## Security and Permissions

### Repository Access

Control access to repositories:
- **Read Access**: Required for cloning and reading code
- **Write Access**: Needed for status updates and comments
- **Admin Access**: Required for webhook configuration

### Security Best Practices

1. **Least Privilege**: Grant minimum necessary permissions
2. **Token Management**: Use secure token storage and rotation
3. **Access Reviews**: Regularly review repository access
4. **Audit Logging**: Monitor repository access and changes

### Enterprise Security

For enterprise deployments:
- **SSO Integration**: Single sign-on with corporate identity
- **IP Restrictions**: Limit access by IP address
- **Audit Trails**: Comprehensive audit logging
- **Compliance**: Meet regulatory compliance requirements

## Webhooks and Automation

### Git Webhooks

Automatically trigger Studio actions on Git events:

#### Supported Events
- **Push**: New commits pushed to repository
- **Pull Request**: PR created, updated, or merged
- **Release**: New release created
- **Tag**: New tag created

#### Webhook Configuration
```yaml
# Webhook configuration
webhooks:
  - event: "push"
    branch: "main"
    action: "run_job"
    job: "data_processing"
    
  - event: "pull_request"
    action: "validate"
    job: "data_validation"
    
  - event: "release"
    action: "deploy"
    environment: "production"
```

### Automated Workflows

Create automated workflows based on Git events:

1. **Code Change Detection**: Automatically detect relevant changes
2. **Job Triggering**: Submit jobs based on changes
3. **Result Reporting**: Report results back to Git provider
4. **Status Updates**: Update commit status based on job results

## Troubleshooting

### Common Issues

#### Connection Problems
- **OAuth Failures**: Check redirect URLs and permissions
- **Token Expiration**: Refresh expired tokens
- **Network Issues**: Verify firewall and network settings

#### Repository Access Issues
- **Permission Denied**: Verify repository permissions
- **App Installation**: Ensure GitHub App is properly installed
- **Branch Access**: Check branch protection rules

#### Job Trigger Issues
- **Webhook Failures**: Verify webhook configuration
- **Branch Mismatch**: Check branch names and patterns
- **Resource Limits**: Verify resource availability

### Getting Help

For Git connection issues:
- Check the [troubleshooting guide](../troubleshooting.md)
- Review provider-specific documentation
- Contact support with repository and error details

## Next Steps

- Set up [GitHub App integration](github-app.md)
- Configure [custom GitLab server](custom-gitlab-server.md)
- Learn about [team collaboration](../team-collaboration.md)
- Explore [API integration](../../api/index.md) for custom workflows