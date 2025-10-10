# Share Dataset

Learn how to share datasets with team members and control access permissions in DataChain Studio.

## Overview

Sharing datasets enables collaboration by allowing team members to access, process, and contribute to shared data resources.

## Share with Team Members

### Individual Sharing
1. Navigate to your dataset
2. Click "Share" button
3. Enter email addresses of team members
4. Select appropriate permissions
5. Send invitations

### Team Sharing
1. Create dataset within team workspace
2. All team members get access based on their role
3. Configure team-level permissions
4. Manage access through team settings

## Access Permissions

### Permission Levels

#### Viewer
- **Read Access**: View dataset contents and metadata
- **Browse**: Navigate dataset structure
- **Download**: Download data files (if enabled)
- **No Modifications**: Cannot modify or delete data

#### Contributor
- **Read/Write**: Full access to dataset contents
- **Job Submission**: Submit processing jobs
- **Metadata Updates**: Update dataset descriptions and tags
- **Limited Admin**: Some administrative functions

#### Admin
- **Full Control**: Complete dataset management
- **User Management**: Add/remove users and manage permissions
- **Settings**: Configure dataset settings and integrations
- **Deletion**: Delete dataset (with appropriate safeguards)

## Sharing Methods

### Direct Invitations
```bash
# Via Studio UI
1. Dataset → Share → Add Members
2. Enter email addresses
3. Select permission level
4. Send invitation

# Via CLI
datachain dataset share --name my-dataset \
  --user alice@company.com --permission contributor
```

### Team Integration
```bash
# Create team dataset
datachain dataset create --team data-team \
  --name shared-customer-data \
  --repo https://github.com/company/customer-data
```

### Public Sharing
```bash
# Make dataset publicly accessible
datachain dataset update --name my-dataset --public true
```

## Access Control

### Fine-grained Control
- **Data Access**: Control access to raw data files
- **Job Permissions**: Control who can submit processing jobs
- **Export Rights**: Control data export and download permissions
- **Metadata Access**: Control access to dataset metadata

### Security Features
- **Audit Logging**: Track all access and modifications
- **Access Reviews**: Regular access permission reviews
- **Expiration**: Set expiration dates for temporary access
- **IP Restrictions**: Restrict access by IP address (Enterprise)

## Collaboration Workflows

### Research Collaboration
1. **Data Owner** creates dataset and invites researchers
2. **Researchers** explore data and submit analysis jobs
3. **Results** are shared back with the team
4. **Insights** are documented and shared

### Data Pipeline Collaboration
1. **Data Engineers** create and maintain datasets
2. **Data Scientists** access data for model development
3. **ML Engineers** deploy models using shared datasets
4. **Analysts** create reports and dashboards

## Managing Shared Access

### User Management
- **Add Users**: Invite new users with appropriate permissions
- **Update Permissions**: Change user permission levels
- **Remove Access**: Revoke access when no longer needed
- **Bulk Operations**: Manage multiple users simultaneously

### Access Monitoring
- **Activity Logs**: Monitor user activity and access patterns
- **Usage Reports**: Generate reports on dataset usage
- **Security Alerts**: Get alerted to suspicious access patterns
- **Compliance**: Maintain compliance with access policies

## Best Practices

### Security
1. **Principle of Least Privilege**: Grant minimum necessary permissions
2. **Regular Reviews**: Regularly review and update access permissions
3. **Strong Authentication**: Use multi-factor authentication
4. **Audit Trails**: Maintain comprehensive audit logs

### Collaboration
1. **Clear Documentation**: Document dataset purpose and usage guidelines
2. **Communication**: Establish clear communication channels
3. **Version Control**: Use proper versioning for dataset changes
4. **Backup**: Maintain backups of critical shared datasets

### Governance
1. **Data Classification**: Classify datasets by sensitivity level
2. **Retention Policies**: Implement appropriate data retention policies
3. **Compliance**: Ensure compliance with relevant regulations
4. **Quality Standards**: Maintain consistent data quality standards

## Enterprise Features

### Advanced Access Control
- **Role-based Access Control (RBAC)**: Define custom roles
- **Attribute-based Access Control (ABAC)**: Dynamic access policies
- **Integration**: SSO and identity provider integration
- **Compliance**: GDPR, HIPAA, SOC 2 compliance features

### Enterprise Management
- **Centralized Management**: Enterprise-wide dataset governance
- **Policy Enforcement**: Automated policy enforcement
- **Reporting**: Comprehensive access and usage reporting
- **Support**: Dedicated enterprise support

## Next Steps

- Learn about [team collaboration](../team-collaboration.md) features
- Set up [authentication](../authentication/single-sign-on.md) for secure access
- Explore [API integration](../../api/index.md) for programmatic sharing
- Configure [webhooks](../../webhooks.md) for access notifications
