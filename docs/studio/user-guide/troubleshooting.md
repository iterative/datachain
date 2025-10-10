# Troubleshooting

This guide helps you resolve common issues when using DataChain Studio.

## Getting Support {#support}

If you can't find a solution to your problem in this guide:

### Community Support
- **GitHub Issues**: Report bugs and feature requests on our [GitHub repository](https://github.com/iterative/datachain)
- **Discord**: Join our [Discord community](https://discord.gg/datachain) for real-time help
- **Stack Overflow**: Ask questions with the `datachain` tag

### Enterprise Support
- **Support Portal**: Access dedicated support through the enterprise portal
- **Email Support**: Contact enterprise support at support@datachain.ai
- **Phone Support**: Available for enterprise customers during business hours

## Common Issues

### Authentication Problems

#### Cannot log in to Studio
**Symptoms**: Login page shows errors or redirects fail

**Solutions**:
1. Clear browser cache and cookies
2. Try incognito/private browsing mode
3. Check if your organization's firewall blocks studio.datachain.ai
4. Verify your Git provider (GitHub/GitLab/Bitbucket) account is accessible

#### Token authentication fails
**Symptoms**: API calls return 401 Unauthorized

**Solutions**:
1. Verify token is correctly copied (no extra spaces)
2. Check token hasn't expired
3. Ensure token has correct scopes for the operation
4. Regenerate token if needed

```bash
# Test token validity
curl -H "Authorization: token $DATACHAIN_STUDIO_TOKEN" \
  https://studio.datachain.ai/api/user
```

### Git Integration Issues

#### Repository not visible in Studio
**Symptoms**: Cannot find your repository when creating datasets

**Solutions**:
1. **GitHub**: Install DataChain Studio GitHub App
   - Go to Account Settings → Git Connections
   - Install app on organization/repository
   - Grant necessary permissions

2. **GitLab**: Check OAuth application access
   - Verify GitLab OAuth app is authorized
   - Check organization membership and permissions

3. **Bitbucket**: Verify Bitbucket Cloud access
   - Check Bitbucket OAuth application access
   - Verify repository permissions

#### Repository access denied
**Symptoms**: Error messages about repository access when creating datasets

**Solutions**:
1. Verify you have read access to the repository
2. For private repositories, ensure proper permissions
3. Check if repository was moved or deleted
4. Re-authorize Git connection if needed

### Job Execution Issues

#### Jobs stuck in QUEUED state
**Symptoms**: Jobs remain queued and never start running

**Solutions**:
1. **Check resource availability**:
   ```bash
   # Check team resource usage
   datachain team status
   ```

2. **Verify quotas**: Ensure you haven't exceeded team limits
3. **Check cluster status**: Verify compute clusters are healthy
4. **Resource requirements**: Reduce CPU/memory requirements if too high

#### Jobs fail during initialization
**Symptoms**: Jobs fail in INIT state with error messages

**Solutions**:
1. **Check script path**: Verify the script exists in the repository
2. **Python environment**: Ensure all dependencies are in requirements.txt
3. **Git branch**: Verify the specified branch exists
4. **File permissions**: Check script has execute permissions

```python
# Example requirements.txt
datachain>=0.1.0
pandas>=1.3.0
numpy>=1.21.0
```

#### Jobs fail during execution
**Symptoms**: Jobs start running but fail with errors

**Solutions**:
1. **Check logs**: Review job logs for error messages
2. **Test locally**: Run script locally with sample data
3. **Resource limits**: Increase memory/CPU if out-of-memory errors
4. **Data access**: Verify input data paths and credentials

```bash
# Download job logs for analysis
datachain job logs <job-id> > job_logs.txt
```

### Data Access Issues

#### Cannot access cloud storage
**Symptoms**: Jobs fail with S3/GCS/Azure access errors

**Solutions**:
1. **Verify credentials**: Check cloud credentials in account settings
2. **Test permissions**: Ensure credentials have required permissions
3. **Network access**: Verify firewall allows cloud storage access
4. **Bucket names**: Check bucket names and paths are correct

```bash
# Test S3 access
aws s3 ls s3://your-bucket-name/

# Test GCS access
gsutil ls gs://your-bucket-name/
```

#### Data not found errors
**Symptoms**: Scripts fail with "file not found" or "path does not exist"

**Solutions**:
1. **Check paths**: Verify data paths are correct and accessible
2. **Case sensitivity**: Ensure path case matches exactly
3. **Permissions**: Verify read permissions on data files
4. **Data availability**: Confirm data exists at expected location

### Performance Issues

#### Slow job execution
**Symptoms**: Jobs take much longer than expected

**Solutions**:
1. **Optimize code**: Profile code to identify bottlenecks
2. **Increase resources**: Allocate more CPU/memory
3. **Parallel processing**: Use multiple workers for large datasets
4. **Data locality**: Ensure data and compute are in same region

```python
# Example parallel processing
from datachain import DataChain
from concurrent.futures import ThreadPoolExecutor

def process_batch(batch):
    return batch.map(expensive_operation)

# Process in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_batch, data_batches))
```

#### High memory usage
**Symptoms**: Jobs fail with out-of-memory errors

**Solutions**:
1. **Batch processing**: Process data in smaller chunks
2. **Memory profiling**: Use memory profilers to identify leaks
3. **Increase allocation**: Request more memory for jobs
4. **Optimize algorithms**: Use memory-efficient algorithms

```python
# Example batch processing
for batch in DataChain.from_storage("s3://data/").batch(1000):
    processed = batch.map(transform_data)
    processed.save_to("s3://results/", append=True)
```

### Network and Connectivity

#### Timeouts and connection errors
**Symptoms**: Network timeouts, connection refused errors

**Solutions**:
1. **Check network**: Verify internet connectivity
2. **Firewall rules**: Ensure firewalls allow Studio traffic
3. **DNS resolution**: Verify studio.datachain.ai resolves correctly
4. **Retry logic**: Implement retry logic for transient failures

#### SSL/TLS certificate errors
**Symptoms**: Certificate verification failures

**Solutions**:
1. **Update certificates**: Ensure system certificates are up to date
2. **Corporate proxy**: Configure proxy settings if behind corporate firewall
3. **Custom CA**: Add custom CA certificates if needed
4. **Disable verification**: Only for testing - not recommended for production

### Team and Permissions

#### Cannot access team resources
**Symptoms**: Permission denied errors for team operations

**Solutions**:
1. **Check role**: Verify you have appropriate role in team
2. **Team membership**: Confirm you're a member of the team
3. **Resource permissions**: Check resource-specific permissions
4. **Contact admin**: Ask team admin to verify your permissions

#### Team invitation issues
**Symptoms**: Cannot join team or invitation errors

**Solutions**:
1. **Check email**: Verify invitation was sent to correct email
2. **Spam folder**: Check spam/junk folder for invitation
3. **Account matching**: Ensure Studio account uses invited email
4. **Resend invitation**: Ask team admin to resend invitation

## Self-hosting Issues

### Installation Problems

#### Docker/Kubernetes deployment fails
**Symptoms**: Deployment fails with various errors

**Solutions**:
1. **Check requirements**: Verify system meets minimum requirements
2. **Resource allocation**: Ensure sufficient CPU/memory allocated
3. **Network configuration**: Verify network settings and connectivity
4. **Persistent storage**: Check storage configuration and availability

#### Database connection issues
**Symptoms**: Cannot connect to PostgreSQL database

**Solutions**:
1. **Connection string**: Verify database URL format and credentials
2. **Network access**: Ensure database is accessible from Studio pods
3. **Database exists**: Confirm database exists and is properly initialized
4. **Permissions**: Verify database user has necessary permissions

### Configuration Issues

#### SSL/TLS certificate problems
**Symptoms**: HTTPS doesn't work or certificate warnings

**Solutions**:
1. **Certificate validity**: Check certificate is valid and not expired
2. **Certificate chain**: Ensure complete certificate chain is provided
3. **Domain matching**: Verify certificate matches Studio domain
4. **Private key**: Confirm private key matches certificate

#### Authentication integration fails
**Symptoms**: SSO or OIDC authentication doesn't work

**Solutions**:
1. **Configuration**: Verify identity provider configuration
2. **Redirect URLs**: Check callback/redirect URLs are correct
3. **Scopes**: Ensure proper OAuth scopes are configured
4. **Network access**: Verify Studio can reach identity provider

## Debugging Tips

### Enable Debug Logging

For local development:
```bash
export DATACHAIN_LOG_LEVEL=DEBUG
datachain run your_script.py
```

For Studio jobs, add to environment variables:
```yaml
environment:
  DATACHAIN_LOG_LEVEL: DEBUG
```

### Collect Diagnostic Information

When reporting issues, include:

1. **Job/Error details**:
   - Job ID (if applicable)
   - Error messages and stack traces
   - Timestamps when issue occurred

2. **Environment information**:
   - DataChain version
   - Python version
   - Operating system
   - Browser version (for UI issues)

3. **Configuration**:
   - Relevant configuration settings
   - Environment variables (sanitized)
   - Network configuration

### Testing Connectivity

Test network connectivity:
```bash
# Test Studio API connectivity
curl -I https://studio.datachain.ai/api/health

# Test specific endpoints
curl -H "Authorization: token $TOKEN" \
  https://studio.datachain.ai/api/user

# Test cloud storage access
aws s3 ls s3://your-bucket/ --region your-region
```

## Reporting Bugs

When reporting bugs, please include:

1. **Clear description**: What you expected vs. what happened
2. **Steps to reproduce**: Detailed steps to reproduce the issue
3. **Environment details**: Versions, OS, browser, etc.
4. **Error messages**: Complete error messages and stack traces
5. **Screenshots**: Screenshots for UI issues
6. **Logs**: Relevant log excerpts (sanitized)

### Bug Report Template

```markdown
**Description:**
Brief description of the issue

**Expected Behavior:**
What should have happened

**Actual Behavior:**
What actually happened

**Steps to Reproduce:**
1. Step one
2. Step two
3. Step three

**Environment:**
- DataChain Studio version:
- Browser/CLI version:
- Operating System:
- Python version:

**Additional Context:**
Any other relevant information
```

## Next Steps

- Check our [API documentation](../api/index.md) for programmatic solutions
- Explore [self-hosting options](../self-hosting/index.md) for enterprise deployments
- Join our [community](https://discord.gg/datachain) for peer support
