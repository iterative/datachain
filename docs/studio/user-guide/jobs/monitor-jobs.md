# Monitor Jobs

DataChain Studio provides comprehensive job monitoring capabilities to help you track execution, debug issues, and optimize performance.

## Job Dashboard

The job dashboard provides an overview of all your jobs:

### Job List View
- **Status Overview**: See all jobs with their current status
- **Filtering**: Filter by status, date, user, or dataset
- **Sorting**: Sort by creation time, duration, or status
- **Search**: Find specific jobs by name or ID

### Job Status Indicators
- 🟢 **COMPLETE**: Job finished successfully
- 🔵 **RUNNING**: Job is currently executing
- 🟡 **QUEUED**: Job is waiting to start
- 🟠 **INIT**: Job is initializing
- 🔴 **FAILED**: Job encountered an error
- ⚫ **CANCELED**: Job was stopped by user

## Real-time Monitoring

### Live Logs
View job logs in real-time as your job executes:

```bash
[2024-01-15 10:30:15] INFO: Starting DataChain job
[2024-01-15 10:30:16] INFO: Loading data from s3://my-bucket/data/
[2024-01-15 10:30:45] INFO: Processing 10,000 files
[2024-01-15 10:31:20] INFO: Completed batch 1/10
[2024-01-15 10:32:10] INFO: Completed batch 2/10
```

### Progress Tracking
Monitor job progress with:
- **Progress Bar**: Visual progress indicator
- **ETA**: Estimated time to completion
- **Throughput**: Processing rate (files/second, records/minute)
- **Current Stage**: Which part of the pipeline is running

### Resource Usage
Track resource consumption:
- **CPU Usage**: Current CPU utilization percentage
- **Memory Usage**: RAM consumption vs. allocated
- **GPU Usage**: GPU utilization (if applicable)
- **Storage**: Disk usage for temporary files

## Job Details

Click on any job to view detailed information:

### Overview Tab
- **Job ID**: Unique identifier
- **Status**: Current execution status
- **Duration**: Total runtime or elapsed time
- **Created By**: User who submitted the job
- **Created At**: Job submission timestamp
- **Started At**: Job execution start time
- **Finished At**: Job completion time (if applicable)

### Configuration Tab
- **Script Path**: Path to the executed script
- **Git Branch**: Branch used for execution
- **Parameters**: Command line arguments passed to the script
- **Environment Variables**: Environment configuration
- **Resource Allocation**: CPU, memory, GPU settings

### Logs Tab
- **Full Logs**: Complete job output and error logs
- **Log Levels**: Filter by INFO, WARNING, ERROR, DEBUG
- **Download**: Download logs for offline analysis
- **Search**: Search within logs for specific messages

### Metrics Tab
- **Performance Metrics**: Job execution statistics
- **Resource Usage Over Time**: CPU, memory, GPU usage graphs
- **Data Processing Metrics**: Files processed, records handled
- **Error Rates**: Track processing errors and success rates

## Advanced Monitoring

### Job Comparison
Compare multiple jobs to analyze performance:
- **Side-by-side View**: Compare job configurations and results
- **Performance Comparison**: Compare execution times and resource usage
- **Success Rates**: Compare error rates across different runs

### Historical Analysis
Analyze job trends over time:
- **Job Success Rate**: Track success/failure trends
- **Average Duration**: Monitor performance improvements
- **Resource Utilization**: Optimize resource allocation
- **Cost Analysis**: Track computational costs

### Custom Dashboards
Create custom monitoring dashboards:
- **Team Dashboard**: Monitor all team jobs
- **Project Dashboard**: Track jobs for specific datasets
- **Performance Dashboard**: Focus on performance metrics
- **Cost Dashboard**: Monitor resource costs and usage

## Alerting and Notifications

### Email Notifications
Set up email alerts for:
- **Job Completion**: Notify when jobs finish
- **Job Failures**: Alert on job errors
- **Long Running Jobs**: Warn about jobs exceeding expected duration
- **Resource Limits**: Alert when approaching resource quotas

### Webhook Notifications
Configure webhooks for real-time notifications:
```json
{
    "action": "job_status",
    "job": {
        "id": "da59df47-d121-4eb6-aa76-dc452755544e",
        "status": "COMPLETE",
        "name": "data_processing.py",
        "created_at": "2024-01-15T10:30:15.000Z",
        "finished_at": "2024-01-15T11:45:30.000Z",
        "url": "https://studio.datachain.ai/team/my-team/jobs/da59df47-d121-4eb6-aa76-dc452755544e"
    }
}
```

### Slack Integration
Send job status updates to Slack channels:
- **Success Messages**: Celebrate job completions
- **Failure Alerts**: Immediate notification of job failures
- **Performance Reports**: Regular performance summaries

## Troubleshooting Jobs

### Common Issues

#### Job Stuck in QUEUED State
- **Check Resource Availability**: Verify if resources are available
- **Review Team Quotas**: Ensure you haven't exceeded limits
- **Check Dependencies**: Verify all dependencies are available

#### Job Fails During INIT
- **Environment Issues**: Check Python environment and dependencies
- **Git Access**: Verify repository access and branch availability
- **Configuration Errors**: Review job parameters and settings

#### Job Fails During Execution
- **Script Errors**: Check script syntax and logic errors
- **Data Access**: Verify input data availability and permissions
- **Resource Limits**: Check if job exceeded memory or storage limits

### Debugging Workflow

1. **Check Job Status**: Review the current status and error messages
2. **Examine Logs**: Look for error messages and stack traces
3. **Verify Configuration**: Ensure all parameters are correct
4. **Test Locally**: Run the script locally with sample data
5. **Check Resources**: Verify resource availability and limits
6. **Contact Support**: If issues persist, contact support with job ID

## Performance Optimization

### Resource Right-sizing
- **Monitor Usage**: Track actual vs. allocated resources
- **Adjust Allocation**: Optimize CPU, memory, and GPU settings
- **Cost Optimization**: Balance performance and cost

### Code Optimization
- **Profile Performance**: Identify bottlenecks in your code
- **Optimize Algorithms**: Improve data processing efficiency
- **Parallel Processing**: Use multiple workers for large datasets

### Data Optimization
- **Data Partitioning**: Split large datasets for parallel processing
- **Caching**: Cache frequently accessed data
- **Compression**: Use compression to reduce I/O overhead

## Next Steps

- Set up [webhook notifications](../../webhooks.md) for automated alerts
- Learn about [team collaboration](../team-collaboration.md) for shared monitoring
- Explore [advanced job configurations](../../../guide/processing.md)
- Check out [API integration](../../api/index.md) for programmatic monitoring