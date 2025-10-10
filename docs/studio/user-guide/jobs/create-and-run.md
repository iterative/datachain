# Create and Run Jobs

This guide covers how to create, configure, and run DataChain jobs in Studio.

## Prerequisites

Before creating jobs, ensure you have:

- A connected Git repository with DataChain code
- Proper cloud credentials configured (if accessing remote data)
- A dataset set up in DataChain Studio

## Creating a Job

### 1. Navigate to Jobs

1. Open your dataset in DataChain Studio
2. Click on the `Jobs` tab
3. Click `Create Job` or `Run Job`

### 2. Configure Job Parameters

#### Basic Configuration
- **Job Name**: Give your job a descriptive name
- **Script Path**: Path to your DataChain script (e.g., `src/process_data.py`)
- **Git Branch**: Select the branch to run from (default: main)
- **Working Directory**: Set the working directory for the job

#### Resource Configuration
- **CPU**: Number of CPU cores (1-16)
- **Memory**: RAM allocation (1GB-64GB)
- **GPU**: GPU type and count (optional)
- **Timeout**: Maximum job runtime (default: 24 hours)

#### Environment Variables
Set environment variables for your job:
```bash
DATACHAIN_CLIENT_TOKEN=<auto-populated>
AWS_REGION=us-east-1
BATCH_SIZE=1000
```

#### Input Parameters
Pass parameters to your DataChain script:
```bash
--input-path s3://my-bucket/data/
--output-path s3://my-bucket/results/
--workers 4
```

### 3. Data Configuration

#### Input Data Sources
- **Cloud Storage**: S3, GCS, Azure Blob Storage
- **Databases**: PostgreSQL, MySQL, MongoDB
- **File Systems**: Local files, network shares
- **APIs**: REST APIs, GraphQL endpoints

#### Output Destinations
- **Data Storage**: Where to save processed data
- **Artifacts**: Location for job outputs and logs
- **Metadata**: Database for storing job metadata

### Script Examples

#### Basic DataChain Processing
```python
#!/usr/bin/env python3

import sys
from datachain import DataChain

def main():
    # Process data with DataChain
    dc = (
        DataChain.from_storage("s3://my-bucket/images/")
        .map(lambda file: {"size": file.size, "name": file.name})
        .save("processed_files")
    )
    
    print(f"Processed {len(dc)} files")

if __name__ == "__main__":
    main()
```

#### Parameterized Job
```python
#!/usr/bin/env python3

import argparse
from datachain import DataChain

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    
    # Process data with parameters
    dc = (
        DataChain.from_storage(args.input_path)
        .batch(args.batch_size)
        .map(process_batch)
        .save_to(args.output_path)
    )

if __name__ == "__main__":
    main()
```

## Running Jobs

### Submit Job
1. Review your job configuration
2. Click `Submit Job` to queue for execution
3. The job will be assigned a unique ID and added to the queue

### Job Scheduling
Jobs are scheduled based on:
- **Resource Availability**: Wait for required resources
- **Queue Priority**: First-in-first-out with priority adjustments
- **Team Limits**: Respect team resource quotas

### Job Execution
Once running, your job will:
1. **Initialize**: Set up environment and dependencies
2. **Execute**: Run your DataChain script
3. **Monitor**: Stream logs and progress updates
4. **Complete**: Save results and clean up resources

## Job Templates

Create reusable job templates for common workflows:

### Data Ingestion Template
```yaml
name: "Data Ingestion"
script: "scripts/ingest_data.py"
resources:
  cpu: 2
  memory: "8GB"
parameters:
  - name: "source_path"
    required: true
  - name: "target_table"
    required: true
```

### Feature Engineering Template
```yaml
name: "Feature Engineering"
script: "scripts/feature_engineering.py"
resources:
  cpu: 4
  memory: "16GB"
  gpu: "nvidia-t4"
environment:
  CUDA_VISIBLE_DEVICES: "0"
```

## Batch Jobs

For processing large datasets, use batch job configurations:

### Parallel Processing
```python
from datachain import DataChain
from concurrent.futures import ThreadPoolExecutor

def process_partition(partition):
    # Process a partition of data
    return partition.map(expensive_operation)

# Split data into partitions for parallel processing
dc = DataChain.from_storage("s3://large-dataset/")
partitions = dc.partition(num_partitions=10)

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_partition, partitions))
```

### Chunked Processing
```python
from datachain import DataChain

# Process data in chunks to manage memory
for chunk in DataChain.from_storage("s3://data/").batch(1000):
    processed = chunk.map(transform_data)
    processed.save_to("s3://results/", append=True)
```

## Troubleshooting

### Common Issues

#### Resource Errors
- **Out of Memory**: Increase memory allocation or reduce batch size
- **CPU Timeout**: Optimize code or increase CPU allocation
- **Storage Full**: Clean up temporary files or increase storage

#### Configuration Errors
- **Script Not Found**: Verify script path and Git branch
- **Import Errors**: Check dependencies and Python environment
- **Permission Denied**: Verify cloud credentials and access permissions

#### Data Access Issues
- **Authentication Failed**: Check cloud credentials configuration
- **Path Not Found**: Verify input data paths and permissions
- **Network Timeout**: Check network connectivity and retry settings

### Debugging Tips

1. **Start Small**: Test with a small dataset first
2. **Check Logs**: Review job logs for error messages
3. **Validate Inputs**: Ensure input data is accessible and valid
4. **Test Locally**: Run scripts locally before submitting to Studio
5. **Monitor Resources**: Check CPU, memory, and storage usage

## Next Steps

- Learn how to [monitor job execution](monitor-jobs.md)
- Set up [job notifications](../../webhooks.md)
- Explore [advanced job configurations](../../../guide/processing.md)
- Configure [team job sharing](../team-collaboration.md)