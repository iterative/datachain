# Retry Processing

The retry functionality in DataChain allows for selective reprocessing of records that either have errors or are missing from the final result dataset. This powerful feature works alongside the existing `delta` functionality, providing both incremental processing of new data and reprocessing of problematic records.

## Overview

When working with large datasets, two common scenarios arise:

1. **Processing new data incrementally**: You only want to process new data that has been added since the last run (handled by the `delta=True` functionality)
2. **Reprocessing problematic records**: Some records may have failed processing or produced errors that need to be fixed and reprocessed

The retry functionality addresses the second scenario while still supporting the first.

## How It Works

The retry functionality allows you to:

1. **Identify error records**: Records in the result dataset with a non-NULL value in a specified error field
2. **Identify missing records**: Records that exist in the source dataset but are missing from the result dataset
3. **Reprocess only these records**: Applying your full processing chain only to these records that need attention

## Usage

You can enable retry functionality in two ways:

### 1. Using read_storage() or read_dataset()

```python
import datachain as dc
from datachain import C

chain = (
    dc.read_storage(
        "path/to/data/",
        # Enable delta processing to handle only new files
        delta=True,
        # Enable retry processing to handle errors
        retry=True,
        # Field(s) that uniquely identify records in the source dataset
        match_on="id",
        # Name of the field in result dataset that indicates an error when not None
        retry_on="error",
        # Whether to also retry records missing from the result
        retry_missing=True
    )
    .map(result=process_function)  # Your processing function
    .save(name="processed_data")    # Save results
)
```

### 2. Using the DataChain._as_retry() method

```python
import datachain as dc

# Create a chain
chain = dc.read_storage("path/to/data/")

# Configure retry mode
chain = chain._as_retry(
    on="id",                 # Field that uniquely identifies records
    retry_on="error",        # Field in result that indicates errors when not None
    retry_missing=True       # Whether to retry missing records too
)

# Process and save
result = (
    chain
    .map(result=process_function)
    .save(name="processed_data")
)
```

## Parameters

- **retry**: Boolean flag to enable retry functionality
- **match_on**: Field(s) in source dataset that uniquely identify records
- **match_result_on**: Corresponding field(s) in result dataset if they differ from source
- **retry_on**: Field in result dataset that indicates an error when not None
- **retry_missing**: If True, also include records missing from result dataset

## Example: Processing Files with Error Handling

```python
import datachain as dc
from datachain import C

def process_file(file):
    """Process a file - may occasionally fail with an error."""
    try:
        # Your processing logic here
        content = file.read_text()
        result = analyze_content(content)
        return {
            "content": content,
            "result": result,
            "error": None  # No error
        }
    except Exception as e:
        # Log the error and return it in the result
        return {
            "content": None,
            "result": None,
            "error": str(e)  # Store the error message
        }

# Process files with both delta and retry functionality
chain = (
    dc.read_storage(
        "data/",
        delta=True,           # Process only new files
        retry=True,           # Reprocess files with errors
        match_on="file.path", # Files are identified by their paths
        retry_on="error",     # Errors are stored in the "error" field
        retry_missing=True    # Also process any missing files
    )
    .map(result=process_file)
    .save(name="processed_files")
)

# Show records with errors that will be retried next time
error_records = chain.filter(C("error") != None)
if not error_records.empty:
    print("Records with errors that will be retried:")
    error_records.show()
```

## Combining Delta and Retry

The real power comes when combining both delta and retry processing:

1. **Delta processing**: Only process new or modified records
2. **Retry processing**: Reprocess any records that had errors previously

When both are enabled, DataChain will:

1. First identify records that need to be retried (based on errors or missing records)
2. Then apply delta processing to that subset
3. Process only the resulting records, avoiding unnecessary reprocessing of unchanged data

This provides the most efficient way to maintain a dataset that is always up-to-date and free of processing errors.
