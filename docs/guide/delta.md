# Delta Processing

Delta processing allows you to optimize the creation of new dataset versions by calculating and processing only the differences (deltas) between dataset versions. This is especially useful when working with large datasets that change incrementally over time.

## Overview

When working with large datasets that receive regular updates, reprocessing the entire dataset each time is inefficient. The `delta=True` flag in DataChain enables incremental processing, where only new or changed records are processed.

## How Delta Processing Works

When delta processing is enabled:

1. DataChain calculates the difference between the latest version of your source dataset and the version used to create the most recent version of your result dataset
2. Only the records that are new or modified in the source dataset are processed
3. These processed records are then merged with the latest version of your result dataset

This optimization significantly reduces processing time and resource usage for incremental updates.

## Usage

Delta processing can be enabled through the `delta` parameter when using `read_storage()` or `read_dataset()`:

```python
import datachain as dc

# Process only new or changed files
chain = (
    dc.read_storage(
        "data/",
        delta=True,                    # Enable delta processing
        delta_on="file.path",          # Field that uniquely identifies records
        delta_compare="file.mtime"     # Field to check for changes
    )
    .map(result=process_function)
    .save(name="processed_data")
)
```

## Parameters

- **delta**: Boolean flag to enable delta processing
- **delta_on**: Field(s) that uniquely identify rows in the source dataset
- **delta_result_on**: Field(s) in the resulting dataset that correspond to `delta_on` fields (if they have different names)
- **delta_compare**: Field(s) used to check if a record has been modified

## Example: Processing New Files Only

```python
import datachain as dc
import time

def process_file(file):
    """Process a file and return results."""
    content = file.read_text()
    # Simulate processing time
    time.sleep(0.1)
    return {
        "content": content,
        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

# Process only new or modified files
chain = (
    dc.read_storage(
        "data/",
        update=True,                   # Update the storage index
        delta=True,                    # Process only new files
        delta_on="file.path"           # Files are identified by their paths
    )
    .map(result=process_file)
    .save(name="processed_files")
)

print(f"Processed {chain.count()} files")
```

## Combining with Retry Processing

Delta processing can be combined with [retry processing](./retry.md) to create a powerful workflow that both:

1. Processes only new or changed records (delta)
2. Reprocesses records with errors or that are missing (retry)
