# Data Processing Features

DataChain offers specialized data processing features that help optimize the handling of large and evolving datasets:

## Key Processing Features

- [Delta Processing](./delta.md): Efficiently process only new or changed records between dataset versions
- [Retry Processing](./retry.md): Automatically identify and reprocess records with errors or missing records

These features can be used independently or combined to create powerful data processing workflows.

## Combining Delta and Retry

When used together, delta and retry processing create a comprehensive approach to data processing:

1. **Delta processing** ensures you only process what's new or changed
2. **Retry processing** ensures you can fix errors without reprocessing the entire dataset

This combination is ideal for production environments where both efficiency and data quality are essential.

## Example: Combined Approach

```python
import datachain as dc
from datachain import C

def process_record(record):
    """Process a record, potentially resulting in errors."""
    try:
        # Your processing logic here
        result = complex_analysis(record)
        return {
            "result": result,
            "error": None  # No error
        }
    except Exception as e:
        # Return the error
        return {
            "result": None,
            "error": str(e)
        }

# Process only new files AND retry any files with errors
chain = (
    dc.read_storage(
        "data/",
        update=True,          # Update the storage index
        delta=True,           # Process only new files (delta)
        delta_on="file.path", # Identify records by path
        retry=True,           # Reprocess files with errors (retry)
        match_on="file.path", # Same identifier for retry
        retry_on="error",     # Field that indicates errors
        retry_missing=True    # Also process missing records
    )
    .map(result=process_record)
    .save(name="processed_data")
)
```

For detailed information about these features, see their dedicated documentation:

- [Delta Processing](./delta.md)
- [Retry Processing](./retry.md)
