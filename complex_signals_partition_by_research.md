# Complex Signals in partition_by Research

## Issue Analysis
The issue [#1211](https://github.com/iterative/datachain/issues/1211) asks for support of complex signals in `partition_by` parameter of the `agg` method. Currently, `partition_by` only supports:

1. String column names (e.g., `"file"`)
2. Function objects (e.g., `C("file.path")`)
3. ColumnElement objects

## What are Complex Signals?
Complex signals are data model types like `File`, `Image`, `Video`, etc. that are composed of multiple fields/columns in the database. For example, a `File` object has fields like:
- `source` (str)
- `path` (str)
- `size` (int)
- `etag` (str)
- `version` (str)
- `is_latest` (bool)
- `last_modified` (datetime)
- `location` (JSON)

## Current Implementation
The `agg` method in `src/datachain/lib/dc/datachain.py` (lines 759-825) processes `partition_by` as follows:

```python
# Convert string partition_by parameters to Column objects
processed_partition_by = partition_by
if partition_by is not None:
    if isinstance(partition_by, (str, Function, ColumnElement)):
        list_partition_by = [partition_by]
    else:
        list_partition_by = list(partition_by)

    processed_partition_columns: list[ColumnElement] = []
    for col in list_partition_by:
        if isinstance(col, str):
            col_db_name = ColumnMeta.to_db_name(col)
            col_type = self.signals_schema.get_column_type(col_db_name)
            column = Column(col_db_name, python_to_sql(col_type))
            processed_partition_columns.append(column)
        elif isinstance(col, Function):
            column = col.get_column(self.signals_schema)
            processed_partition_columns.append(column)
        else:
            # Assume it's already a ColumnElement
            processed_partition_columns.append(col)
```

## Proposed Solution
To support complex signals, I need to:

1. **Detect complex signal types**: Check if the partition_by parameter is a DataModel type (like `File`)
2. **Find representative columns**: For complex signals, find the columns that uniquely identify the signal
3. **Convert to appropriate columns**: Generate the necessary Column objects for partitioning

### Key Components:

1. **Type Detection**: Check if the partition_by parameter is a DataModel subclass
2. **Column Resolution**: Use the `_unique_id_keys` property of DataModel classes to determine which columns to use for partitioning
3. **Schema Integration**: Ensure the generated columns are properly typed using the signals_schema

### Implementation Strategy:
- Extend the `PartitionByType` to include DataModel types
- Add logic to process DataModel types in the `agg` method
- Use the `_unique_id_keys` from the DataModel to determine which columns to use for partitioning
- Generate appropriate Column objects for each unique key

## Files to Modify:
1. `src/datachain/query/dataset.py` - Update `PartitionByType` to include DataModel types
2. `src/datachain/lib/dc/datachain.py` - Add logic to handle complex signals in the `agg` method
3. Tests to verify the functionality

## Test Cases:
1. Basic complex signal partitioning with File objects
2. Multiple complex signals in partition_by
3. Mixed partition_by with strings and complex signals
4. Error handling for invalid complex signal types