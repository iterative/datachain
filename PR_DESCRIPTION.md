# Add String Support for `partition_by` in `agg` Method

## Problem

The `partition_by` parameter in the `agg` method only supported `Column` and `C("name")` notation, but not simple string notation like `"file.path"`. This was inconsistent with the `group_by` method, which already supports string notation.

From the Slack discussion:
- Users expected to be able to use strings like `"file.path"` in `partition_by`
- Currently, only `Column` and `C("name")` notation worked
- This created an inconsistent API where `group_by` supported strings but `agg` did not

## Solution

This PR adds string support to the `partition_by` parameter in the `agg` method by:

1. **Updating `PartitionByType`**: Extended the type definition to include `str` and sequences containing strings:
   ```python
   PartitionByType = Union[
       str,
       Function,
       ColumnElement,
       Sequence[Union[str, Function, ColumnElement]],
   ]
   ```

2. **Adding String Conversion Logic**: Modified the `agg` method to convert strings to `Column` objects before passing them to the underlying UDF steps, similar to how `group_by` handles strings:
   ```python
   # Convert string partition_by parameters to Column objects
   if isinstance(col, str):
       col_db_name = ColumnMeta.to_db_name(col)
       col_type = self.signals_schema.get_column_type(col_db_name)
       column = Column(col_db_name, python_to_sql(col_type))
   ```

## Changes Made

### `src/datachain/query/dataset.py`
- Updated `PartitionByType` to include `str` and sequences of strings
- Maintains backward compatibility with existing `Function` and `ColumnElement` types

### `src/datachain/lib/dc/datachain.py`
- Added string-to-Column conversion logic in the `agg` method
- Uses the same conversion pattern as the `group_by` method for consistency
- Processes both single strings and sequences of mixed types (strings, Functions, ColumnElements)

## Usage Examples

Now users can use all of these syntaxes:

```python
# 1. Simple string notation (NEW)
chain.agg(
    total=lambda values: [sum(values)],
    partition_by="category",
    params="value",
    output=int,
)

# 2. Sequence of strings (NEW)
chain.agg(
    total=lambda values: [sum(values)],
    partition_by=["category", "subcategory"],
    params="value",
    output=int,
)

# 3. Mixed sequences (NEW)
chain.agg(
    total=lambda values: [sum(values)],
    partition_by=["category", C("subcategory")],
    params="value",
    output=int,
)

# 4. Original C() notation (UNCHANGED)
chain.agg(
    total=lambda values: [sum(values)],
    partition_by=C("category"),
    params="value",
    output=int,
)

# 5. Nested column paths (NEW)
chain.agg(
    total=lambda values: [sum(values)],
    partition_by="file.path",  # Works with nested paths
    params="value",
    output=int,
)
```

## Benefits

1. **Consistency**: Makes `partition_by` API consistent with `group_by` method
2. **Usability**: Simpler, more intuitive syntax for common use cases
3. **Backward Compatibility**: All existing code continues to work unchanged
4. **Flexibility**: Supports mixing strings with Column objects in sequences

## Testing

- Added comprehensive test cases covering all supported syntaxes
- Verified backward compatibility with existing `C()` notation
- Tested both single strings and sequences of mixed types

## Breaking Changes

None. This is a purely additive change that maintains full backward compatibility.
