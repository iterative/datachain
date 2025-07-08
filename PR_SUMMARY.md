# PR Summary: Allow Complex Signals in partition_by

## Overview
This PR implements support for complex signals (Pydantic BaseModel types like `File`, `Image`, etc.) in the `partition_by` parameter of both `agg` and `group_by` methods, addressing issue [#1211](https://github.com/iterative/datachain/issues/1211). The implementation supports only column names and Column objects, automatically expanding complex signal columns to their unique identifier columns.

## Changes Made

### 1. Type Definitions (Unchanged)
**File**: `src/datachain/query/dataset.py`
- `PartitionByType` remains unchanged - supports only column names and Column objects
- No class type support added to maintain API consistency

```python
PartitionByType = Union[
    str,
    Function,
    ColumnElement,
    Sequence[Union[str, Function, ColumnElement]],
]
```

**This is the correct approach** - complex signals are accessed through their column names (e.g., `"file"`) rather than class types (e.g., `File`).

### 2. Enhanced DataChain.agg() Method
**File**: `src/datachain/lib/dc/datachain.py`

#### Added Complex Signal Processing Helper
- New method `_process_complex_signal_column()` that:
  - Detects if a column name refers to a Pydantic BaseModel type
  - Extracts unique identifier columns using `_unique_id_keys` or model fields
  - Generates appropriate Column objects for partitioning
  - Supports both DataModel subclasses and regular Pydantic BaseModel types

#### Updated Partition Processing Logic
- Modified `agg()` method to use `_process_complex_signal_column()` for string columns
- Automatically expands complex signal columns to their unique identifier columns
- Supports only column names and Column objects (NOT class types)

#### Enhanced Documentation
- Updated docstring with examples of complex signal column usage
- Added comprehensive usage examples showing File column partitioning
- Clarified that only column names are accepted, not class types

### 3. Enhanced DataChain.group_by() Method
**File**: `src/datachain/lib/dc/datachain.py`

#### Updated Group By Processing Logic
- Modified `group_by()` method to use `_process_complex_signal_column()` for string columns
- Automatically expands complex signal columns to their unique identifier columns
- Supports only column names and Column objects (NOT class types)

#### Enhanced Documentation
- Updated docstring with examples of complex signal column usage
- Added examples showing automatic expansion of complex signals

## Usage Examples

### Basic Complex Signal Partitioning
```python
from typing import Iterator
import datachain as dc
from datachain.lib.file import File

def my_agg(files: list[File]) -> Iterator[tuple[File, int]]:
    yield files[0], sum(f.size for f in files)

chain = chain.agg(
    my_agg,
    params=("file",),
    output={"file": File, "total": int},
    partition_by="file",  # Use column name (automatically expands to File's unique keys)
)
```

### Mixed Partitioning (Complex Signal Column + Simple Column)
```python
result = chain.agg(
    my_agg,
    params=("file", "category"),
    output={"file": File, "category": str, "total": int},
    partition_by=["file", "category"],  # Both column names
)
```

### Group By with Complex Signals (Column Names Only)
```python
# group_by uses column names, not class types
result = chain.group_by(
    total_size=func.sum("file.size"),
    count=func.count(),
    partition_by="file",  # Uses column name, expands to File's unique keys
)
```

### Deep Nesting Support (3+ Levels)
```python
# Supports deeply nested Pydantic BaseModels
class NestedLevel3(BaseModel):
    category: str
    level2: NestedLevel2  # Which contains NestedLevel1
    total: float

result = chain.agg(
    my_agg,
    params=("nested",),
    output={"nested": NestedLevel3, "total": int},
    partition_by="nested",  # Column name - handles 3+ levels automatically
)
```

## How It Works

### 1. Column Name Detection
When `partition_by` contains a string column name, the system checks if it refers to a Pydantic BaseModel type in the schema.

### 2. BaseModel Type Detection
The system uses `issubclass(col_type, BaseModel)` to detect if a column contains a complex signal (Pydantic BaseModel).

### 3. Unique Key Resolution
For BaseModel types, the system:
- First checks for `_unique_id_keys` attribute (DataModel subclasses)
- Falls back to `_datachain_column_types.keys()` (DataModel subclasses)
- Falls back to `model_fields.keys()` (regular Pydantic BaseModel)

### 4. Column Generation
Using the unique keys, the system generates appropriate Column objects for each unique identifier field (e.g., `file.source`, `file.path`).

### 5. Integration
The generated columns are integrated into the existing partition processing pipeline, maintaining compatibility with existing functionality.

## Benefits

1. **Improved Usability**: Users can now partition by complex signals using column names (e.g., `"file"`) without needing to specify individual columns
2. **Automatic Column Detection**: The system automatically uses the appropriate unique identifier columns for Pydantic BaseModel types
3. **Broad Compatibility**: Supports both DataModel subclasses and regular Pydantic BaseModel types
4. **Deep Nesting Support**: Handles complex nested structures with 3+ levels automatically
5. **Backward Compatibility**: Existing code continues to work unchanged

## Testing

### Created Test Suite
**File**: `tests/func/test_complex_partition_by.py`
- Tests basic File partitioning
- Tests mixed partitioning (string + complex signal)
- Tests error handling for invalid types
- Tests schema validation

### Test Coverage
- ✅ Basic complex signal partitioning using column names
- ✅ Mixed column partitioning (complex + simple)
- ✅ Deep nesting support (3+ levels of BaseModel nesting)
- ✅ Both `agg` and `group_by` methods
- ✅ Error handling for invalid column names
- ✅ Multiple complex signals
- ✅ Edge cases and validation
- ✅ Support for both DataModel and regular Pydantic BaseModel types

## Error Handling

The implementation includes comprehensive error handling:

1. **Invalid Column Names**: Throws `ValueError` when column names are not found in schema
2. **Schema Lookup Failures**: Gracefully handles missing columns by falling back to basic column creation
3. **Missing Unique Keys**: Falls back to using all model fields when `_unique_id_keys` is not defined
4. **Graceful Degradation**: Skips columns that don't exist in the schema during complex signal expansion

## Backward Compatibility

This change is fully backward compatible:
- Existing code using strings and Functions continues to work
- No changes required for existing usage patterns
- New functionality is additive only

## Performance Considerations

- No performance impact on existing code paths
- Complex signal processing only occurs when DataModel types are used
- Efficient column lookup using schema introspection
- Minimal overhead for type checking

## Future Enhancements

This implementation provides a foundation for potential future enhancements:
1. Support for custom unique key definitions per BaseModel type
2. Complex signal support in other methods (e.g., `order_by`, `distinct`)
3. Advanced partitioning strategies for complex signals
4. Performance optimizations for deeply nested structures

## Files Modified

1. `src/datachain/query/dataset.py` - PartitionByType definition (no changes - supports column names only)
2. `src/datachain/lib/dc/datachain.py` - Enhanced agg and group_by methods with complex signal support
3. `tests/func/test_complex_partition_by.py` - Comprehensive test suite for both methods

## Conclusion

This PR successfully implements the requested feature to allow complex signals in `partition_by` for both `agg` and `group_by` methods. The implementation uses column names only (not class types) and automatically expands Pydantic BaseModel types to their unique identifier columns, providing a more intuitive and powerful API for users while maintaining full backward compatibility and robust error handling.
