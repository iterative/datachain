# Bug Fix: Handle Invalid Python Identifiers in HuggingFace Dataset Column Names

## Issue Description

**GitHub Issue**: [#1204](https://github.com/iterative/datachain/issues/1204) - Reading from huggingface dataset

### Problem
Some datasets on the HuggingFace Hub have column names that are not valid Python identifiers (e.g., containing `?`, `-`, spaces, or starting with numbers). When using `dc.read_hf()` with such datasets, the function would fail with a `KeyError` because the system was trying to access these invalid column names directly as Python attribute names.

### Example Reproduction
```python
import datachain as dc
from datasets import load_dataset

ds = load_dataset('toxigen/toxigen-data', 'annotated', split='test')
dc.read_hf(ds)  # This would fail with KeyError: 'factual?'
```

### Root Cause
The issue was in the `read_hf()` function in `src/datachain/lib/dc/hf.py`. When creating the data model using `dict_to_data_model()`, the function was not passing the `original_names` parameter, which is required for the system to properly handle column names that aren't valid Python identifiers.

## Solution

### Code Changes

1. **Modified `src/datachain/lib/dc/hf.py`**:
   - In the `read_hf()` function, extract the original column names from the HuggingFace dataset features
   - Pass these original names to `dict_to_data_model()` function
   - Also fixed the nested dictionary case in `_feature_to_chain_type()` function

2. **Updated test files**:
   - Added comprehensive tests in `tests/unit/lib/test_hf.py`
   - Created functional tests in `tests/func/test_hf_invalid_column_names.py`

### Key Changes

#### In `read_hf()` function:
```python
# Before (line 64):
model = dict_to_data_model(model_name, output)

# After:
original_names = list(hf_features.keys())
model = dict_to_data_model(model_name, output, original_names)
```

#### In `_feature_to_chain_type()` function:
```python
# Before (line 160):
return dict_to_data_model(name, sequence_dict)

# After:
original_names = list(val.keys())
return dict_to_data_model(name, sequence_dict, original_names)
```

### How It Works

The `dict_to_data_model()` function in `src/datachain/lib/data_model.py` has built-in support for handling invalid Python identifiers through the `original_names` parameter:

1. It normalizes column names using `normalize_col_names()` to create valid Python identifiers
2. It uses Pydantic's `validation_alias=AliasChoices()` to map between normalized names and original names
3. This allows the system to access data using normalized field names while preserving the original column names for data access

### Column Name Transformations

The `normalize_col_names()` function handles various invalid characters:
- `factual?` → `factual_`
- `user-name` → `user_name`
- `123column` → `c0_123column`
- `has spaces` → `has_spaces`
- `with.dots` → `with_dots`
- `with/slashes` → `with_slashes`

## Testing

### Unit Tests
- `test_hf_invalid_column_names()`: Tests the basic functionality with invalid column names
- `test_hf_invalid_column_names_with_read_hf()`: Tests the `read_hf()` function directly
- `test_hf_sequence_dict_with_invalid_names()`: Tests nested dictionary features with invalid names

### Functional Tests
- `test_hf_invalid_column_names_functional()`: Comprehensive test with various invalid column name patterns
- `test_toxigen_dataset_simulation()`: Simulates the exact issue reported in the GitHub issue

### Test Coverage
The tests cover:
- Column names with special characters (`?`, `-`, `.`, `/`)
- Column names with spaces
- Column names starting with numbers
- Nested dictionary features with invalid names
- Multiple invalid patterns in a single dataset

## Impact

This fix ensures that:
1. **Compatibility**: DataChain can now work with any HuggingFace dataset, regardless of column naming conventions
2. **Backward Compatibility**: Existing code continues to work unchanged
3. **Data Integrity**: All original data is preserved and accessible
4. **User Experience**: No more mysterious KeyError exceptions when working with common HuggingFace datasets

## Files Modified

1. `src/datachain/lib/dc/hf.py` - Main fix implementation
2. `tests/unit/lib/test_hf.py` - Unit tests
3. `tests/func/test_hf_invalid_column_names.py` - Functional tests

## Verification

The fix has been thoroughly tested and addresses the exact issue described in the GitHub issue. Users can now successfully use `dc.read_hf()` with datasets like `toxigen/toxigen-data` that contain column names with special characters.
