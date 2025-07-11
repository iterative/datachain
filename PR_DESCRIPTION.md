# Fix HuggingFace dataset column name handling for invalid identifiers

## 🐛 Issue Fixed

Fixes #1204 - Reading from huggingface dataset fails with `KeyError` when column names contain invalid Python identifiers.

## 📋 Problem Description

Some datasets on the HuggingFace Hub have column names that are not valid Python identifiers (e.g., containing `?`, `-`, spaces, or starting with numbers). When using `dc.read_hf()` with such datasets, the function would fail with a `KeyError` because the system was trying to access these invalid column names directly as Python attribute names.

### Example that failed before this fix:
```python
import datachain as dc
from datasets import load_dataset

ds = load_dataset('toxigen/toxigen-data', 'annotated', split='test')
dc.read_hf(ds)  # KeyError: 'factual?'
```

## 🔧 Solution

The fix leverages the existing `normalize_col_names()` function and `dict_to_data_model()` support for handling invalid Python identifiers by passing the `original_names` parameter.

### Changes Made:

1. **`src/datachain/lib/dc/hf.py`** - Main fix in `read_hf()` function:
   - Extract original column names from HuggingFace dataset features
   - Pass these original names to `dict_to_data_model()` function

2. **`src/datachain/lib/hf.py`** - Fix in `_feature_to_chain_type()` function:
   - Handle nested dictionary features with invalid column names
   - Pass original names for nested structures

3. **Comprehensive test coverage** - Added tests in both unit and functional test files

## 🧪 Testing

### Unit Tests (`tests/unit/lib/test_hf.py`):
- `test_hf_invalid_column_names()` - Tests basic functionality with invalid column names
- `test_hf_invalid_column_names_with_read_hf()` - Tests the `read_hf()` function directly
- `test_hf_sequence_dict_with_invalid_names()` - Tests nested dictionary features with invalid names
- Updated all existing tests to use the `original_names` parameter

### Functional Tests (`tests/func/test_hf_invalid_column_names.py`):
- `test_hf_invalid_column_names_functional()` - Comprehensive test with various invalid column name patterns
- `test_toxigen_dataset_simulation()` - Simulates the exact issue from the GitHub issue

### Test Coverage:
- Column names with special characters (`?`, `-`, `.`, `/`)
- Column names with spaces
- Column names starting with numbers
- Nested dictionary features with invalid names
- Multiple invalid patterns in a single dataset

## 📊 Column Name Transformations

The fix automatically transforms invalid column names to valid Python identifiers:
- `factual?` → `factual_`
- `user-name` → `user_name`  
- `123column` → `c0_123column`
- `has spaces` → `has_spaces`
- `with.dots` → `with_dots`
- `with/slashes` → `with_slashes`

## ✅ Benefits

1. **Compatibility**: DataChain can now work with any HuggingFace dataset, regardless of column naming conventions
2. **Backward Compatibility**: Existing code continues to work unchanged
3. **Data Integrity**: All original data is preserved and accessible
4. **User Experience**: No more mysterious KeyError exceptions when working with common HuggingFace datasets

## 🔍 How It Works

The `dict_to_data_model()` function has built-in support for handling invalid Python identifiers:
1. It normalizes column names using `normalize_col_names()` to create valid Python identifiers
2. It uses Pydantic's `validation_alias=AliasChoices()` to map between normalized names and original names
3. This allows the system to access data using normalized field names while preserving the original column names

## 📁 Files Modified

- `src/datachain/lib/dc/hf.py` - Main fix implementation
- `src/datachain/lib/hf.py` - Nested dictionary fix  
- `tests/unit/lib/test_hf.py` - Unit tests
- `tests/func/test_hf_invalid_column_names.py` - Functional tests
- `BUGFIX_SUMMARY.md` - Detailed documentation

## 🧪 Manual Testing

The fix can be verified with:
```python
import datachain as dc
from datasets import Dataset

# Create dataset with invalid column names
ds = Dataset.from_dict({
    "factual?": ["yes", "no"],
    "user-name": ["alice", "bob"],
    "123column": ["value1", "value2"]
})

# This now works without error
chain = dc.read_hf(ds)
```

## 📚 References

- Original issue: https://github.com/iterative/datachain/issues/1204
- Similar pattern used in other parts of the codebase for handling invalid identifiers
- Follows existing test patterns in `test_hf.py`