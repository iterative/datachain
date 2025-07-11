"""
Test for fixing invalid Python identifier column names in Hugging Face datasets.
This addresses the issue reported in https://github.com/iterative/datachain/issues/1204
"""

from datasets import Dataset

from datachain.lib.dc.hf import read_hf


def test_hf_invalid_column_names_functional():
    """Test that read_hf works with datasets containing invalid Python identifier column names."""
    # Create a dataset with column names that aren't valid Python identifiers
    # This simulates the actual datasets from Hugging Face Hub that have such column names
    ds = Dataset.from_dict(
        {
            "factual?": ["yes", "no", "maybe"],
            "user-name": ["alice", "bob", "charlie"],
            "123column": ["value1", "value2", "value3"],
            "valid_name": ["data1", "data2", "data3"],
            "has spaces": ["space1", "space2", "space3"],
            "with.dots": ["dot1", "dot2", "dot3"],
            "with/slashes": ["slash1", "slash2", "slash3"],
        }
    )

    # Mock the read_values function to avoid needing a full session
    import unittest.mock

    with unittest.mock.patch("datachain.lib.dc.values.read_values") as mock_read_values:
        mock_chain = unittest.mock.Mock()
        mock_read_values.return_value = mock_chain

        # This should work without raising KeyError with the fix
        result = read_hf(ds)

        # Verify that read_values was called
        mock_read_values.assert_called_once()

        # Verify that gen was called on the chain
        mock_chain.gen.assert_called_once()

        # Get the HFGenerator and model that were passed to gen
        gen_call = mock_chain.gen.call_args
        hf_generator = gen_call[0][0]

        # The HFGenerator should be set up correctly with the model
        assert hasattr(hf_generator, "output_schema")
        output_schema = hf_generator.output_schema

        # The model should have fields that can handle the original column names
        # Check that we have the expected number of fields
        assert len(output_schema.model_fields) == 7  # All columns should be present

        # Test that the generator can actually process the data
        hf_generator.setup()
        row = next(iter(hf_generator.process()))

        # The field names should remain as the original names
        field_names = list(output_schema.model_fields.keys())
        assert len(field_names) == 7

        # Check that we can access values from the row
        row_dict = row.model_dump()
        assert len(row_dict) == 7  # All values should be accessible

        # Verify that the values are correct (first row)
        # The field names should be the original names
        assert row_dict["factual?"] == "yes"
        assert row_dict["user-name"] == "alice"
        assert row_dict["123column"] == "value1"
        assert row_dict["valid_name"] == "data1"
        assert row_dict["has spaces"] == "space1"
        assert row_dict["with.dots"] == "dot1"
        assert row_dict["with/slashes"] == "slash1"


def test_toxigen_dataset_simulation():
    """Test that simulates the exact issue from the GitHub issue with toxigen dataset."""
    # Create a simplified version of the toxigen dataset structure
    ds = Dataset.from_dict(
        {
            "factual?": ["yes", "no", "maybe"],
            "text": ["sample text 1", "sample text 2", "sample text 3"],
            "target_group": ["group1", "group2", "group3"],
            "annotator_id": [1, 2, 3],
        }
    )

    # Mock the read_values function to avoid needing a full session
    import unittest.mock

    with unittest.mock.patch("datachain.lib.dc.values.read_values") as mock_read_values:
        mock_chain = unittest.mock.Mock()
        mock_read_values.return_value = mock_chain

        # This should work without raising KeyError: 'factual?'
        result = read_hf(ds)

        # Verify that the function completed without error
        mock_read_values.assert_called_once()
        mock_chain.gen.assert_called_once()

        # Get the HFGenerator
        hf_generator = mock_chain.gen.call_args[0][0]

        # Test processing
        hf_generator.setup()
        row = next(iter(hf_generator.process()))

        # The row should contain all the data
        row_dict = row.model_dump()
        assert len(row_dict) == 4

        # The 'factual?' column should be accessible with its original name
        assert row_dict["factual?"] == "yes"
        assert row_dict["text"] == "sample text 1"
        assert row_dict["target_group"] == "group1"
        assert row_dict["annotator_id"] == 1
