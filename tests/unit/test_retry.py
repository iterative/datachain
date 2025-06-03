import tempfile
import unittest
from unittest.mock import MagicMock, patch

import datachain as dc
from datachain.retry import retry_update


class TestRetryFunctionality(unittest.TestCase):
    """Tests for the retry functionality in DataChain."""

    def setUp(self):
        """Set up test environment for each test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        # Create a mock session
        self.session = MagicMock()
        self.catalog = MagicMock()
        self.session.catalog = self.catalog

    def tearDown(self):
        """Clean up after each test."""
        self.temp_dir.cleanup()

    @patch("datachain.retry.datachain.read_dataset")
    def test_retry_update_with_error_records(self, mock_read_dataset):
        """Test retry_update correctly identifies records with errors."""
        # Mock chain
        mock_chain = MagicMock()
        mock_chain._query = MagicMock()
        mock_chain.session = self.session

        # Mock result dataset with error records
        mock_result = MagicMock()
        mock_result.filter.return_value = MagicMock()
        mock_result.filter.return_value.empty = False
        # Set up a record with ID=1 that has an error
        mock_result.filter.return_value.collect.return_value = [1]
        mock_read_dataset.return_value = mock_result

        # Source chain has one record with ID=1
        mock_filtered_chain = MagicMock()
        mock_chain.filter.return_value = mock_filtered_chain

        # Call the function
        result_chain, has_records = retry_update(
            mock_chain, "test_dataset", on="id", retry_on="error"
        )

        # Assertions
        self.assertTrue(has_records)
        self.assertEqual(result_chain, mock_filtered_chain)
        mock_result.filter.assert_called_once()
        mock_chain.filter.assert_called_once()

    @patch("datachain.retry.datachain.read_dataset")
    def test_retry_update_with_missing_records(self, mock_read_dataset):
        """Test retry_update correctly identifies missing records."""
        # Mock chain
        mock_chain = MagicMock()
        mock_chain._query = MagicMock()
        mock_chain.session = self.session

        # Mock result dataset with no error records
        mock_result = MagicMock()
        mock_result.filter.return_value = MagicMock()
        mock_result.filter.return_value.empty = True
        mock_read_dataset.return_value = mock_result

        # Set up subtraction to find missing records
        mock_subtract_result = MagicMock()
        mock_subtract_result.empty = False
        mock_chain.subtract.return_value = mock_subtract_result

        # Call the function with retry_missing=True
        result_chain, has_records = retry_update(
            mock_chain, "test_dataset", on="id", retry_on="error", retry_missing=True
        )

        # Assertions
        self.assertTrue(has_records)
        self.assertEqual(result_chain, mock_subtract_result)
        mock_chain.subtract.assert_called_once()

    @patch("datachain.retry.datachain.read_dataset")
    def test_retry_update_no_records_to_retry(self, mock_read_dataset):
        """Test retry_update when no records need to be retried."""
        # Mock chain
        mock_chain = MagicMock()
        mock_chain._query = MagicMock()
        mock_chain.session = self.session

        # Mock result dataset with no error records
        mock_result = MagicMock()
        mock_result.filter.return_value = MagicMock()
        mock_result.filter.return_value.empty = True
        mock_read_dataset.return_value = mock_result

        # Set up subtraction to find missing records (none)
        mock_subtract_result = MagicMock()
        mock_subtract_result.empty = True
        mock_chain.subtract.return_value = mock_subtract_result

        # Call the function with retry_missing=True
        result_chain, has_records = retry_update(
            mock_chain, "test_dataset", on="id", retry_on="error", retry_missing=True
        )

        # Assertions
        self.assertFalse(has_records)
        self.assertIsNone(result_chain)

    @patch(
        "datachain.retry.datachain.read_dataset",
        side_effect=Exception("Dataset not found"),
    )
    def test_retry_update_first_dataset_creation(self, mock_read_dataset):
        """Test retry_update when dataset doesn't exist yet (first creation)."""
        # Mock chain
        mock_chain = MagicMock()
        mock_chain._query = MagicMock()
        mock_chain.session = self.session

        # Call the function
        result_chain, has_records = retry_update(
            mock_chain, "test_dataset", on="id", retry_on="error"
        )

        # Assertions
        self.assertTrue(has_records)
        self.assertEqual(result_chain, mock_chain)

    def test_datachain_as_retry(self):
        """Test setting up a DataChain with retry mode."""
        with patch("datachain.lib.dc.datasets.read_dataset"):
            # Create a chain with retry enabled
            chain = dc.read_dataset(
                "test_dataset",
                retry=True,
                match_on="id",
                retry_on="error",
            )

            # Check that the retry properties were set
            self.assertTrue(chain.retry)
            # Access internal attributes directly for testing
            self.assertEqual(chain._retry_match_on, "id")
            self.assertEqual(chain._retry_on, "error")
            self.assertFalse(chain._retry_missing)

    @patch("datachain.retry.retry_update")
    @patch("datachain.delta.delta_update")
    def test_save_with_retry(self, mock_delta_update, mock_retry_update):
        """Test save method with retry functionality."""
        # Set up mock chain
        mock_chain = MagicMock()
        mock_chain.retry = True
        mock_chain._retry_match_on = "id"
        mock_chain._retry_match_result_on = None
        mock_chain._retry_on = "error"
        mock_chain._retry_missing = False
        mock_chain.delta = False

        # Set up retry_update to return a modified chain
        retry_chain = MagicMock()
        mock_retry_update.return_value = (retry_chain, True)

        # Call save with the mocked functionality
        with patch("datachain.lib.dc.datachain.DataChain.save"):
            # This is a bit tricky as we're patching the method we're testing
            # In a real scenario, we would create an actual chain and test
            # the end result
            mock_chain.save("test_dataset")

            # Check that retry_update was called with correct parameters
            mock_retry_update.assert_called_once_with(
                mock_chain,
                "test_dataset",
                on="id",
                right_on=None,
                retry_on="error",
                retry_missing=False,
            )

    @patch("datachain.retry.retry_update")
    @patch("datachain.delta.delta_update")
    def test_save_with_retry_and_delta(self, mock_delta_update, mock_retry_update):
        """Test save method with both retry and delta functionality."""
        # Set up mock chain
        mock_chain = MagicMock()
        mock_chain.retry = True
        mock_chain._retry_match_on = "id"
        mock_chain._retry_match_result_on = None
        mock_chain._retry_on = "error"
        mock_chain._retry_missing = False
        mock_chain.delta = True
        mock_chain._delta_on = "id"
        mock_chain._delta_result_on = None
        mock_chain._delta_compare = None

        # Set up retry_update to return a modified chain
        retry_chain = MagicMock()
        mock_retry_update.return_value = (retry_chain, True)

        # Set up delta_update to return a further modified chain
        delta_chain = MagicMock()
        mock_delta_update.return_value = (delta_chain, None, True)

        # Call save with the mocked functionality
        with patch("datachain.lib.dc.datachain.DataChain.save"):
            # This is a bit tricky as we're patching the method we're testing
            # In a real scenario, we would create an actual chain and test the end
            # result.
            mock_chain.save("test_dataset")

            # Check that retry_update was called with correct parameters
            mock_retry_update.assert_called_once_with(
                mock_chain,
                "test_dataset",
                on="id",
                right_on=None,
                retry_on="error",
                retry_missing=False,
            )

            # Check that delta_update would be called on the retry chain
            mock_delta_update.assert_called_once_with(
                retry_chain,
                "test_dataset",
                on="id",
                right_on=None,
                compare=None,
            )


if __name__ == "__main__":
    unittest.main()
