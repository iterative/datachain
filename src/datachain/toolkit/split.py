from datachain import C, DataChain


def train_test_split(dc: DataChain, weights: list[float]) -> list[DataChain]:
    """
    Splits a DataChain into multiple subsets based on the provided weights.

    This function partitions the rows or items of a DataChain into disjoint subsets,
    ensuring that the relative sizes of the subsets correspond to the given weights.
    It is particularly useful for creating training, validation, and test datasets.

    Args:
        dc (DataChain):
            The DataChain instance to split.
        weights (list[float]):
            A list of weights indicating the relative proportions of the splits.
            The weights do not need to sum to 1; they will be normalized internally.
            For example:
            - `[0.7, 0.3]` corresponds to a 70/30 split;
            - `[2, 1, 1]` corresponds to a 50/25/25 split.

    Returns:
        list[DataChain]:
            A list of DataChain instances, one for each weight in the weights list.

    Examples:
        Train-test split:
        ```python
        from datachain import DataChain
        from datachain.toolkit import train_test_split

        # Load a DataChain from a storage source (e.g., S3 bucket)
        dc = DataChain.from_storage("s3://bucket/dir/")

        # Perform a 70/30 train-test split
        train, test = train_test_split(dc, [0.7, 0.3])

        # Save the resulting splits
        train.save("dataset_train")
        test.save("dataset_test")
        ```

        Train-test-validation split:
        ```python
        train, test, val = train_test_split(dc, [0.7, 0.2, 0.1])
        train.save("dataset_train")
        test.save("dataset_test")
        val.save("dataset_val")
        ```

    Note:
        The splits are random but deterministic, based on Dataset `sys__rand` field.
    """
    if len(weights) < 2:
        raise ValueError("Weights should have at least two elements")
    if any(weight < 0 for weight in weights):
        raise ValueError("Weights should be non-negative")

    weights_normalized = [weight / sum(weights) for weight in weights]

    resolution = 2**31 - 1  # Maximum positive value for a 32-bit signed integer.

    return [
        dc.filter(
            C("sys__rand") % resolution
            >= round(sum(weights_normalized[:index]) * resolution),
            C("sys__rand") % resolution
            < round(sum(weights_normalized[: index + 1]) * resolution),
        )
        for index, _ in enumerate(weights_normalized)
    ]
