"""
Functional tests for UDF (User-Defined Functions) behavior.

This module tests:
- UDF parameter handling with nested DataModels
- None value handling for complex objects (e.g., after outer joins)
- UDF execution with various prefetch settings
- DataModel validation and reconstruction in UDF context

Tests that should potentially be migrated here from test_datachain.py:
- test_udf (basic UDF functionality)
- test_udf_parallel (parallel execution)
- test_class_udf (Mapper class usage)
- test_class_udf_parallel (parallel Mapper execution)
- test_udf_after_limit (UDF behavior with query limits)
- test_udf_different_types (type handling in UDFs)
- test_map_file (file object handling)
- test_gen_file (generator UDFs with files)
- test_batch_for_map (batching behavior)

Note: Error handling tests (test_udf_*_error, test_udf_*_interrupt) and
distributed tests can remain in test_datachain.py as they test infrastructure
rather than UDF parameter/validation behavior.
"""

import datachain as dc
from datachain.lib.file import AudioFile, AudioFragment


def test_udf_none_nested_datamodel_after_outer_merge(test_session):
    """
    Test that UDFs can handle None values for nested DataModel objects
    """
    # Get warehouse default values for proper NULL handling checks
    catalog = test_session.catalog
    from datachain.sql.types import Int, String

    int_default = Int.default_value(catalog.warehouse.db.dialect)
    str_default = String.default_value(catalog.warehouse.db.dialect)

    # Create sample data with AudioFragment (which has nested AudioFile)
    left = dc.read_values(
        id=[1, 2],
        audio_fragment=[
            AudioFragment(
                audio=AudioFile(path="audio1.wav", source="file://"),
                start=0.0,
                end=1.0,
            ),
            AudioFragment(
                audio=AudioFile(path="audio2.wav", source="file://"),
                start=0.0,
                end=1.0,
            ),
        ],
        session=test_session,
    )

    right = dc.read_values(
        id=[2, 3],
        audio_info=[
            AudioFragment(
                audio=AudioFile(path="audio2_right.wav", source="file://"),
                start=1.0,
                end=2.0,
            ),
            AudioFragment(
                audio=AudioFile(path="audio3_right.wav", source="file://"),
                start=0.0,
                end=1.0,
            ),
        ],
        session=test_session,
    )

    # Full outer merge creates None (or default values in CH) for unmatched sides
    merged = left.merge(right, on="id", full=True)
    assert merged.count() == 3

    def extract_paths(
        id: int, right_id: int, audio_fragment: AudioFragment, audio_info: AudioFragment
    ) -> tuple:
        left_path = audio_fragment.audio.path if audio_fragment else None
        right_path = audio_info.audio.path if audio_info else None
        extracted_file = audio_fragment.audio if audio_fragment else None
        return id, right_id, left_path, right_path, extracted_file

    result = merged.settings(prefetch=False).map(
        extract_paths,
        params=["id", "right_id", "audio_fragment", "audio_info"],
        output={
            "merged_id": int,
            "merged_right_id": int,
            "left_path": str,
            "right_path": str,
            "extracted_file": AudioFile,
        },
    )
    rows = sorted(
        result.select(
            "merged_id",
            "merged_right_id",
            "left_path",
            "right_path",
            "extracted_file",
        ).to_iter(),
        key=lambda r: (
            r[0] if r[0] is not None else -1,
            r[1] if r[1] is not None else -1,
        ),
    )

    assert len(rows) == 3

    # Row with right-only data: left id=int_default (NULL), right_id=3
    assert rows[0][0] == int_default  # merged_id (NULL → int_default)
    assert rows[0][1] == 3  # merged_right_id
    assert rows[0][2] == str_default  # left_path (NULL → str_default)
    assert rows[0][3] == "audio3_right.wav"  # right_path
    # extracted_file: NULL AudioFile → None on SQLite, default AudioFile on ClickHouse
    if rows[0][4] is None:
        # SQLite: NULL object becomes None
        assert rows[0][4] is None
    else:
        # ClickHouse: NULL object becomes object with default values
        assert isinstance(rows[0][4], AudioFile)
        assert rows[0][4].path == str_default

    # Row with left-only data: left id=1, right_id=0 (NULL)
    assert rows[1][0] == 1
    assert rows[1][1] == int_default
    assert rows[1][2] == "audio1.wav"
    assert rows[1][3] == str_default
    assert isinstance(rows[1][4], AudioFile)
    assert rows[1][4].path == "audio1.wav"
    assert rows[1][4].source == "file://"

    # Row with matched data: id=2, right_id=2 (both sides present)
    assert rows[2][0] == 2
    assert rows[2][1] == 2
    assert rows[2][2] == "audio2.wav"
    assert rows[2][3] == "audio2_right.wav"
    assert isinstance(rows[2][4], AudioFile)
    assert rows[2][4].path == "audio2.wav"
    assert rows[2][4].source == "file://"
