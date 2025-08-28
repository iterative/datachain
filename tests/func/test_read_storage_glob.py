import pytest

import datachain as dc
from tests.utils import instantiate_tree

DEEP_TREE = {
    "deep": {
        "level1": {
            "level2": {
                "level3": {
                    "level4": {
                        "data.json": '{"key": "value"}',
                        "info.txt": "information",
                        "config.yaml": "setting: true",
                    },
                    "audio.mp3": "audio data",
                    "video.mp4": "video data",
                },
                "documents": {
                    "report.pdf": "pdf content",
                    "notes.md": "# Notes",
                },
                "images": {
                    "photo1.jpg": "jpeg data",
                    "photo2.png": "png data",
                },
            },
            "backup": {
                "old": {
                    "archive.zip": "zip data",
                },
                "new": {
                    "current.tar": "tar data",
                },
            },
            "temp": {
                "log1.log": "log 1",
                "logfile": "log 2",
                "temp1.tmp": "temp data 1",
                "temp2.tmp": "temp data 2",
            },
        },
        "media": {
            "music": {
                "rock": {
                    "song1.mp3": "rock song 1",
                    "song2.flac": "rock song 2",
                },
                "jazz": {
                    "track1.wav": "jazz track 1",
                    "track2.mp3": "jazz track 2",
                },
            },
            "videos": {
                "movie1.mp4": "movie 1",
                "movie2.avi": "movie 2",
            },
        },
    }
}


def count_dict_items(d):
    if not isinstance(d, dict):
        return 1

    count = 0
    for value in d.values():
        count += count_dict_items(value)
    return count

@pytest.fixture
def deep_structure(tmp_path):
    """Create deep directory structure for testing"""
    instantiate_tree(tmp_path, DEEP_TREE)
    return tmp_path


def test_simple_wildcard(deep_structure):
    """Test single-level wildcard patterns don't recurse inappropriately"""
    tmp_dir = deep_structure

    # Single level wildcard
    result = dc.read_storage(f"{tmp_dir}/deep/level1/temp/*.tmp")
    names = {f.name for f in result.to_values("file")}
    assert names == {"temp1.tmp", "temp2.tmp"}


def test_globstar_recursive_patterns(deep_structure):
    """Test globstar patterns work correctly with deep recursion"""
    tmp_dir = deep_structure

    # Test recursive MP3 matching across all levels
    result = dc.read_storage(f"{tmp_dir}/deep/**/*.mp3")
    files = {f.name for f in result.to_values("file")}
    assert files == {"audio.mp3", "song1.mp3", "track2.mp3"}

    # Test specific path with globstar
    result = dc.read_storage(f"{tmp_dir}/deep/level1/**/level4/*")
    files = {f.name for f in result.to_values("file")}
    assert files == {"config.yaml", "data.json", "info.txt"}
    
    # Test complex globstar pattern
    result = dc.read_storage(f"{tmp_dir}/deep/**/music/**/*.mp3")
    files = {f.name for f in result.to_values("file")}
    assert files == {"song1.mp3", "track2.mp3"}


def test_question_mark_patterns(deep_structure):
    """Test question mark wildcard patterns"""
    tmp_dir = deep_structure
    
    # Test question mark matching single character
    result = dc.read_storage(f"{tmp_dir}/deep/media/videos/movie?.mp4")
    files = {f.name for f in result.to_values("file")}
    assert files == {"movie1.mp4"}
    
    # Test question mark in filename
    result = dc.read_storage(f"{tmp_dir}/deep/level1/temp/temp?.tmp")
    files = {f.name for f in result.to_values("file")}
    assert files == {"temp1.tmp", "temp2.tmp"}
    
    # Test question mark with path components
    result = dc.read_storage(f"{tmp_dir}/deep/media/videos/movie?.avi")
    files = {f.name for f in result.to_values("file")}
    assert files == {"movie2.avi"}


def test_brace_expansion_patterns(deep_structure):
    """Test brace expansion patterns"""
    tmp_dir = deep_structure
    
    # Test brace expansion for audio file extensions
    result = dc.read_storage(f"{tmp_dir}/deep/**/*.{{mp3,wav,flac}}")
    files = {f.name for f in result.to_values("file")}
    assert files == {"audio.mp3", "song1.mp3", "song2.flac", "track1.wav", "track2.mp3"}

    ### TODO
    # Test brace expansion with directory names
    result = dc.read_storage(f"{tmp_dir}/deep/level1/level2/{{documents,images}}/*")
    files = {f.name for f in result.to_values("file")}
    assert files == {"notes.md", "photo1.jpg", "photo2.png", "report.pdf"}
    
    # Test brace expansion for video formats
    result = dc.read_storage(f"{tmp_dir}/deep/media/videos/*.{{mp4,avi,mkv}}")
    files = {f.name for f in result.to_values("file")}
    assert files == {"movie1.mp4", "movie2.avi"}


def test_combined_complex_patterns(deep_structure):
    """Test complex combinations of different glob patterns"""
    tmp_dir = deep_structure
    
    # Test combined globstar, question mark, and brace expansion
    result = dc.read_storage(f"{tmp_dir}/deep/**/level?/**/*.{{json,yaml,txt}}")
    files = {f.name for f in result.to_values("file")}
    assert files == {"config.yaml", "data.json", "info.txt"}

    ### TODO
    # Test complex directory and file matching
    result = dc.read_storage(f"{tmp_dir}/deep/level1/{{backup,temp}}/**/*.*")
    files = {f.name for f in result.to_values("file")}
    assert files == {"archive.zip", "current.tar", "temp1.tmp", "temp2.tmp"}


def test_directory_boundary_respect(deep_structure):
    """Test that patterns respect directory boundaries and don't leak"""
    tmp_dir = deep_structure

    # Test that music pattern doesn't match videos - no mp3 in music dir
    result = dc.read_storage(f"{tmp_dir}/deep/media/music/**/*.mp4")
    assert [f.name for f in result.to_values("file")] == []

    # Test that temp pattern doesn't match backup files - empty
    result = dc.read_storage(f"{tmp_dir}/deep/level1/temp/*.zip")
    assert [f.name for f in result.to_values("file")] == []

    # Test specific level matching - empty
    result = dc.read_storage(f"{tmp_dir}/deep/level1/level2/level3/*.txt")
    assert [f.name for f in result.to_values("file")] == []


def test_very_deep_nesting_patterns(deep_structure):
    """Test patterns work correctly with very deep directory nesting (5+ levels)"""
    tmp_dir = deep_structure
    
    # Test reaching deepest level (level4)
    result = dc.read_storage(f"{tmp_dir}/deep/level1/level2/level3/level4/*.json")
    files = {f.name for f in result.to_values("file")}
    assert files == {"data.json"}
    
    # Test globstar reaching deep levels
    result = dc.read_storage(f"{tmp_dir}/deep/level1/level2/level3/level4/*")
    files = {f.name for f in result.to_values("file")}
    assert files == {"config.yaml", "data.json", "info.txt"}

    ## TODO
    # Test that we can traverse the full depth with globstar
    result = dc.read_storage(f"{tmp_dir}/deep/**/level4/**/*")
    files = {f.name for f in result.to_values("file")}
    assert files == {"config.yaml", "data.json", "info.txt"}


def test_edge_cases_and_empty_results(deep_structure):
    """Test edge cases and patterns that should return no results"""
    tmp_dir = deep_structure
    
    # Test non-existent extension
    result = dc.read_storage(f"{tmp_dir}/deep/**/*.nonexistent")
    assert  [f.name for f in result.to_values("file")] == []
    
    # Test non-existent directory pattern
    result = dc.read_storage(f"{tmp_dir}/deep/nonexistent/*.txt")
    assert  [f.name for f in result.to_values("file")] == []
    
    # Test pattern that would match if recursion was wrong
    result = dc.read_storage(f"{tmp_dir}/deep/media/*.mp3")
    assert  [f.name for f in result.to_values("file")] == []
    
    # Test very specific pattern that shouldn't match
    result = dc.read_storage(f"{tmp_dir}/deep/**/level5/*.txt")
    assert  [f.name for f in result.to_values("file")] == []


def test_pattern_performance_with_large_structure(deep_structure):
    """Test that glob patterns perform reasonably with complex directory structures"""
    tmp_dir = deep_structure
    
    # Test broad globstar pattern
    assert dc.read_storage(f"{tmp_dir}/deep/**/*").count() == count_dict_items(DEEP_TREE)

    # Test that specific patterns are more efficient than broad ones
    result = dc.read_storage(f"{tmp_dir}/deep/**/*.txt")
    assert [f.name for f in result.to_values("file")] == ["info.txt"]


def test_no_pattern_behavior_unchanged(deep_structure):
    """Test that URIs without patterns continue to work as before"""
    tmp_dir = deep_structure

    # Test directory listing without patterns
    result = dc.read_storage(f"{tmp_dir}/deep/media/videos")
    files = {f.name for f in result.to_values("file")}
    assert files == {"movie1.mp4", "movie2.avi"}

    # Test specific directory path
    result = dc.read_storage(f"{tmp_dir}/deep/level1/temp")
    files = {f.name for f in result.to_values("file")}
    assert files == {"temp1.tmp", "temp2.tmp", "log1.log", "logfile"}


def test_mixed_pattern_types(deep_structure):
    """Test mixing different types of patterns in complex ways"""
    tmp_dir = deep_structure
    
    # Mix wildcard and brace expansion
    result = dc.read_storage(f"{tmp_dir}/deep/media/*/rock/*.{{mp3,flac}}")
    files = {f.name for f in result.to_values("file")}
    assert files == {"song1.mp3", "song2.flac"}

    # Mix question mark with globstar
    result = dc.read_storage(f"{tmp_dir}/deep/**/temp?.*")
    files = {f.name for f in result.to_values("file")}
    assert files == {"temp1.tmp", "temp2.tmp"}

    # TODO
    # Complex pattern with all types
    result = dc.read_storage(f"{tmp_dir}/deep/**/level?/{{backup,temp}}/**/*.{{tmp,zip,tar}}")
    files = {f.name for f in result.to_values("file")}
    assert files == {"archive.zip", "current.tar", "temp1.tmp", "temp2.tmp"}
