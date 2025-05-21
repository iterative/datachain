#!/usr/bin/env python
"""
File Generator Helper

This helper creates numbered text files in a 'test' directory each time it runs.
The files follow the naming pattern: file-0.txt, file-1.txt, file-2.txt, etc.

Each execution, the script:

1. Creates the 'test' directory if it doesn't exist
2. Finds the highest numbered file currently present
3. Creates a new file with the next number in sequence
4. Adds timestamped content to the file
"""

import re
import time
from pathlib import Path


def generate_next_file() -> Path:
    """
    Generate (appends) a new numbered text file in the 'test' directory.
    """
    test_dir = Path("test")
    test_dir.mkdir(exist_ok=True)

    max_num = -1
    for file in test_dir.glob("file-*.txt"):
        if file.is_file():
            match = re.search(r"file-(\d+)\.txt", file.name)
            if match:
                max_num = max(max_num, int(match.group(1)))

    next_num = max_num + 1
    new_file_path = test_dir / f"file-{next_num}.txt"
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    content = f"This is file number {next_num}\nCreated at: {timestamp}\n"
    new_file_path.write_text(content)

    return new_file_path
