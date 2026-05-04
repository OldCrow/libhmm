#!/usr/bin/env python3
"""Verify that all C/C++ header files use #pragma once.

libhmm uses #pragma once as the header guard convention. This hook
prevents regressions when new headers are added.
"""

import sys
from pathlib import Path

exit_code = 0

for path_str in sys.argv[1:]:
    path = Path(path_str)
    if not path.exists():
        continue  # deleted in this commit

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        continue

    if not any(line.startswith("#pragma once") for line in text.splitlines()):
        print(f"ERROR: {path} is missing '#pragma once'")
        print("       libhmm uses #pragma once as the header guard convention.")
        print("       Add '#pragma once' as the first line of the file.")
        exit_code = 1

sys.exit(exit_code)
