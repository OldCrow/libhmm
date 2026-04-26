#!/bin/bash

# Check that all header files use #pragma once
# libhmm uses #pragma once as the header guard convention (enforced in Phase 1 refactor).
# This hook prevents regressions when new headers are added.

EXIT_CODE=0

for file in "$@"; do
    # Skip if file doesn't exist (might be deleted in this commit)
    [ -f "$file" ] || continue

    if ! grep -q "^#pragma once" "$file"; then
        echo "ERROR: $file is missing '#pragma once'"
        echo "       libhmm uses #pragma once as the header guard convention."
        echo "       Add '#pragma once' as the first line of the file."
        EXIT_CODE=1
    fi
done

exit $EXIT_CODE
