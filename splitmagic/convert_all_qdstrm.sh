#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${1:-./nsys_results}"

IMPORTER="${IMPORTER:-/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter}"

find "$ROOT" -type f -name "*.qdstrm" | while read -r qdstrm; do
    rep="${qdstrm%.qdstrm}.nsys-rep"

    # 이미 변환되어 있으면 skip
    if [[ -f "$rep" ]]; then
        echo "[SKIP] $(basename "$rep") already exists"
        continue
    fi

    echo "[IMPORT] $(basename "$qdstrm")"

    if "$IMPORTER" \
        --input-file "$qdstrm" \
        --output-file "$rep"; then
        echo "[ OK ] $(basename "$rep")"
    else
        echo "[FAIL] $(basename "$qdstrm")"
    fi

    echo
done

echo "Done."