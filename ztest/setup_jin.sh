#!/usr/bin/env bash
set -euo pipefail

# 사용 예:
#   cd ~/pytorch/zext
#   ./setup_jin_payload_ext.sh

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[setup] python = $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
echo "[setup] torch  = $($PYTHON_BIN -c 'import torch; print(torch.__version__); print(torch.__file__)')"

# editable install (builds .so and registers for import)
$PYTHON_BIN -m pip install -e .

echo "[setup] OK. Test import..."
$PYTHON_BIN -c "import jin_payload_ext; print('jin_payload_ext import OK:', jin_payload_ext)"