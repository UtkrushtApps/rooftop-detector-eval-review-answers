#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "[run.sh] Installing dependencies..."
python3 -m pip install -q -e ".[test]" || { echo "[run.sh] dependency install failed"; exit 1; }

echo "[run.sh] Loading JSON fixtures..."
python3 - <<'PY'
import json, sys
from pathlib import Path
base = Path("data")
for split in ("validation", "test"):
    for name in ("annotations.json", "predictions.json"):
        p = base / split / name
        with open(p) as f:
            json.load(f)
        print(f"[run.sh] loaded {p}")
print("[run.sh] fixtures OK")
PY
if [ $? -ne 0 ]; then
    echo "[run.sh] fixture load failed"
    exit 1
fi

echo "[run.sh] Running test suite (deployability probe)..."
set +e
python3 -m pytest -q
rc=$?
set -e 2>/dev/null || true

echo "[run.sh] pytest exit code: ${rc}"
if [ "${rc}" -eq 0 ]; then
    echo "[run.sh] deployability: OK (all tests passed)"
    exit 0
elif [ "${rc}" -eq 1 ]; then
    echo "[run.sh] deployability: OK (some designed checks fail on unsolved starter)"
    exit 0
else
    echo "[run.sh] deployability: BROKEN (pytest rc=${rc})"
    exit 1
fi
