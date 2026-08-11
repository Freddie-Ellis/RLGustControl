#!/usr/bin/env bash
set -euo pipefail

# Minimal init script: prefer .venv executables, run `uv sync` then build the Rust wheel

# Determine venv python and candidate uv
PY_CMD=""
UV_CMD=""
UV_CAND=""

if [ -x ".venv/bin/python" ]; then
	PY_CMD=".venv/bin/python"
	UV_CAND=".venv/bin/uv"
elif [ -x ".venv/Scripts/python.exe" ]; then
	PY_CMD=".venv/Scripts/python.exe"
	UV_CAND=".venv/Scripts/uv.exe"
fi

# Prefer venv uv if present, otherwise use system uv
if [ -n "${UV_CAND:-}" ] && [ -x "$UV_CAND" ]; then
	UV_CMD="$UV_CAND"
else
	UV_SYS="$(command -v uv || true)"
	if [ -n "$UV_SYS" ]; then
		UV_CMD="$UV_SYS"
	fi
fi

if [ -z "$UV_CMD" ]; then
	echo "Error: 'uv' command not found. Install 'uv' or install it into .venv and retry." >&2
	exit 1
fi

echo "Make sure you have the rust stuff .whl, ask Freddie for this if you don't have it."

echo "Running: $UV_CMD sync"
"$UV_CMD" sync

echo "Done."

