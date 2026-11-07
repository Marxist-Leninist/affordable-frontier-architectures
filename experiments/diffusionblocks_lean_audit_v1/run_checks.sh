#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

LEAN="${LEAN:-lean}"
PYTHON="${PYTHON:-python3}"
version="$("$LEAN" --version)"
printf '%s\n' "$version"
if [[ "$version" != *"version 4.35.0,"* ]]; then
  printf 'Expected Lean 4.35.0. Set LEAN to the matching executable.\n' >&2
  exit 1
fi

"$LEAN" -D Elab.async=false -j1 DiffusionEquivalence.lean
"$PYTHON" numerical_checks.py
