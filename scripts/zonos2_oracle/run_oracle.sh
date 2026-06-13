#!/usr/bin/env bash
# Dump the plain-torch ZONOS2 oracle parity fixtures.
# One-time, slow on Apple Silicon; that's expected.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

uv run --extra oracle python scripts/zonos2_oracle/dump_fixtures.py
