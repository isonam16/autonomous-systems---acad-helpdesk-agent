#!/usr/bin/env bash
set -euo pipefail
export CHROMA_DIR="${CHROMA_DIR:-/var/data/chroma}"
export CHUNKS_JSON="${CHUNKS_JSON:-/var/data/chunks.json}"
mkdir -p "$CHROMA_DIR" "$(dirname "$CHUNKS_JSON")"
if [ ! -s "$CHUNKS_JSON" ] || [ ! -f "$CHROMA_DIR/chroma.sqlite3" ]; then
  python -m helpdesk.ingest
fi
exec uvicorn backend.main:app --host 0.0.0.0 --port "${PORT:-10000}"
