#!/usr/bin/env bash
# Fetch Talkie's vocab.txt at the pinned commit, for local tokenizer tests.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p models
curl -sSfL -o models/vocab.txt \
  "https://huggingface.co/talkie-lm/talkie-1930-13b-base/resolve/b7c97680791f7fca4262c3c80b36ff7d666faab0/vocab.txt"
echo "fetched data/models/vocab.txt ($(wc -c < models/vocab.txt) bytes)"
