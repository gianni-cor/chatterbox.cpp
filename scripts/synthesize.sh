#!/usr/bin/env bash
# End-to-end text -> wav synthesis using the C++/ggml pipeline.
#
# Pipeline:
#   chatterbox   (T3)     : text  -> speech_tokens
#   chatterbox-tts        : speech_tokens + ref_dir -> wav
#
# Usage:
#   scripts/synthesize.sh "Hello, world." out.wav
#   scripts/synthesize.sh "Hello, world." out.wav --seed 123

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 TEXT OUT.wav [--seed N]" >&2
    exit 1
fi

TEXT="$1"
OUT="$2"
shift 2
EXTRA_ARGS="$*"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
T3_BIN="$ROOT/build/chatterbox"
TTS_BIN="$ROOT/build/chatterbox-tts"
T3_GGUF="$ROOT/models/chatterbox-t3-turbo.gguf"
S3G_GGUF="$ROOT/models/chatterbox-s3gen.gguf"
TOKENIZER_DIR="${CHATTERBOX_TOKENIZER_DIR:-$HOME/.cache/huggingface/hub/models--ResembleAI--chatterbox-turbo/snapshots/749d1c1a46eb10492095d68fbcf55691ccf137cd}"
REF_DIR="$ROOT/artifacts/s3gen-ref"

if [[ ! -x "$T3_BIN" ]] || [[ ! -x "$TTS_BIN" ]]; then
    echo "error: binaries not built; run 'cmake --build build' first" >&2
    exit 1
fi
for f in "$T3_GGUF" "$S3G_GGUF"; do
    [[ -f "$f" ]] || { echo "error: missing $f" >&2; exit 1; }
done
[[ -d "$REF_DIR" ]] || { echo "error: missing $REF_DIR (run scripts/dump-s3gen-reference.py once)" >&2; exit 1; }

TMP="$(mktemp)"
trap "rm -f $TMP" EXIT

echo ">>> [1/2] T3: text -> speech tokens"
"$T3_BIN" \
    --model "$T3_GGUF" \
    --tokenizer-dir "$TOKENIZER_DIR" \
    --text "$TEXT" \
    --output "$TMP" \
    ${EXTRA_ARGS} > /dev/null

N_TOK=$(tr ',' '\n' < "$TMP" | wc -l | tr -d ' ')
echo "    generated $N_TOK speech tokens"

echo ">>> [2/2] S3Gen + HiFT: speech tokens -> wav"
"$TTS_BIN" \
    --s3gen-gguf "$S3G_GGUF" \
    --ref-dir "$REF_DIR" \
    --tokens-file "$TMP" \
    --out "$OUT" \
    ${EXTRA_ARGS}

echo "done: $OUT"
