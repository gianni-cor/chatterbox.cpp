#!/usr/bin/env bash
# End-to-end text -> wav synthesis using the C++/ggml pipeline.
#
# Pipeline:
#   chatterbox   (T3)     : text           -> speech_tokens
#   chatterbox-tts        : speech_tokens  -> wav (S3Gen + HiFT)
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

# The tokenizer is embedded in chatterbox-t3-turbo.gguf as GGUF metadata,
# so no separate path is required. For legacy GGUFs without an embedded
# tokenizer, set CHATTERBOX_TOKENIZER_DIR (or use --tokenizer-dir directly).
TOKENIZER_ARGS=""
if [[ -n "${CHATTERBOX_TOKENIZER_DIR:-}" ]] && [[ -f "$CHATTERBOX_TOKENIZER_DIR/vocab.json" ]]; then
    TOKENIZER_ARGS="--tokenizer-dir $CHATTERBOX_TOKENIZER_DIR"
elif [[ -f "$ROOT/tokenizer/vocab.json" ]]; then
    TOKENIZER_ARGS="--tokenizer-dir $ROOT/tokenizer"
fi

if [[ ! -x "$T3_BIN" ]] || [[ ! -x "$TTS_BIN" ]]; then
    echo "error: binaries not built; run 'cmake --build build' first" >&2
    exit 1
fi
for f in "$T3_GGUF" "$S3G_GGUF"; do
    [[ -f "$f" ]] || { echo "error: missing $f" >&2; exit 1; }
done

TMP="$(mktemp)"
trap "rm -f $TMP" EXIT

echo ">>> [1/2] T3: text -> speech tokens"
"$T3_BIN" \
    --model "$T3_GGUF" \
    ${TOKENIZER_ARGS} \
    --text "$TEXT" \
    --output "$TMP" \
    ${EXTRA_ARGS} > /dev/null

N_TOK=$(tr ',' '\n' < "$TMP" | wc -l | tr -d ' ')
echo "    generated $N_TOK speech tokens"

echo ">>> [2/2] S3Gen + HiFT: speech tokens -> wav (built-in voice from $S3G_GGUF)"
"$TTS_BIN" \
    --s3gen-gguf "$S3G_GGUF" \
    --tokens-file "$TMP" \
    --out "$OUT" \
    ${EXTRA_ARGS}

echo "done: $OUT"
