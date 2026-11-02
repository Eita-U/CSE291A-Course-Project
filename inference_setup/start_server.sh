
#!/usr/bin/env bash
set -euo pipefail

# Resolve paths relative to the project root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BINARY="${LLAMA_SERVER_BIN:-$PROJECT_ROOT/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL_PATH:-$PROJECT_ROOT/models/qwen3-1.7b/Qwen3-1.7B-Q8_0.gguf}"

# Validate paths.
[ -x "$BINARY" ] || {
    echo "Missing executable: $BINARY" >&2
    exit 1
}

[ -f "$MODEL" ] || {
    echo "Missing model: $MODEL" >&2
    exit 1
}

mkdir -p "$PROJECT_ROOT/logs"

echo "Starting llama-server"
echo "Model: $MODEL"
echo "Context size: ${CTX_SIZE:-4096}"
echo "Parallel slots: ${PARALLEL:-2}"
echo "GPU layers: ${GPU_LAYERS:-99}"

"$BINARY" \
    -m "$MODEL" \
    --host 127.0.0.1 \
    --port "${PORT:-8080}" \
    --ctx-size "${CTX_SIZE:-4096}" \
    --parallel "${PARALLEL:-2}" \
    --n-gpu-layers "${GPU_LAYERS:-99}" \
    --metrics \
    --slots \
    --jinja \
    2>&1 | tee -a "$PROJECT_ROOT/logs/llama-server.log"

