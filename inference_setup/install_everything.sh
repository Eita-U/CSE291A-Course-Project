
#!/usr/bin/env bash
set -euo pipefail

# Resolve paths relative to the project root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LLAMA_DIR="$PROJECT_ROOT/llama.cpp"

# Check NVIDIA driver and CUDA toolkit.
command -v nvidia-smi >/dev/null || {
    echo "Error: NVIDIA driver missing." >&2
    exit 1
}

command -v nvcc >/dev/null || {
    echo "Error: CUDA toolkit (nvcc) missing." >&2
    exit 1
}

nvidia-smi
nvcc --version

# Install system dependencies.
sudo apt-get update
sudo apt-get install -y \
    git \
    cmake \
    build-essential \
    libcurl4-openssl-dev \
    pkg-config

# Clone llama.cpp only if it doesn't exist.
if [ ! -d "$LLAMA_DIR/.git" ]; then
    if [ -e "$LLAMA_DIR" ]; then
        echo "Error: $LLAMA_DIR exists but is not a Git repository." >&2
        exit 1
    fi

    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi

cd "$LLAMA_DIR"

# Pin a tested commit for reproducibility (override with LLAMA_CPP_COMMIT).
LLAMA_CPP_COMMIT="${LLAMA_CPP_COMMIT:-d854d98bae598388ecea1ed84396587e5edc326d}"

if [ "$(git rev-parse HEAD)" != "$(git rev-parse --verify --quiet "$LLAMA_CPP_COMMIT^{commit}" || true)" ]; then
    git fetch origin
    git checkout "$LLAMA_CPP_COMMIT"
fi

# Record the commit used for experiments.
git rev-parse HEAD | tee "$PROJECT_ROOT/llama_cpp_commit.txt"

# Add KV cache occupancy gauges to /metrics. Skipped if already applied.
PATCH="$SCRIPT_DIR/llama-kv-metrics.patch"

if git apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "KV metrics patch already applied."
else
    git apply "$PATCH"
    echo "Applied KV metrics patch."
fi

# Build llama-server with CUDA support.
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON

cmake --build build \
    --target llama-server \
    -j "${BUILD_JOBS:-2}"

# Verify the resulting executable.
BINARY="$LLAMA_DIR/build/bin/llama-server"

if [ ! -x "$BINARY" ]; then
    echo "Error: llama-server build failed." >&2
    exit 1
fi

echo
echo "llama-server built successfully!"
echo "Binary: $BINARY"
echo "Commit: $(git rev-parse HEAD)"

