
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

# Optional: pin a tested commit for reproducibility.
if [ -n "${LLAMA_CPP_COMMIT:-}" ]; then
    git fetch origin
    git checkout "$LLAMA_CPP_COMMIT"
fi

# Record the commit used for experiments.
git rev-parse HEAD | tee "$PROJECT_ROOT/llama_cpp_commit.txt"

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

