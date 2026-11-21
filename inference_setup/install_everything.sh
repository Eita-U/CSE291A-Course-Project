#!/usr/bin/env bash
set -euo pipefail

# Resolve paths relative to the project root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LLAMA_DIR="$PROJECT_ROOT/llama.cpp"

CUDA_BIN="${CUDA_BIN:-/usr/local/cuda/bin}"
VENV_DIR="$SCRIPT_DIR/.venv"
MODEL_FILE="models/qwen3-1.7b/Qwen3-1.7B-Q8_0.gguf"

# Install system dependencies.
sudo apt-get update
sudo apt-get install -y \
    git \
    cmake \
    build-essential \
    libcurl4-openssl-dev \
    pkg-config \
    pciutils \
    python3-venv \
    software-properties-common \
    ubuntu-drivers-common

# Install the NVIDIA driver if missing. It only loads after a reboot.
if ! command -v nvidia-smi >/dev/null; then
    lspci | grep -qi nvidia || {
        echo "Error: no NVIDIA GPU found on this instance." >&2
        exit 1
    }

    sudo ubuntu-drivers install --gpgpu
    echo "NVIDIA driver installed. Reboot, then rerun this script." >&2
    exit 1
fi

nvidia-smi >/dev/null || {
    echo "Error: nvidia-smi failed. If the driver was just installed, reboot and rerun." >&2
    exit 1
}

# Install the CUDA toolkit if missing. GPU AMIs often have it off PATH.
if ! command -v nvcc >/dev/null; then
    if [ -x "$CUDA_BIN/nvcc" ]; then
        export PATH="$CUDA_BIN:$PATH"
        grep -qs "$CUDA_BIN" "$HOME/.bashrc" || echo "export PATH=$CUDA_BIN:\$PATH" >> "$HOME/.bashrc"
    else
        sudo add-apt-repository -y multiverse
        sudo apt-get update
        sudo apt-get install -y nvidia-cuda-toolkit
    fi
fi

command -v nvcc >/dev/null || {
    echo "Error: CUDA toolkit (nvcc) missing." >&2
    exit 1
}

nvidia-smi
nvcc --version

# Clone llama.cpp only if it doesn't exist.
if [ ! -d "$LLAMA_DIR/.git" ]; then
    if [ -e "$LLAMA_DIR" ]; then
        echo "Error: $LLAMA_DIR exists but is not a Git repository." >&2
        exit 1
    fi

    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi

cd "$LLAMA_DIR"

# Undo any earlier patch so it can't block the checkout or be applied twice.
git checkout -- tools/server/server-context.cpp tools/server/server-task.cpp tools/server/server-task.h

# Pin a tested commit for reproducibility (override with LLAMA_CPP_COMMIT).
LLAMA_CPP_COMMIT="${LLAMA_CPP_COMMIT:-d854d98bae598388ecea1ed84396587e5edc326d}"

if [ "$(git rev-parse HEAD)" != "$(git rev-parse --verify --quiet "$LLAMA_CPP_COMMIT^{commit}" || true)" ]; then
    git fetch origin
    git checkout "$LLAMA_CPP_COMMIT"
fi

# Record the commit used for experiments.
git rev-parse HEAD | tee "$PROJECT_ROOT/llama_cpp_commit.txt"

# Add KV cache and prompt cache metrics to /metrics.
git apply "$SCRIPT_DIR/llama-kv-metrics.patch"
echo "Applied KV metrics patch."

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

cd "$PROJECT_ROOT"

# Set up the Python environment.
[ -x "$VENV_DIR/bin/python" ] || python3 -m venv "$VENV_DIR"

# A venv created with uv has no pip.
"$VENV_DIR/bin/python" -m pip --version >/dev/null 2>&1 || "$VENV_DIR/bin/python" -m ensurepip --upgrade

for requirements in inference_setup/requirements.txt requirements-backend.txt requirements-bench.txt; do
    if [ -f "$requirements" ]; then
        "$VENV_DIR/bin/python" -m pip install -r "$requirements"
    fi
done

"$VENV_DIR/bin/python" -m pip install 'huggingface_hub>=0.25,<2'

# Download the model only if it doesn't exist, then verify it.
if [ ! -f "$MODEL_FILE" ]; then
    "$VENV_DIR/bin/python" -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Qwen/Qwen3-1.7B-GGUF', filename='$(basename "$MODEL_FILE")', local_dir='$(dirname "$MODEL_FILE")')"
fi

sha256sum -c model.sha256

echo
echo "Setup complete. Start the server with: bash inference_setup/start_server.sh"

