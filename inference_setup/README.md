# CSE 291A Option 1 — inference server preparation

Target: **UCSD-authorized g4dn.xlarge**, Ubuntu Linux GPU AMI (NVIDIA T4 16 GB), Qwen3-1.7B Q4_K_M. Nothing here provisions AWS or incurs AWS charges until you deploy it.

## Prerequisites
- Course account access and permission to launch g4dn.xlarge, approved AMI and region.
- Ubuntu image with functioning NVIDIA driver **and CUDA toolkit**; `nvidia-smi` and `nvcc --version` must succeed. Install a compatible CUDA toolkit from NVIDIA/AMI instructions if missing. `nvidia-smi` alone is not sufficient for compiling CUDA code.
- Disk storage sufficient for OS, build, model (~1.28 GB) and logs. Check `df -h`.
- Restrict SSH inbound to authorized IPs. Do **not** expose port 8080 publicly. Run inference server bound to `127.0.0.1`.

## Build
From this directory:
```bash
bash install.sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Pin a tested llama.cpp commit before final benchmarking, e.g. `LLAMA_CPP_COMMIT=<SHA> bash install.sh`. Save the SHA and compiled flags.

## Download model
```bash
mkdir -p models
python -m pip install 'huggingface_hub>=0.25,<2'
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='ggml-org/Qwen3-1.7B-GGUF', filename='Qwen3-1.7B-Q4_K_M.gguf', local_dir='models')"
sha256sum models/Qwen3-1.7B-Q4_K_M.gguf > model.sha256
```
To reproduce precisely, pin the Hugging Face revision as well (in `hf_hub_download(revision=...)`).

## Start and test
```bash
bash start_server.sh
# In another terminal:
source .venv/bin/activate
python smoke_test.py
python monitor.py
```
Start settings default to `CTX_SIZE=4096 PARALLEL=2 GPU_LAYERS=99`. These are **pilot** values, not guaranteed optimal. Capture server startup output to verify CUDA offload and KV allocation. Test `--help` if flags differ in the pinned version. Tool-calling support may depend on `--jinja` and model chat-template behavior.

## Teammate connection
For direct testing, each teammate opens an SSH tunnel **on their laptop**:
```bash
ssh -i /path/to/their_key.pem -N -L 8080:127.0.0.1:8080 ubuntu@<EC2_PUBLIC_IP>
```
They can then call `http://127.0.0.1:8080/v1`. Later expose only your **scheduler** through an authenticated endpoint or a separate SSH tunnel; bypassing it would invalidate scheduler experiments.

## Reproducibility and caution
- Record EC2 type, region, AMI ID, CUDA version, driver, llama.cpp commit, model hash, startup flags, workload seed, and logs.
- `nvidia-smi memory.used` shows device-level allocated/used VRAM, **not occupied KV tokens** or per-request allocation. `psutil` shows host RAM; use additional instrumentation for precise phase-specific allocations.
- Keep `--ctx-size`, `--parallel`, batching, cache controls fixed within comparisons; note that runtime may queue requests internally.
- Benchmarks should run with unrelated teammate traffic disabled.
- Stop EC2 when not in use; EBS persists and costs money. Check the course's $3 daily cap and instance eligibility.
