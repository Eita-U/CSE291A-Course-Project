# CSE 291A Option 1 — inference server preparation

Target: UCSD-authorized **g6.xlarge** (NVIDIA L4 24 GB), Ubuntu Linux GPU AMI, **Qwen3-1.7B Q8_0**. This project installs llama.cpp on an *existing* EC2 instance.

## Expected directory layout

The two shell scripts live in `inference_setup/` and automatically treat its **parent** as the project root:

```text
project_root/
├── inference_setup/
│   ├── install_everything.sh
│   ├── start_server.sh
│   ├── smoke_test.py
│   ├── monitor.py
│   ├── inference_api.md
│   └── requirements.txt
├── llama.cpp/                  # installed by install_everything.sh
├── models/qwen3-1.7b/         # download model here
└── logs/                      # created by start_server.sh
```

## Prerequisites

- Authorized EC2 instance and approved AWS region/AMI.
- A working NVIDIA driver and CUDA toolkit on the instance. `nvidia-smi` and `nvcc --version` must succeed for the current build script.
- Enough disk space for OS, llama.cpp build, the **~2.17 GB** GGUF model, and logs (`df -h`).
- Restrict SSH inbound to authorized IPs. Do **not** publicly expose the inference port: `start_server.sh` binds to `127.0.0.1`.

## Build

From the **project root**:

```bash
bash inference_setup/install_everything.sh
python3 -m venv inference_setup/.venv
source inference_setup/.venv/bin/activate
python -m pip install -r inference_setup/requirements.txt
python -m pip install 'huggingface_hub>=0.25,<2'
```

For reproducible benchmarks, choose a tested llama.cpp commit:

```bash
LLAMA_CPP_COMMIT=<SHA> bash inference_setup/install_everything.sh
```

The script records the checked-out SHA at `llama_cpp_commit.txt` in the project root. Check that the checked-out build supports the server flags used below.

## Download model

From the **project root**, after activating the virtual environment:

```bash
mkdir -p models/qwen3-1.7b
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='ggml-org/Qwen3-1.7B-GGUF', filename='Qwen3-1.7B-Q8_0.gguf', local_dir='models/qwen3-1.7b')"
sha256sum models/qwen3-1.7b/Qwen3-1.7B-Q8_0.gguf > model.sha256
```

To reproduce precisely, pin the Hugging Face `revision` and retain `model.sha256`.

## Start and test

From the **project root**, start the server:

```bash
bash inference_setup/start_server.sh
```

In another EC2 terminal:

```bash
source inference_setup/.venv/bin/activate
python inference_setup/smoke_test.py
python inference_setup/monitor.py  # Ctrl-C to stop
```

Default flags: `PORT=8080`, `CTX_SIZE=4096`, `PARALLEL=2`, `GPU_LAYERS=99`, `--metrics`, `--slots`, and `--jinja`. The Q8_0 model is loaded from `models/qwen3-1.7b/`. You can override `PORT`, `MODEL_PATH`, and the other environment variables; use `SERVER_URL` for smoke tests when the port differs. Check the server output to confirm CUDA offload and GPU/KV usage. These defaults are pilot settings, not benchmark optima.

## Teammate connection

From a teammate's **laptop**, establish an SSH tunnel (use their personal SSH key if needed):

```bash
ssh -i /path/to/private_key -N -L 8080:127.0.0.1:8080 ubuntu@<CURRENT_EC2_PUBLIC_IP>
```

Then use `http://127.0.0.1:8080/v1/chat/completions` locally. The EC2 public IP can change after stop/start unless a static address is configured. Avoid running direct requests during scheduler benchmarks, as they bypass the scheduler; keep the scheduler behind authentication or a private SSH tunnel.

See [`inference_api.md`](inference_api.md) for basic completion and tool calling examples. Tool calling requires compatible model/template behavior as well as `--jinja`; successful HTTP responses do not guarantee that the model selected a tool.

## Reproducibility and cost

Record instance type, AMI ID, region, driver and CUDA version, llama.cpp commit, model SHA, launch flags, workload seeds, and logs. `nvidia-smi` reports device memory, not occupied KV tokens; `psutil` reports host memory. Keep contexts, parallelism, batching and cache controls fixed across comparisons, and benchmark without unrelated teammate traffic. Stop idle EC2 compute when permitted, remembering that EBS can continue to accrue charges. Verify any course-specific daily spending rules before running experiments.

**Note:** The two install/start scripts assume they are stored exactly one directory below the intended project root. If placed directly in the project root, their paths will be wrong.
