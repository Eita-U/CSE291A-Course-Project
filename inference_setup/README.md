# CSE 291A Option 1 — inference server preparation

Target: UCSD-authorized **g6.xlarge** (NVIDIA L4 24 GB), Ubuntu Linux GPU AMI, **Qwen3-1.7B Q8_0**. `install_everything.sh` sets up a raw Ubuntu GPU instance or an existing one.

## Expected directory layout

The two shell scripts live in `inference_setup/` and automatically treat its **parent** as the project root:

```text
project_root/
├── inference_setup/
│   ├── install_everything.sh
│   ├── start_server.sh
│   ├── smoke_test.py
│   ├── monitor.py
│   ├── llama-kv-metrics.patch  # applied to llama.cpp by install_everything.sh
│   ├── kv_cache_metrics.md
│   ├── kv_cache_notes.md
│   ├── inference_api.md
│   ├── requirements.txt
│   └── .venv/                  # created by install_everything.sh
├── llama.cpp/                  # created by install_everything.sh
├── models/qwen3-1.7b/          # created by install_everything.sh
└── logs/                       # created by start_server.sh
```

## Prerequisites

- Authorized EC2 instance and approved AWS region/AMI.
- An Ubuntu instance with an NVIDIA GPU and `sudo` access. `install_everything.sh` installs the NVIDIA driver and CUDA toolkit if they are missing.
- Python 3.11+ for the scheduler and benchmarks (`python3 --version`).
- Enough disk space for the OS, the NVIDIA driver and CUDA toolkit (if not preinstalled), the llama.cpp build, the **~1.83 GB** GGUF model, and logs (`df -h`).
- Restrict SSH inbound to authorized IPs. Do **not** publicly expose the inference port: `start_server.sh` binds to `127.0.0.1`.

## Set up the environment

1. Clone this repository on the instance and go to its root:

   ```bash
   git clone https://github.com/Eita-U/CSE291A-Course-Project.git
   cd CSE291A-Course-Project
   ```

2. Run the install script:

   ```bash
   bash inference_setup/install_everything.sh
   ```

3. If the script says `NVIDIA driver installed. Reboot, then rerun this script.`, reboot:

   ```bash
   sudo reboot
   ```

   Then SSH back in and repeat step 2 from the repository root. A new driver only loads after a reboot, so a raw instance needs two runs. If the instance already has a driver, one run is enough.

4. Setup is finished when the script prints `Setup complete.`

The script installs system packages, the NVIDIA driver and CUDA toolkit if missing, llama.cpp at a pinned commit with `llama-kv-metrics.patch`, the Python virtual environment at `inference_setup/.venv`, and the model at `models/qwen3-1.7b/`, verified against `model.sha256`.

The script is safe to rerun: existing installs and the model are kept. It does reset the three patched files in `llama.cpp/tools/server/`, so manual edits to them are lost. A running server keeps using the old binary until you restart it.

To build a different llama.cpp commit (the patch may not apply to it):

```bash
LLAMA_CPP_COMMIT=<SHA> bash inference_setup/install_everything.sh
```

The script records the checked-out SHA at `llama_cpp_commit.txt` in the project root. To pin the model precisely, set a Hugging Face `revision` in the script.

## Start and test

From the **project root**, start the server and leave it running:

```bash
bash inference_setup/start_server.sh
```

In another EC2 terminal, from the **project root**, activate the virtual environment and check the server:

```bash
source inference_setup/.venv/bin/activate
python inference_setup/smoke_test.py
```

The test passes when it prints `PASS: basic completion and structured tool call`.

To record GPU, memory and cache metrics during a run:

```bash
python inference_setup/monitor.py  # Ctrl-C to stop
```

This writes `gpu_system_metrics.csv` to the current directory, or to the path in `MONITOR_OUTPUT`. See [kv_cache_metrics.md](kv_cache_metrics.md) for the columns.

Default flags: `PORT=8080`, `CTX_SIZE=4096`, `PARALLEL=2`, `GPU_LAYERS=99`, `--metrics`, `--slots`, and `--jinja`. The Q8_0 model is loaded from `models/qwen3-1.7b/`. You can override `PORT`, `MODEL_PATH`, and the other environment variables; use `SERVER_URL` for `smoke_test.py` and `monitor.py` when the port differs. Check the server output to confirm CUDA offload and GPU/KV usage. These defaults are pilot settings, not benchmark optima.

`/metrics` also reports GPU KV cache occupancy and CPU RAM prompt cache usage (`kv_cache_*`, `prompt_cache_*`); see [kv_cache_metrics.md](kv_cache_metrics.md). For how slots and llama.cpp's prompt caching work, see [kv_cache_notes.md](kv_cache_notes.md).

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
