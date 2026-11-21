"""Record GPU usage via nvidia-smi, system RAM via psutil, and llama-server cache metrics via /metrics."""
import csv
import datetime as dt
import os
import subprocess
import time
import psutil
import requests

interval = float(os.getenv('MONITOR_INTERVAL', '1'))
outfile = os.getenv('MONITOR_OUTPUT', 'gpu_system_metrics.csv')
metrics_url = os.getenv('SERVER_URL', 'http://127.0.0.1:8080').rstrip('/') + '/metrics'

# llama-server /metrics fields to record (prefix "llamacpp:" is stripped).
# See kv_cache_metrics.md for what each one means.
server_metrics = [
    # prefix cache hits (counters)
    'prompt_tokens_total', 'prompt_tokens_cached_total',
    # GPU KV cache occupancy (gauges)
    'kv_cache_tokens', 'kv_cache_tokens_max', 'kv_cache_usage_ratio',
    # CPU RAM prompt cache (gauges, then counters)
    'prompt_cache_entries', 'prompt_cache_tokens', 'prompt_cache_bytes', 'prompt_cache_bytes_max',
    'prompt_cache_saves_total', 'prompt_cache_loads_total', 'prompt_cache_loaded_tokens_total', 'prompt_cache_evictions_total',
    # queueing (gauges)
    'requests_processing', 'requests_deferred',
]

def find_llama_server():
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if proc.info['name'] == 'llama-server':
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    raise RuntimeError("llama-server is not running")

def read_server_metrics():
    """Return {name: value} from /metrics; empty values if the server doesn't answer."""
    values = dict.fromkeys(server_metrics, '')
    try:
        text = requests.get(metrics_url, timeout=2).text
    except requests.RequestException:
        return values
    for line in text.splitlines():
        if line.startswith('llamacpp:'):
            name, _, value = line[len('llamacpp:'):].partition(' ')
            if name in values:
                v = float(value)
                values[name] = int(v) if v.is_integer() else v
    return values

def read_server_gpu_memory_mib(pid):
    """GPU memory used by the llama-server process; empty if nvidia-smi can't attribute it."""
    out = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
    for line in out.strip().splitlines():
        app_pid, used = [x.strip() for x in line.split(',')]
        if app_pid == str(pid):
            return float(used)
    return ''

server = find_llama_server()
print(f"Monitoring llama-server (PID: {server.pid}, metrics: {metrics_url})")

if read_server_metrics()['kv_cache_tokens'] == '':
    print("Warning: no kv_cache metrics at", metrics_url, "- server not reachable or not built with llama-kv-metrics.patch")

fields = ['utc_time','gpu_memory_used_mib','gpu_memory_total_mib','gpu_utilization_pct','server_gpu_memory_mib','system_available_bytes','system_used_bytes','server_rss_bytes'] + server_metrics
with open(outfile,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields)
    w.writeheader()
    print('Writing',outfile,'(Ctrl-C to stop)')
    try:
        while True:
            result=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0]
            used,total,util=[float(x.strip()) for x in result.split(',')]
            ram=psutil.virtual_memory()
            server_rss = server.memory_info().rss
            row = dict(utc_time=dt.datetime.now(dt.timezone.utc).isoformat(),gpu_memory_used_mib=used,gpu_memory_total_mib=total,gpu_utilization_pct=util,server_gpu_memory_mib=read_server_gpu_memory_mib(server.pid),system_available_bytes=ram.available,system_used_bytes=ram.used,server_rss_bytes=server_rss)
            row.update(read_server_metrics())
            w.writerow(row)
            f.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
