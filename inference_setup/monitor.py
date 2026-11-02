"""Record GPU usage via nvidia-smi and system RAM via psutil (requires nvidia-smi)."""
import csv
import datetime as dt
import os
import subprocess
import time
import psutil

interval = float(os.getenv('MONITOR_INTERVAL', '1'))
outfile = os.getenv('MONITOR_OUTPUT', 'gpu_system_metrics.csv')

def find_llama_server():
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if proc.info['name'] == 'llama-server':
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    raise RuntimeError("llama-server is not running")

server = find_llama_server()
print(f"Monitoring llama-server (PID: {server.pid})")

fields = ['utc_time','gpu_memory_used_mib','gpu_memory_total_mib','gpu_utilization_pct','system_available_bytes','system_used_bytes','server_rss_bytes']
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
            w.writerow(dict(utc_time=dt.datetime.now(dt.timezone.utc).isoformat(),gpu_memory_used_mib=used,gpu_memory_total_mib=total,gpu_utilization_pct=util,system_available_bytes=ram.available,system_used_bytes=ram.used,server_rss_bytes=server_rss))
            f.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
