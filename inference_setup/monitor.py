"""Record GPU usage via nvidia-smi and system RAM via psutil (requires nvidia-smi)."""
import csv
import datetime as dt
import os
import subprocess
import time
import psutil

interval = float(os.getenv('MONITOR_INTERVAL', '1'))
outfile = os.getenv('MONITOR_OUTPUT', 'gpu_system_metrics.csv')
fields = ['utc_time','gpu_memory_used_mib','gpu_memory_total_mib','gpu_utilization_pct','system_available_bytes','system_used_bytes']
with open(outfile,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    print('Writing',outfile,'(Ctrl-C to stop)')
    try:
        while True:
            result=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0]
            used,total,util=[float(x.strip()) for x in result.split(',')]
            ram=psutil.virtual_memory()
            w.writerow(dict(utc_time=dt.datetime.now(dt.timezone.utc).isoformat(),gpu_memory_used_mib=used,gpu_memory_total_mib=total,gpu_utilization_pct=util,system_available_bytes=ram.available,system_used_bytes=ram.used))
            f.flush(); time.sleep(interval)
    except KeyboardInterrupt:
        pass
