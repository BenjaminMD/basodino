#!/bin/bash
# Read-only sampler of the s03 campaigns beside the IGM GPU work (author's answer 1, as Zn/mesh48g/campaign_rate.sh):
# every 30 s one line: epoch, time, load average (1 min), state counts of the ph.x ranks of the GPU build
# (the processes igm/pause_zn_gpu.sh stops: R running, T stopped), GPU utilisation of GPU 0 and 1 (%), number of
# igm_run processes, then for each campaign directory the newest ph_<i>.out and its last 'iter #' line (total cpu time).
# Campaign rate = deltas of that cpu time per wall second; the slowdown per stage is read from these lines.
#   setsid nohup bash rate_sampler.sh >> logs/rate_sampler.log 2>&1 < /dev/null &   (stop: kill its pid)
GPU_EXE=/scratch/befahl/Mg/dfpt_qe/q-e-qe-7.5/bin/ph.x
DIRS="/scratch/befahl/Zn/mesh48g /scratch/befahl/Zn/mesh48"
while true; do
  st=$(ps -o stat= -p "$(pgrep -d, -f "^$GPU_EXE")" 2>/dev/null | cut -c1 | sort | uniq -c | awk '{printf "%s%s ", $2, $1}')
  gu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr '\n' ' ')
  line="$(date +%s) $(date +%T) load $(cut -d' ' -f1 /proc/loadavg) gpu_phx ${st:-none} util ${gu}igm $(pgrep -c -f 'bin/igm_run')"
  for d in $DIRS; do f=$(ls -t $d/ph_*.out 2>/dev/null | head -1)
    line="$line | $(basename $d) $(basename "$f") $(grep -E 'iter #' "$f" | tail -1 | tr -s ' ')"; done
  echo "$line"; sleep 30
done
