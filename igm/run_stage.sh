#!/bin/bash
# run_stage.sh STAGE LIST [NSLOT] [FIRST]: the IGM runs of LIST on s03, at most NSLOT (default 16) at once, on the
# slots FIRST (default 0) to FIRST + NSLOT - 1 (so a small stage can run beside one that holds the low slots).
#   LIST   text file, one run per line: NAME EXPERIMENT [hydra overrides ...]; '#' lines and blank lines skipped
#   slot s (0..15) runs one process at a time on hardware thread CPUS[s] and GPU s % 2, so 16 slots = 8 IGM
#          processes per H200 (author's answer 1); CPUS = 0-5, 64-69 (no ph.x mask covers them, 6 Oct) and 6-7,
#          70-71 (shared with the Zn CPU ph.x, accepted); re-check the masks before each stage (start.txt below)
#   every process: taskset to its thread, nice 10, TF_NUM_INTRAOP_THREADS = TF_NUM_INTEROP_THREADS =
#          OMP_NUM_THREADS = 1, CUDA_VISIBLE_DEVICES its GPU, hydra.run.dir runs/STAGE/NAME, log logs/STAGE/NAME.log
#   queue  slots take the next line when free (flock on logs/STAGE/next)
# Writes logs/STAGE/: start.txt (time, load, GPU use, the CPU masks of every ph.x), runs.csv (name, slot, cpu, gpu,
# start and end epoch, exit code), pids.txt, gpu_mem.txt (every 5 s: epoch, pid, MiB of our processes).
# TERM/INT: kills the IGM processes this script started, nothing else.
# TIMEOUT (environment, s; default 0 = none): a run still alive after TIMEOUT s gets TERM, KILL 30 s later (exit code
#          124 or 137 in runs.csv; step 6: a training run hung asleep for 25 min, 2026-10-08); rerun such runs in a fix stage
# usage (s03):  source ~/dev/AGL/basodino/igm/env.sh && cd $IGM_HOME && nrun bash run_stage.sh STAGE LIST [NSLOT] [FIRST]
set -u
cd "$(dirname "$0")" && source ./env.sh
STAGE=$1 LIST=$2 NSLOT=${3:-16} FIRST=${4:-0} TIMEOUT=${TIMEOUT:-0}
CPUS=(0 1 2 3 4 5 64 65 66 67 68 69 6 7 70 71)
LOG=logs/$STAGE
mkdir -p "$LOG" "runs/$STAGE"
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1
grep -v '^\s*#' "$LIST" | grep -v '^\s*$' > "$LOG/queue.txt"
echo 0 > "$LOG/next"
: > "$LOG/pids.txt"
[ -f "$LOG/runs.csv" ] || echo "name,slot,cpu,gpu,start,end,rc" > "$LOG/runs.csv"
{ date -Is; uptime; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
  for p in $(pgrep -f 'bin/ph.x'); do echo "$(readlink /proc/$p/exe) $(taskset -cp $p 2>/dev/null | sed 's/.*: //')"; done \
    | sort | uniq -c; } > "$LOG/start.txt"
cat "$LOG/start.txt"

next_line() {   # the next queue line, or nothing
  exec 9> "$LOG/next.lock"; flock 9
  local n; n=$(cat "$LOG/next"); echo $((n + 1)) > "$LOG/next"
  flock -u 9; exec 9>&-
  sed -n "$((n + 1))p" "$LOG/queue.txt"
}

slot() {
  local s=$1 cpu=${CPUS[$1]} gpu=$(($1 % 2)) line name exp t0 pid rc
  while line=$(next_line) && [ -n "$line" ]; do
    set -- $line; name=$1 exp=$2; shift 2
    t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=$gpu taskset -c "$cpu" nice -n 10 timeout -k 30 "$TIMEOUT" igm_run +experiment="$exp" "$@" \
      hydra.run.dir="runs/$STAGE/$name" > "$LOG/$name.log" 2>&1 &
    pid=$!; echo "$pid $name" >> "$LOG/pids.txt"
    wait $pid; rc=$?
    echo "$name,$s,$cpu,$gpu,$t0,$(date +%s),$rc" >> "$LOG/runs.csv"
  done
}

stop() {        # only live processes of this stage (a finished pid may have been reused)
  for p in $(awk '{print $1}' "$LOG/pids.txt"); do
    tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q "runs/$STAGE/" && kill "$p"; done
  kill "$MEM" 2>/dev/null; exit 143; }
trap stop TERM INT

( while true; do
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
      | while IFS=', ' read -r p m; do echo "$p $(ps -o ppid= -p "$p" | tr -d ' ') $m"; done \
      | awk -v t="$(date +%s)" 'NR == FNR {mine[$1]; next} ($1 in mine) {print t, $1, $3} ($2 in mine) {print t, $2, $3}' "$LOG/pids.txt" - \
      >> "$LOG/gpu_mem.txt"     # a run's pid in pids.txt is igm_run's or, with TIMEOUT, its parent timeout's
    sleep 5; done ) &
MEM=$!
for ((s = FIRST; s < FIRST + NSLOT; s++)); do slot $s & done
wait $(jobs -p | grep -v "^$MEM\$")
kill $MEM 2>/dev/null
echo "stage $STAGE done: $(grep -c . "$LOG/queue.txt") runs, exit codes $(tail -n +2 "$LOG/runs.csv" | cut -d, -f7 | sort | uniq -c | tr '\n' ' ')"
