#!/bin/sh
# Pause or resume the author's Zn mesh48g GPU stream on s03 (it runs the ph.x binary compiled under the Mg folder) for the ice-flow ensemble (author's answer 2026-10-07:
# 'pause Zn and start ice flow'). Acts only on processes whose executable is the GPU build
# /scratch/befahl/Mg/dfpt_qe/q-e-qe-7.5/bin/ph.x (the Zn CPU campaign, the drivers, nrun and the MPS server are left
# alone). SIGSTOP keeps every rank in memory; SIGCONT resumes them where they stopped.
# usage (on s03):  sh pause_zn_gpu.sh stop | cont | status      log: ~/dev/AGL/basodino/igm/logs/pause_mg.log (log name kept)
set -eu
EXE=/scratch/befahl/Mg/dfpt_qe/q-e-qe-7.5/bin/ph.x
LOG=$HOME/dev/AGL/basodino/igm/logs/pause_mg.log
PIDS=$(pgrep -f "^$EXE" || true)
case "${1:-status}" in
  stop) [ -n "$PIDS" ] && kill -STOP $PIDS; echo "$(date -Is) STOP $(echo $PIDS | wc -w) ranks" >> "$LOG" ;;
  cont) [ -n "$PIDS" ] && kill -CONT $PIDS; echo "$(date -Is) CONT $(echo $PIDS | wc -w) ranks" >> "$LOG" ;;
esac
ps -o pid=,stat=,etime= -p "$(echo $PIDS | tr ' ' ',')" 2>/dev/null | awk '{s[$2]++} END {for (k in s) print k, s[k]}'
