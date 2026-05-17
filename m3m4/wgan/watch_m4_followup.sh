#!/usr/bin/env zsh
set -euo pipefail

cd /Users/andrewzhang/prelim/m3m4/wgan
mkdir -p logs

log="logs/m4_followup_$(date +%Y%m%d_%H%M%S).log"

{
  echo "[$(date)] Watcher started for PIDs 9804 and 9830"
  echo "[$(date)] Waiting for current GPU training processes to finish"

  while ps -p 9804 >/dev/null 2>&1 || ps -p 9830 >/dev/null 2>&1; do
    sleep 60
  done

  echo "[$(date)] Both watched processes finished; starting M4 run"
  python train.py --model M4 --csv m4.csv

  echo "[$(date)] Finished M4 run; starting M4 vanilla run"
  python train.py --model M4 --recon-weight 0 --csv m4vanilla.csv

  echo "[$(date)] All queued M4 runs finished"
} >> "$log" 2>&1
