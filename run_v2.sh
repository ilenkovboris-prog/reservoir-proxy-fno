#!/bin/bash
# Обучение data_v2 в фоне. Лог и pid переживают обрыв SSH.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-/home/user/miniconda3/envs/lpr/bin/python}"
LOG="runs/v2/train.log"
PIDFILE="runs/v2/train.pid"
mkdir -p runs/v2

if [[ -f "$PIDFILE" ]]; then
  old="$(cat "$PIDFILE")"
  if kill -0 "$old" 2>/dev/null; then
    echo "уже запущено, pid $old"
    echo "лог: $PWD/$LOG"
    exit 1
  fi
fi

nohup "$PYTHON" -u train_v2.py --data-dir data_v2 "$@" >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "pid $(cat "$PIDFILE")"
echo "лог: $PWD/$LOG"
echo "смотреть: tail -f $PWD/$LOG"
