#!/bin/bash
# Запуск обучения в фоне. Лог и pid переживают обрыв SSH.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-/home/user/miniconda3/envs/lpr/bin/python}"
LOG="runs/2phase_v1/train.log"
PIDFILE="runs/2phase_v1/train.pid"
mkdir -p runs/2phase_v1

if [[ -f "$PIDFILE" ]]; then
  old="$(cat "$PIDFILE")"
  if kill -0 "$old" 2>/dev/null; then
    echo "уже запущено, pid $old"
    echo "лог: $PWD/$LOG"
    exit 1
  fi
fi

nohup "$PYTHON" -u train_2phase_v1.py "$@" >> "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "pid $(cat "$PIDFILE")"
echo "лог: $PWD/$LOG"
echo "смотреть: tail -f $PWD/$LOG"
