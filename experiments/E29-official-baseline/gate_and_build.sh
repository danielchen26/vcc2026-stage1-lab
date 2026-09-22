#!/bin/bash
# 等到 swap 空闲 > 2 GB 且磁盘 > 8 GB 再启动 generic-response 基线构建。
# 理由见 RESULT.md §5.3：本机 swap 住在数据卷上，卷满时内核直接回收 RSS 最大的进程，
# 日志 0 字节、无 traceback。本脚本只是把「等到安全窗口」自动化，不改任何评分配置。
cd "$(dirname "$0")"
PY=/Users/chetianc/vcc2026/.venv/bin/python
for i in $(seq 1 55); do
  SWAP=$(sysctl -n vm.swapusage | sed 's/.*free = \([0-9.]*\)M.*/\1/' | cut -d. -f1)
  DISK=$(df -g /System/Volumes/Data | awk 'NR==2{print $4}')
  HEAVY=$(ps -Ao rss,command | awk '/[p]ython/ && $1 > 400000' | wc -l | tr -d ' ')
  echo "[$(date +%H:%M:%S)] try $i  swap_free=${SWAP}M  disk_free=${DISK}G  heavy_procs=${HEAVY}"
  if [ "${SWAP:-0}" -gt 2000 ] && [ "${DISK:-0}" -gt 8 ] && [ "${HEAVY:-9}" -lt 2 ]; then
    echo "[$(date +%H:%M:%S)] GATE OPEN -> launching baseline build"
    exec $PY -u run_official_baseline.py baseline
  fi
  sleep 45
done
echo "GATE NEVER OPENED after 55 tries (~41 min); baseline NOT built."
