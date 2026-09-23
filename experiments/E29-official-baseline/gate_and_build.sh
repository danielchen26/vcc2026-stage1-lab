#!/bin/bash
# 等到机器真的有余量，再启动 generic-response 基线构建。
#
# 门槛用 memory_pressure 的 free 百分比 + 数据卷余量，**不用 swap free**。
# 理由（measured，2026-09-22）：macOS 会把 swap 文件本身缩小来贴合用量，所以本机
# "swap free" 结构性地钉在接近 0 —— 实测 swap total 34,816 -> 30,720 -> 24,576 MB
# 而 used 始终维持在 total 的约 97%。用 "swap free > 2G" 当门槛永远满足不了。
# 真正能预测进程被回收的是**数据卷填满**（swap 就住在那个卷上）。
#
# 另外记一条 measured 的教训：即使门槛全绿、机器只剩这一个进程，本次构建仍然在
# wall 07:10 / CPU 00:02 的状态下 I/O 绑死 —— 全系统 pagein 20,103,026 页 x 16 KB
# 约等于 320 GB 换页，而卷 460 GB 只剩 8 GB。所以下面的 DISK 门槛设得比较松是不够的：
# 这份工作需要的是一台**磁盘不满**的机器，建议 DISK_MIN 调到 50 以上再跑。
cd "$(dirname "$0")" || exit 1
PY=/Users/chetianc/vcc2026/.venv/bin/python
FREE_PCT_MIN=15      # memory_pressure 的 "System-wide memory free percentage"
DISK_MIN=8           # GB；见上方注释，满盘时即使通过也会 I/O 绑死
TRIES=55
SLEEP=45

for i in $(seq 1 $TRIES); do
  FREE=$(memory_pressure | tail -1 | sed 's/[^0-9]*\([0-9]*\)%.*/\1/')
  DISK=$(df -g /System/Volumes/Data | awk 'NR==2{print $4}')
  HEAVY=$(ps -Ao rss,command | awk '/[p]ython/ && $1 > 400000' | wc -l | tr -d ' ')
  echo "[$(date +%H:%M:%S)] try $i/$TRIES  mem_free=${FREE}%  disk_free=${DISK}G  heavy_py=${HEAVY}"
  if [ "${FREE:-0}" -ge "$FREE_PCT_MIN" ] && [ "${DISK:-0}" -gt "$DISK_MIN" ] \
     && [ "${HEAVY:-9}" -lt 2 ]; then
    echo "[$(date +%H:%M:%S)] GATE OPEN -> exec baseline build (~25 min, zero .h5ad written)"
    exec $PY -u run_official_baseline.py baseline
  fi
  sleep $SLEEP
done
echo "GATE NEVER OPENED after $TRIES tries (~$((TRIES*SLEEP/60)) min); baseline NOT built."
exit 1
