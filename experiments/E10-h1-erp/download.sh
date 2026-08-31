#!/usr/bin/env bash
# VCC 2025 H1 validation split。6.93 GB。
# 64 MB 分块 + 逐块字节数校验 + 多轮续传 —— 单流和 8 路并行在不稳定代理下都会 partial。
set -u
cd "$(dirname "$0")/../../data/vcc2025" || exit 1
U="https://storage.googleapis.com/arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/validation/adata_Validation.h5ad"
SZ=6928967541; CH=$((64*1024*1024)); N=$(( (SZ + CH - 1) / CH ))
for pass in 1 2 3 4 5; do
  for i in $(seq 0 $((N-1))); do
    S=$(( i * CH )); E=$(( S + CH - 1 )); [ $E -ge $SZ ] && E=$(( SZ - 1 ))
    WANT=$(( E - S + 1 )); f="c_$(printf %04d $i)"
    [ -f "$f" ] && [ "$(stat -f%z "$f")" = "$WANT" ] && continue
    curl -sS --max-time 300 --retry 3 --retry-delay 2 -r "$S-$E" -o "$f" "$U" 2>/dev/null
    [ -f "$f" ] && [ "$(stat -f%z "$f")" = "$WANT" ] || rm -f "$f"
  done
  have=$(ls c_* 2>/dev/null | wc -l | tr -d ' ')
  echo "第 $pass 轮: $have / $N 块"
  [ "$have" = "$N" ] && break
  sleep 5
done
[ "$(ls c_* 2>/dev/null | wc -l | tr -d ' ')" = "$N" ] && cat c_* > adata_Validation.h5ad && rm -f c_* && echo "完成 $(stat -f%z adata_Validation.h5ad) / $SZ"
