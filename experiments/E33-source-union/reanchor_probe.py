"""E33 步骤 1-2（Main 的新委托）：给 E34 的四个臂定标尺，并量出 harness→build 偏移。

## 步骤 1 的答案（读代码即可，不需要跑）

`decouple_probe.py:119` 是决定性的一行：

    got = m.pds(m._pred_bulk_closed(m_full, lf, perts, bts), real_bulk, genes)

`_pred_bulk_closed`（`E28-pds/run.py:143-155`）直接做

    tgt = m_full * 2**lf ; tgt *= 1e6/tgt.sum() ; row = log1p(bts*tgt/1e6)

其 docstring 自己写着「**无需模拟细胞**」。所以：**E34 的每个臂都是闭式值，
既没有 hamilton 取整，也没有 400 细胞抽样、20k 深度稀释或逐细胞 CPM 重归一。**
E34 自己的 docstring 第 28-31 行也已声明「绝对值不可与 agg_*.parquet 比」。

⇒ 结论一句话：**四个臂彼此内部可比，其绝对水平不迁移到提交。**

## 臂 A 与 V8 的差异不止「闭式 vs 构建」，还有三处设计差异（本脚本逐项拆开）

| | E34 臂 A | build_v8.py |
|---|---|---|
| 基因宇宙 | `genes`(18,080) ∩ symbol = **7,481**（`run.py:124`） | `gate_sym`(10,779) ∩ symbol = **7,016**（`build_v8.py:171-172`）|
| 排序前的可用性掩码 | 无，`Bc = nan→0`（`run.py:167`） | `good = isfinite(b) & isfinite(s) & (s>0)`（`build_v8.py:183`）|
| lfc 幅度 | `scale=1.0` | **`LAMBDA=0.7`**（`build_v8.py:196`）|

（顺带：F28 的「7,481」正是 `_load_beta` 这个 7,481，与被打分的 gate 无关 —— 这解释了
F28 那个错误分母的来源。）

## 本脚本给出的三个数

    A       = E34 臂 A，闭式            —— 应复现 0.7321
    A'      = V8 的真实设计，闭式        —— 同一闭式路径，但宇宙/掩码/LAMBDA 全照 build_v8
    V8_meas = 0.7500                   —— agg_v8.parquet，cell_eval2 对真实提交的实测

于是  (A' − A) = 设计差异偏移，  (V8_meas − A') = **闭式→构建偏移**。
把后者加到任何臂的闭式值上，才是该臂在提交刻度上的预期值。

跑法：~/vcc2026/.venv/bin/python experiments/E33-source-union/reanchor_probe.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

_spec = importlib.util.spec_from_file_location(
    "e28run", ROOT / "experiments" / "E28-pds" / "run.py")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

OUT = Path(__file__).resolve().parent / "out"
K_CALL, LAMBDA, TOPK_OFF = 288, 0.7, 1000


def main() -> None:
    perts, genes, real, gi_full, B, S = m._load_beta()
    m_full = m._ctrl_profile(real)
    bts = float(m.vcc_cfg().bulk_target_sum)
    real_bulk = m.pseudobulk_bulk_lognorm(real, m.PERT_COL, bulk_target_sum=bts)
    G, n_p = genes.size, len(perts)
    print(f"扰动 {n_p}  基因 {G:,}  E34 宇宙(共同基因) {gi_full.size:,}  bts {bts:g}")

    # pds 的粒度：8 个扰动、rank_denominator="n-1" → 每扰动只能取 k/7，
    # 均值粒度 = 1/(8*7) = 1/56 = 0.017857。所有已见值都是它的整数倍。
    gran = 1.0 / (n_p * (n_p - 1))
    print(f"pds 粒度 1/{n_p*(n_p-1)} = {gran:.6f}（一个扰动挪一个秩位 = {gran:.4f}）")

    Bc = np.where(np.isfinite(B), B, 0.0)
    Bd = Bc - Bc.mean(1, keepdims=True)

    def topk_mask(v, k):
        if k is None or k >= v.size:
            return np.ones(v.size, bool)
        keep = np.zeros(v.size, bool)
        keep[np.argsort(np.abs(v))[::-1][:k]] = True
        return keep

    # ---- E34 的臂（7,481 宇宙、无掩码、scale=1.0）----
    def lf_e34(vals, topk=None, scale=1.0):
        lf = np.zeros((n_p, G))
        for j in range(n_p):
            v = vals[:, j].copy()
            v[~topk_mask(v, topk)] = 0.0
            lf[j, gi_full] = scale * v
        return lf

    def lf_e34_split(topk_off=None, scale_off=1.0):
        lf = np.zeros((n_p, G))
        for j in range(n_p):
            call = topk_mask(Bc[:, j], K_CALL)
            v = np.where(call, Bc[:, j], 0.0)
            off = Bd[:, j].copy()
            off[call] = 0.0
            off[~topk_mask(off, topk_off)] = 0.0
            lf[j, gi_full] = v + scale_off * off
        return lf

    # ---- V8 的真实设计，同一闭式路径 ----
    # 宇宙：gate_sym ∩ symbol。gate_sym 从 out/gate_sym.npy 取（sig_gate_probe 已缓存，
    # 与 build_v8 的 genes[ref.gidx] 逐位相同，reach_probe 已 assert 过）。
    gate_sym = np.load(OUT / "gate_sym.npy", allow_pickle=True)
    gpos_all = pd.Index(genes)
    e34_syms = genes[gi_full]                       # E34 宇宙的 symbol
    v8_syms = pd.Index(gate_sym).intersection(pd.Index(e34_syms))
    # E34 宇宙内 → V8 宇宙的行选择
    sel_v8 = pd.Index(e34_syms).get_indexer(v8_syms)
    gi_v8 = gpos_all.get_indexer(v8_syms)           # 全基因下标
    print(f"V8 宇宙(gate ∩ symbol) {len(v8_syms):,}  "
          f"（E34 多出 {gi_full.size - len(v8_syms):,} 个 gate 外基因）")

    Bv, Sv = B[sel_v8], S[sel_v8]
    good_v = np.isfinite(Bv) & np.isfinite(Sv) & (Sv > 0)

    def lf_v8(scale=LAMBDA):
        """逐字照 build_v8.py:182-196 的召集与赋值，只是走闭式 pds 而非 decoder。"""
        lf = np.zeros((n_p, G))
        for j in range(n_p):
            b = Bv[:, j]
            score = np.where(good_v[:, j], np.abs(b), -np.inf)
            sel = np.argsort(score)[::-1][:min(K_CALL, int(good_v[:, j].sum()))]
            lf[j, gi_v8[sel]] = scale * b[sel]
        return lf

    def lf_v8_split(topk_off=TOPK_OFF, scale_off=1.0, scale_call=LAMBDA):
        """臂 I 的 V8 刻度版：召集集逐位等于 V8，其余用去面板均值填 lfc_all 通道。"""
        Bdv = Bv - Bv.mean(1, keepdims=True)
        Bdv = np.where(np.isfinite(Bdv), Bdv, 0.0)
        lf = np.zeros((n_p, G))
        for j in range(n_p):
            b = Bv[:, j]
            score = np.where(good_v[:, j], np.abs(b), -np.inf)
            sel = np.argsort(score)[::-1][:min(K_CALL, int(good_v[:, j].sum()))]
            call = np.zeros(len(b), bool); call[sel] = True
            off = Bdv[:, j].copy()
            off[call] = 0.0
            off[~topk_mask(off, topk_off)] = 0.0
            lf[j, gi_v8] = scale_off * off
            lf[j, gi_v8[sel]] = scale_call * b[sel]     # 召集集覆盖，逐位等于 V8
        return lf

    arms = [
        ("A   E34 臂A（7481/无掩码/scale1.0）", lf_e34(Bc, topk=K_CALL)),
        ("C   E34 全基因去均值",                lf_e34(Bd)),
        ("G   E34 top1000 去均值",              lf_e34(Bd, topk=1000)),
        ("I   E34 召集288+其余top1000去均值",   lf_e34_split(topk_off=1000)),
        ("A'  V8 真实设计（7016/掩码/λ0.7）",   lf_v8()),
        ("A'' V8 设计但 scale=1.0",             lf_v8(scale=1.0)),
        ("I'  V8刻度的臂I（λ0.7召集+top1000）", lf_v8_split()),
        ("I'' V8刻度臂I，off scale 0.7",        lf_v8_split(scale_off=0.7)),
    ]
    print(f"\n{'臂':40s} {'pds(闭式)':>10s} {'Δ vs A':>9s} {'逐扰动':>8s}")
    print("-" * 92)
    vals = {}
    base = None
    for name, lf in arms:
        got = m.pds(m._pred_bulk_closed(m_full, lf, perts, bts), real_bulk, genes)
        mu = float(np.mean(list(got.values())))
        vals[name.split()[0]] = mu
        if base is None:
            base = mu
        print(f"{name:40s} {mu:10.4f} {mu-base:+9.4f}  "
              + " ".join(f"{got[p]:.2f}" for p in perts))
    print("-" * 92)

    v8_meas = {r["metric"]: r["mean"] for r in pl.read_parquet(
        ROOT / "experiments" / "E28-pds" / "out"
        / "agg_v8.parquet").iter_rows(named=True)}["pds_cosine"]
    off_design = vals["A'"] - vals["A"]
    off_build = v8_meas - vals["A'"]
    print(f"\nA  (E34 臂A, 闭式)          {vals['A']:.4f}   —— 应为 0.7321")
    print(f"A' (V8 真实设计, 闭式)       {vals[chr(65)+chr(39)]:.4f}")
    print(f"V8 实测 (cell_eval2, 提交)   {v8_meas:.4f}")
    print(f"→ 设计差异偏移 (A'−A)        {off_design:+.4f}  "
          f"= {off_design/gran:+.1f} 个粒度单位")
    print(f"→ 闭式→构建偏移 (V8−A')      {off_build:+.4f}  "
          f"= {off_build/gran:+.1f} 个粒度单位")
    print(f"→ 复合偏移 (V8−A)            {v8_meas - vals['A']:+.4f}")
    print(f"\n臂 I 在提交刻度上的预期值：")
    print(f"  用 E34 的 I ({vals['I']:.4f}) + 复合偏移 = "
          f"{vals['I'] + (v8_meas - vals['A']):.4f}  ← E34 刻度 + 平移，最粗的估计")
    print(f"  用 V8 刻度的 I' ({vals[chr(73)+chr(39)]:.4f}) + 闭式→构建偏移 = "
          f"{vals[chr(73)+chr(39)] + off_build:.4f}  ← 只需平移闭式→构建这一项，更可信")
    print(f"  相对 V8 实测 {v8_meas:.4f} 的增量 = "
          f"{vals[chr(73)+chr(39)] + off_build - v8_meas:+.4f}")
    np.save(OUT / "reanchor_vals.npy", np.array(
        [[vals[k] for k in ("A", "C", "G", "I", "A'", "A''", "I'", "I''")]]))
    print(f"\n⚠️ 粒度警告：8 个扰动下 pds 只有 {n_p*(n_p-1)} 个可分辨档位，"
          f"一档 {gran:.4f}。E34 的 Δ(A→I)=+0.0714 只有 4 档，"
          f"落在 8 扰动的抽样噪声里，不能当作连续量读。")


if __name__ == "__main__":
    main()
