"""Part 1f：解决 reach 与符号一致率的表面矛盾（Main 的 a / b / c 三个假设）。

背景矛盾：我实测 K562GW 的 |beta| 前 288 与 real lfc 的符号一致率 50.0-55.6%（掷硬币），
但 V8 的 `de_wilcoxon_direction_reach_raw` = 0.1482，而该成员量的是「前缀符号纯度 ≥ 0.9」
能走多深。50% 的排序造不出 90% 纯的前缀 —— 除非两者量的不是同一件事。

逐字读过 `cell_eval2/metrics/direction.py`（`de_direction_reach` 862-1056、
`_purity_curve` 411-526、`_k_star` 529-563、`_reference_stats`）后的定义：

  * universe="adjudicated"（catalog 默认）→ 排序池 **只含 reference-significant 基因**
    （`p_adj_real < alpha`），且剔除靶基因自身。**不是**全基因，**不是**我们的前 288。
  * 排序键依次：`_sig_pred`(降) → `rank_p_adj`(升) → `rank_p_value`(升) →
    `abs_lfc_pred`(降) → `feature`(升)。**主键是预测侧的 p_adj，不是 |lfc|。**
  * k = 深度 = 可裁决对的累计数；purity(k) = n_match/n_denom；
    k* = purity ≥ REACH_PURITY_FLOOR(=0.9) 的**最深** k（纯度非单调，可回升）。
  * N_conf = reference-significant 计数（剔除自身），与预测无关。
  * reach = k*/N_conf。

⚠️ 该 docstring 自己写明（issue #279）：`k* >= 1` 只需**第一对**命中，即一次掷硬币，
且这一项主导无技巧提交的期望值；raw reach 的无技巧点是 `~c/N_conf`，**不是常数**。

本脚本在内存里精确复刻 V8 的预测（不落盘任何 .h5ad），算出逐扰动的 k* / N_conf，
并用能否复现官方实测的 0.1482 来验证复刻。

跑法：~/vcc2026/.venv/bin/python experiments/E33-source-union/reach_probe.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import polars as pl
from scipy import sparse as sp
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vcclab.decoder import design_cells  # noqa: E402
from vcclab.scorer import TS_CELL, bh_adjust  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "E10-h1-erp"))
from run import VCC_CTRL_CELLS, VCC_PERT_CELLS, VCC_UMI, read_rows, thin  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sig_gate_probe import ref_from_matrix  # noqa: E402  内存版 ControlRef.__init__

DATA = ROOT / "data"
H5 = DATA / "vcc2025" / "adata_Validation.h5ad"
MAPCSV = DATA / "external" / "ens2sym.csv"
OUT = Path(__file__).resolve().parent / "out"
ALPHA, Z_BH, SEED, LAMBDA, K = 0.05, 3.184, 0, 0.7, 288
PURITY_FLOOR = 0.9          # REACH_PURITY_FLOOR，catalog 不可覆写
DEPTHS = (1, 2, 3, 5, 10, 25, 50, 100, 288)


def as_str(g):
    if isinstance(g, h5py.Group) and "categories" in g:
        c = [x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]
        return np.asarray(c)[g["codes"][:]]
    if isinstance(g, h5py.Group) and "values" in g:
        g = g["values"]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g[:]])


def main() -> None:
    t0 = time.time()
    picks = [l.strip() for l in
             (ROOT / "experiments" / "E28-pds" / "out" / "perts.csv").read_text().split()
             if l.strip()]
    zr = np.load(OUT / "real_de.npz", allow_pickle=True)
    gate_sym_real = zr["gate_sym"]
    real = {str(p): (zr[f"p_{p}"], zr[f"l_{p}"]) for p in zr["perts"]}
    e2s_df = pd.read_csv(MAPCSV).dropna()
    e2s = dict(zip(e2s_df.ensembl, e2s_df.symbol))

    # ---- 源侧：逐字照 build_v8.py:142-178 ----
    gw = pd.read_csv(DATA / "nadig2025" / "K562GW_lfc.csv.gz", index_col=0,
                     usecols=["Unnamed: 0"] + picks,
                     dtype={c: np.float32 for c in picks}, engine="c")[picks]
    gw.index = gw.index.astype(str)
    se = pd.read_csv(DATA / "nadig2025" / "K562GW_se.csv.gz", index_col=0,
                     usecols=["Unnamed: 0"] + picks,
                     dtype={c: np.float32 for c in picks}, engine="c")[picks]
    se.index = se.index.astype(str)
    se = se.reindex(index=gw.index)
    sym = pd.Index(gw.index.map(e2s))
    keep = pd.notna(sym)
    gw, se, sym = gw[keep], se[keep], sym[keep]
    dd = ~sym.duplicated()
    gw, se, sym = gw[dd], se[dd], sym[dd]
    gw.index, se.index = sym, sym

    # ---- (c) 符号一致率 vs 前缀深度：只需源 lfc + real lfc，最便宜，先做 ----
    gpos_r = pd.Index(gate_sym_real)
    common_r = gpos_r.intersection(gw.index)
    print("=== (c) K562GW |beta| 排序的符号一致率 vs 前缀深度 ===")
    print("总体（全 CPM gate 内 K562GW 覆盖基因，真值 = real lfc）")
    hdr = "".join(f"{f'k={d}':>9s}" for d in DEPTHS)
    print(f"{'扰动':10s}{hdr}{'  全覆盖':>9s}")
    print("-" * (10 + 9 * len(DEPTHS) + 9))
    pooled = {d: [0, 0] for d in DEPTHS}
    for p in picks:
        b = gw.loc[common_r, p].to_numpy(dtype=np.float64)
        s = se.loc[common_r, p].to_numpy(dtype=np.float64)
        rl = real[p][1][gpos_r.get_indexer(common_r)]
        good = np.isfinite(b) & np.isfinite(s) & (s > 0)
        order = np.argsort(np.where(good, np.abs(b), -np.inf))[::-1]
        row = ""
        for d in DEPTHS:
            sel = order[:d]
            ok = (b[sel] != 0) & (rl[sel] != 0) & np.isfinite(rl[sel])
            n = int(ok.sum())
            m = int((np.sign(b[sel][ok]) == np.sign(rl[sel][ok])).sum())
            pooled[d][0] += m; pooled[d][1] += n
            row += f"{(m/n if n else float('nan')):9.1%}"
        allok = (b != 0) & (rl != 0) & np.isfinite(rl)
        full = float(np.mean(np.sign(b[allok]) == np.sign(rl[allok])))
        print(f"{p:10s}{row}{full:9.1%}")
    print("-" * (10 + 9 * len(DEPTHS) + 9))
    print(f"{'合并':10s}" + "".join(
        f"{(pooled[d][0]/pooled[d][1] if pooled[d][1] else float('nan')):9.1%}"
        for d in DEPTHS))
    print(f"（合并分母 n：" + " ".join(f"k={d}:{pooled[d][1]}" for d in DEPTHS) + "）")

    # 同一条曲线，但只在 real_sig（被计分的那群基因）上
    print("\n只在 real_sig（reach / nmae 的实际人口）内计数：")
    print(f"{'扰动':10s}{hdr}")
    print("-" * (10 + 9 * len(DEPTHS)))
    for p in picks:
        pa, lf = real[p]
        b = gw.loc[common_r, p].to_numpy(dtype=np.float64)
        s = se.loc[common_r, p].to_numpy(dtype=np.float64)
        gi = gpos_r.get_indexer(common_r)
        rl, ra = lf[gi], pa[gi]
        insig = (ra < ALPHA) & np.isfinite(rl) & (common_r.to_numpy() != p)
        good = np.isfinite(b) & np.isfinite(s) & (s > 0)
        order = np.argsort(np.where(good, np.abs(b), -np.inf))[::-1]
        row = ""
        for d in DEPTHS:
            sel = order[:d]
            ok = insig[sel] & (b[sel] != 0) & (rl[sel] != 0)
            n = int(ok.sum())
            m = int((np.sign(b[sel][ok]) == np.sign(rl[sel][ok])).sum())
            row += f"{m:4d}/{n:<4d}" if n else f"{'—':>9s}"
        print(f"{p:10s}{row}")

    # ---- (a)/(b)：精确复刻 V8 的预测，量 k* 与 N_conf ----
    print("\n=== 复刻 V8 的预测（内存，不落盘）===")
    with h5py.File(H5, "r") as f:
        genes = as_str(f["var"]["_index"])
        tg_all = as_str(f["obs"]["target_gene"])
    rng = np.random.default_rng(SEED)
    ntc_rows = rng.choice(np.flatnonzero(tg_all == "non-targeting"),
                          VCC_CTRL_CELLS, replace=False)
    ctrl = thin(read_rows(ntc_rows), VCC_UMI, rng)
    # ControlRef.load 只吃路径，会逼出一次 ~236 MB 的临时 h5ad 写（build_v8.py:131-136
    # 就是这么做的）。本机磁盘/swap 紧张，故改用内存版构造器 —— 与 __init__ 逐字等价，
    # 只是不落盘。（首次运行本脚本时用过一次临时 h5ad 并即刻 unlink；已改掉。）
    ref = ref_from_matrix(sp.csr_matrix(ctrl), genes)
    gidx = np.asarray(ref.gidx)
    gate_sym = genes[gidx]
    assert np.array_equal(gate_sym, gate_sym_real), "pred 侧与 real 侧 gate 不一致"
    print(f"gate {ref.G:,}（与 real 侧一致 ✓）")

    gi_g = pd.Index(gate_sym).get_indexer(common_r)
    gi_s = gw.index.get_indexer(common_r)
    rows = []
    for p in picks:
        b = gw[p].to_numpy(dtype=np.float64)[gi_s]
        s = se[p].to_numpy(dtype=np.float64)[gi_s]
        good = np.isfinite(b) & np.isfinite(s) & (s > 0)
        sel = np.argsort(np.where(good, np.abs(b), -np.inf))[::-1][:min(K, int(good.sum()))]
        r_set, lfc_t = gi_g[sel], LAMBDA * b[sel]
        V = design_cells(ref, r_set, lfc_t, n_cells=VCC_PERT_CELLS, seed=SEED)
        lib = V.sum(1, keepdims=True)
        pa_pred, lf_pred = ref.de_table(V * (TS_CELL / lib), tie_correct=True)
        pa_real, lf_real = real[p]

        # _purity_curve：universe="adjudicated" → 只留 p_adj_real < alpha；剔除自身
        pool = (pa_real < ALPHA) & (gate_sym != p)
        idx = np.flatnonzero(pool)
        n_conf = int(len(idx))                       # _reference_stats：只要显著
        in_denom = np.isfinite(lf_real[idx]) & (lf_real[idx] != 0)
        match = in_denom & (np.sign(lf_pred[idx]) == np.sign(lf_real[idx]))
        # 排序键：_sig_pred 降、rank_p_adj 升、abs_lfc_pred 降、feature 升
        # （rank_p_value 我方 de_table 不产出，polars 中为 null → 该键惰性）
        order = np.lexsort((gate_sym[idx], -np.abs(lf_pred[idx]), pa_pred[idx],
                            -( pa_pred[idx] < ALPHA).astype(int)))
        nd = np.cumsum(in_denom[order].astype(int))
        nm = np.cumsum(match[order].astype(int))
        purity = np.where(nd > 0, nm / np.maximum(nd, 1), np.nan)
        hit = (nd > 0) & (purity >= PURITY_FLOOR)
        k_star = int(nd[hit].max()) if hit.any() else 0
        reach = k_star / n_conf if n_conf else float("nan")
        n_rec_in_pool = int(np.isin(idx, r_set).sum())
        # 前 k* 个可裁决对里，有几个来自我们召集的 288
        head = order[:int(np.flatnonzero(hit)[-1] + 1)] if hit.any() else np.array([], int)
        rec_in_head = int(np.isin(idx[head], r_set).sum())
        rows.append((p, n_conf, k_star, reach, n_rec_in_pool, rec_in_head,
                     float(purity[nd > 0][0]) if (nd > 0).any() else float("nan")))

    print(f"\n=== (a)/(b) V8 的 reach 分解 ===")
    print(f"{'扰动':10s} {'N_conf':>7s} {'k*':>5s} {'reach':>8s} {'召集∩池':>8s} "
          f"{'k*前缀里的召集基因':>18s} {'purity(1)':>10s}")
    print("-" * 74)
    for p, n_conf, k_star, reach, nrp, rih, p1 in rows:
        print(f"{p:10s} {n_conf:7,d} {k_star:5d} {reach:8.4f} {nrp:8d} {rih:18d} {p1:10.3f}")
    print("-" * 74)
    mean_reach = float(np.nanmean([r[3] for r in rows]))
    meas = {r["metric"]: r["mean"] for r in pl.read_parquet(
        ROOT / "experiments" / "E28-pds" / "out" / "agg_v8.parquet").iter_rows(named=True)}
    print(f"复刻均值 {mean_reach:.4f}  vs  官方实测 "
          f"{meas['de_wilcoxon_direction_reach_raw']:.4f}  差 "
          f"{mean_reach - meas['de_wilcoxon_direction_reach_raw']:+.4f}")
    print(f"若 k*≡1（#279 说的一次掷硬币）则 reach = "
          f"{float(np.mean([1 / r[1] for r in rows if r[1]])):.4f}")
    print(f"若 k*≡0 则 0.0000；k* 中位 {np.median([r[2] for r in rows]):.1f}")
    contrib = sorted(((r[3] / len(rows), r[0]) for r in rows), reverse=True)
    print("对 0.1482 的贡献（reach/8，降序）：" +
          "  ".join(f"{n}={c:.4f}" for c, n in contrib))
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
