"""Part 1e（把离线算术校准到实测值，然后换算成官方尺度的货币）。

先校准：我的离线 nmae 模型对 **V8 本身** 算一遍，和 agg_v8.parquet 里官方打分器
实测的 1.000361 对比。若两者到 3-4 位小数吻合，则同一模型对 V11 / V12 的预测
继承这份可信度，不必花 ~22 分钟的 compute_metrics 去确认一个已被折断的机制。

两个必须照官方的 gate 规则（metrics/de.py:706-800，已逐字读过）：
  * p_adj < 0.05、real lfc 有限、**剔除被扰动基因自身的行**（issue #172）
  * n_gate < min_gate_size(=10，vcc2026 未覆写) 的扰动被**整个省略**，不计 1.0
    → MAT2A 的显著集只有 5 个，故 MAT2A 根本不进 de_wilcoxon_lfc_nmae 的均值。
    而 MAT2A 是 8 个扰动里**唯一**能被并集填充的那个。

跑法：~/vcc2026/.venv/bin/python experiments/E33-source-union/calib_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data"
MAPCSV = DATA / "external" / "ens2sym.csv"
OUT = Path(__file__).resolve().parent / "out"
ALPHA, LAMBDA, K, MIN_GATE = 0.05, 0.7, 288, 10
FILL_ORDER = ("HepG2Essential", "JurkatEssential", "RPE1Essential", "K562Essential")
# 官方两端尺度的 b / r（docs/01-scoring.md:167-174，由 OfficialBaseline 确立）
B_LO, B_HI, R_LO, R_HI = 1.0009, 1.0017, 0.369, 0.431


def load_source(name, picks, e2s, what="lfc"):
    cols = [str(c) for c in pd.read_csv(DATA / "nadig2025" / f"{name}_{what}.csv.gz",
                                        index_col=0, nrows=1).columns]
    have = [p for p in picks if p in cols]
    if not have:
        return pd.DataFrame()
    d = pd.read_csv(DATA / "nadig2025" / f"{name}_{what}.csv.gz", index_col=0,
                    usecols=["Unnamed: 0"] + have,
                    dtype={c: np.float32 for c in have}, engine="c")
    d.index = d.index.astype(str)
    d = d[have]
    sym = pd.Index(d.index.map(e2s))
    keep = pd.notna(sym)
    d, sym = d[keep], sym[keep]
    dd = ~sym.duplicated()
    d, sym = d[dd], sym[dd]
    d.index = sym
    return d


def scaled(u: float) -> tuple[float, float]:
    """误差型指标的两端尺度：score = (b - u) / (b - r)。返回 (不利端, 有利端)。"""
    a = (B_LO - u) / (B_LO - R_HI)
    b = (B_HI - u) / (B_HI - R_LO)
    return (min(a, b), max(a, b))


def main() -> None:
    picks = [l.strip() for l in
             (ROOT / "experiments" / "E28-pds" / "out" / "perts.csv").read_text().split()
             if l.strip()]
    z = np.load(OUT / "real_de.npz", allow_pickle=True)
    gate_sym = z["gate_sym"]
    de = {str(p): (z[f"p_{p}"], z[f"l_{p}"]) for p in z["perts"]}
    gpos = pd.Index(gate_sym)
    e2s_df = pd.read_csv(MAPCSV).dropna()
    e2s = dict(zip(e2s_df.ensembl, e2s_df.symbol))
    gw = load_source("K562GW", picks, e2s, "lfc")
    se = load_source("K562GW", picks, e2s, "se").reindex(index=gw.index)
    ess = {s: load_source(s, picks, e2s, "lfc") for s in FILL_ORDER}
    ess = {s: d for s, d in ess.items() if not d.empty}
    common = gpos.intersection(gw.index)
    gi_g, gi_s = gpos.get_indexer(common), gw.index.get_indexer(common)
    gw_syms = set(common.tolist())

    # ---- 官方 gate：逐扰动 real_sig，并套用 min_gate_size ----
    gates, dropped = {}, []
    for p in picks:
        pa, lf = de[p]
        m = (pa < ALPHA) & np.isfinite(lf) & (gate_sym != p)
        idx = np.flatnonzero(m)
        if len(idx) < MIN_GATE:
            dropped.append((p, len(idx)))
            continue
        gates[p] = (idx, lf[idx])
    print(f"min_gate_size={MIN_GATE} 省略 {len(dropped)} 个扰动: "
          f"{[(p, n) for p, n in dropped]}")
    print(f"进入 de_wilcoxon_lfc_nmae 均值的扰动 {len(gates)} 个: {list(gates)}\n")

    def pred_v8(p):
        """V8 实况：lfc_all 未传，只有 |beta| 前 288 个召集基因非 0。"""
        out = np.zeros(len(gate_sym))
        b = gw[p].to_numpy(dtype=np.float64)[gi_s]
        s = se[p].to_numpy(dtype=np.float64)[gi_s]
        good = np.isfinite(b) & np.isfinite(s) & (s > 0)
        sel = np.argsort(np.where(good, np.abs(b), -np.inf))[::-1][:min(K, int(good.sum()))]
        out[gi_g[sel]] = LAMBDA * b[sel]
        return out

    def pred_v11(p):
        """lfc_all = K562GW 全部覆盖基因（唯一旋钮：源统计量喂进 lfc_all 通道）。"""
        out = np.zeros(len(gate_sym))
        v = gw[p].to_numpy(dtype=np.float64)[gi_s]
        out[gi_g] = LAMBDA * np.where(np.isfinite(v), v, 0.0)
        return out

    def pred_v12(p):
        """lfc_all = K562GW 优先 + Essential 按自身 gate 覆盖率降序填空。"""
        out = pred_v11(p)
        for s_name, d in ess.items():
            if p not in d.columns:
                continue
            cand = pd.Index(sorted((set(d.index) & set(gate_sym.tolist())) - gw_syms))
            if len(cand) == 0:
                continue
            pos = gpos.get_indexer(cand)
            v2 = d.loc[cand, p].to_numpy(dtype=np.float64)
            free = (out[pos] == 0.0) & np.isfinite(v2)
            out[pos[free]] = LAMBDA * v2[free]
        return out

    def pred_floor(p, use_fill: bool):
        """天花板：覆盖到的基因给**真值**，未覆盖的给 0。"""
        idx, r = gates[p]
        out = np.zeros(len(gate_sym))
        syms = set(gw_syms)
        if use_fill:
            for s_name, d in ess.items():
                if p in d.columns:
                    syms |= set(d.index) & set(gate_sym.tolist())
        cov = np.isin(gate_sym[idx], list(syms))
        out[idx[cov]] = r[cov]
        return out

    arms = {
        "预测 0（member 的 no-skill 锚）": lambda p: np.zeros(len(gate_sym)),
        "V8 实况（288 召集基因，无 lfc_all）": pred_v8,
        "V11：lfc_all = K562GW only": pred_v11,
        "V12：lfc_all = K562GW + Essential 填充": pred_v12,
        "天花板 A：K562GW 覆盖处给真值": lambda p: pred_floor(p, False),
        "天花板 B：并集覆盖处给真值": lambda p: pred_floor(p, True),
    }
    print(f"{'方案':40s} {'raw nmae':>9s} {'官方 scaled（不利→有利）':>26s} "
          f"{'avg_score 贡献':>14s}")
    print("-" * 94)
    raws, per = {}, {}
    for name, fn in arms.items():
        acc = {}
        for p, (idx, r) in gates.items():
            pr = fn(p)[idx]
            acc[p] = float(np.abs(pr - r).mean() / np.abs(r).mean())
        u = float(np.mean(list(acc.values())))
        raws[name], per[name] = u, acc
        lo, hi = scaled(u)
        print(f"{name:40s} {u:9.4f} {lo:11.4f} → {hi:-10.4f} "
              f"{lo/6:6.4f}→{hi/6:7.4f}")
    print("-" * 94)
    v8_meas = {r["metric"]: r["mean"] for r in
               pl.read_parquet(ROOT / "experiments" / "E28-pds" / "out"
                               / "agg_v8.parquet").iter_rows(named=True)}
    off = raws["V8 实况（288 召集基因，无 lfc_all）"] - v8_meas["de_wilcoxon_lfc_nmae"]
    print(f"校准：离线 V8 模型 {raws['V8 实况（288 召集基因，无 lfc_all）']:.6f}  vs  "
          f"官方实测 {v8_meas['de_wilcoxon_lfc_nmae']:.6f}   差 {off:+.6f}")
    d12 = raws["V12：lfc_all = K562GW + Essential 填充"] - raws["V11：lfc_all = K562GW only"]
    print(f"并集净效应（V12 - V11） = {d12:+.6f} raw；"
          f"天花板净效应（B - A） = "
          f"{raws['天花板 B：并集覆盖处给真值'] - raws['天花板 A：K562GW 覆盖处给真值']:+.6f} raw")

    # ---- 逐扰动分解：集中还是分散（Main 的第 4 问）----
    print(f"\n逐扰动 nmae（分散 = 机制真实；集中在单个扰动 = 噪声或泄漏）")
    hdr = "".join(f"{p[:9]:>10s}" for p in gates)
    print(f"{'方案':40s}{hdr}")
    print("-" * (40 + 10 * len(gates)))
    for name in arms:
        print(f"{name:40s}" + "".join(f"{per[name][p]:10.4f}" for p in gates))


if __name__ == "__main__":
    main()
