"""Part 1d（机制确认）：K562GW 的 lfc 在本验证 context 上到底传不传。

sig_gate_probe 给出两个结论，本脚本把第二个钉死：
  (a) 并集在 de_lfc_nmae 自己的 gate 内新增 0 个基因（MAT2A 的显著集只有 5 个，
      K562GW 已全覆盖）→ 并集对该指标的影响恒等于 0，不是「小」，是 0。
  (b) GW-only 的 nmae = 1.0517 > 1.0000，即 K562GW 的 lfc **比预测 0 更差**。
      若 (b) 成立，F28 的「覆盖率是该指标唯一出路」从根上错了：瓶颈不是覆盖，是信号。

本脚本量 (b) 的三个独立侧面，全部只用缓存，不跑 scorer：
  1. V8 真正召集的 288 个基因（|beta| 降序）与 real lfc 的符号一致率 —— 这是 V8
     实际喂进 decoder 的那批数，也是 direction_reach 读的那批。
  2. real_sig 上 λ*b 与 real lfc 的 Spearman / Pearson。
  3. expr_mse 的 lfc 空间代理（全 CPM gate）：填充是帮还是害。

跑法：~/vcc2026/.venv/bin/python experiments/E33-source-union/transfer_probe.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data"
MAPCSV = DATA / "external" / "ens2sym.csv"
OUT = Path(__file__).resolve().parent / "out"
ALPHA, LAMBDA, K = 0.05, 0.7, 288
SOURCES = ("K562GW", "K562Essential", "RPE1Essential",
           "JurkatEssential", "HepG2Essential")
FILL_ORDER = ("HepG2Essential", "JurkatEssential", "RPE1Essential", "K562Essential")


def load_source(name: str, picks, e2s, what="lfc"):
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


def main() -> None:
    t0 = time.time()
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
    gwse = load_source("K562GW", picks, e2s, "se")
    gwse = gwse.reindex(index=gw.index)
    common = gpos.intersection(gw.index)
    gi_g, gi_s = gpos.get_indexer(common), gw.index.get_indexer(common)

    # ---- 1. V8 实际召集的 288 个基因 ----
    print("=== 1. V8 召集集（K562GW |beta| 前 288）与 real lfc 的符号一致率 ===")
    print(f"{'扰动':10s} {'|R̂|':>5s} {'∩real_sig':>10s} {'符号一致(全288)':>15s} "
          f"{'符号一致(∩sig)':>15s} {'real|lfc|中位':>13s}")
    print("-" * 74)
    for p in picks:
        pa, lf = de[p]
        b = gw[p].to_numpy(dtype=np.float64)[gi_s]
        s = gwse[p].to_numpy(dtype=np.float64)[gi_s]
        good = np.isfinite(b) & np.isfinite(s) & (s > 0)
        score = np.where(good, np.abs(b), -np.inf)
        sel = np.argsort(score)[::-1][:min(K, int(good.sum()))]
        pos = gi_g[sel]                          # gate 内下标
        rl = lf[pos]
        pred = LAMBDA * b[sel]
        nz = (pred != 0) & (rl != 0)
        ag_all = float(np.mean(np.sign(pred[nz]) == np.sign(rl[nz])))
        insig = (pa[pos] < ALPHA) & np.isfinite(rl) & (gate_sym[pos] != p)
        ag_sig = (float(np.mean(np.sign(pred[insig]) == np.sign(rl[insig])))
                  if insig.any() else float("nan"))
        print(f"{p:10s} {len(sel):5d} {int(insig.sum()):10d} {ag_all:15.1%} "
              f"{ag_sig:15.1%} {np.median(np.abs(rl)):13.4f}")

    # ---- 2. real_sig 上的相关性 ----
    print("\n=== 2. real_sig 上 λ·b(K562GW) vs real lfc 的相关（0 = 不传） ===")
    print(f"{'扰动':10s} {'n':>6s} {'Spearman':>10s} {'p':>10s} {'Pearson':>10s} {'p':>10s}")
    print("-" * 60)
    for p in picks:
        pa, lf = de[p]
        m = (pa < ALPHA) & np.isfinite(lf) & (gate_sym != p)
        idx = np.flatnonzero(m)
        covered = np.isin(gate_sym[idx], common.to_numpy())
        idx = idx[covered]
        if len(idx) < 5:
            print(f"{p:10s} {len(idx):6d} {'n<5':>10s}")
            continue
        sub = gpos[idx]
        v = gw.loc[sub, p].to_numpy(dtype=np.float64)
        r = lf[idx]
        ok = np.isfinite(v) & np.isfinite(r)
        sr = spearmanr(v[ok], r[ok]); pr = pearsonr(v[ok], r[ok])
        print(f"{p:10s} {int(ok.sum()):6d} {sr.statistic:10.4f} {sr.pvalue:10.2e} "
              f"{pr.statistic:10.4f} {pr.pvalue:10.2e}")

    # ---- 3. expr_mse 的 lfc 空间代理：填充帮还是害 ----
    ess = {s: load_source(s, picks, e2s, "lfc") for s in FILL_ORDER}
    ess = {s: d for s, d in ess.items() if not d.empty}
    print("\n=== 3. 全 CPM gate 上的 SSE 代理（expr_mse 的 lfc 空间近似，标注为代理）===")
    print(f"{'扰动':10s} {'预测0':>10s} {'GW only':>10s} {'GW+填充':>10s} {'填充Δ':>9s}")
    print("-" * 54)
    tot = [0.0, 0.0, 0.0]
    for p in picks:
        _, lf = de[p]
        z0 = float((lf ** 2).sum())
        pa_ = np.zeros(len(gate_sym))
        v = gw[p].to_numpy(dtype=np.float64)[gi_s]
        pa_[gi_g] = LAMBDA * np.where(np.isfinite(v), v, 0.0)
        a = float(((pa_ - lf) ** 2).sum())
        pb_ = pa_.copy()
        for s_name, d in ess.items():
            if p not in d.columns:
                continue
            cand = pd.Index(sorted((set(d.index) & set(gate_sym.tolist()))
                                   - set(common.tolist())))
            if len(cand) == 0:
                continue
            p2 = gpos.get_indexer(cand)
            v2 = d.loc[cand, p].to_numpy(dtype=np.float64)
            free = (pb_[p2] == 0.0) & np.isfinite(v2)
            pb_[p2[free]] = LAMBDA * v2[free]
        b = float(((pb_ - lf) ** 2).sum())
        tot[0] += z0; tot[1] += a; tot[2] += b
        print(f"{p:10s} {z0:10.2f} {a:10.2f} {b:10.2f} {b-a:+9.2f}")
    print("-" * 54)
    print(f"{'合计':10s} {tot[0]:10.2f} {tot[1]:10.2f} {tot[2]:10.2f} "
          f"{tot[2]-tot[1]:+9.2f}")
    print(f"→ 代理比值：GW only / 预测0 = {tot[1]/tot[0]:.4f}；"
          f"GW+填充 / 预测0 = {tot[2]/tot[0]:.4f}")

    # ---- 4. MAT2A 的 5 个显著基因逐个看 ----
    print("\n=== 4. MAT2A 的显著集（只有 5 个）逐基因 ===")
    pa, lf = de["MAT2A"]
    m = (pa < ALPHA) & np.isfinite(lf) & (gate_sym != "MAT2A")
    for i in np.flatnonzero(m):
        g = gate_sym[i]
        vals = {"real": lf[i]}
        for s_name, d in [("K562GW", gw)] + list(ess.items()):
            if g in d.index and "MAT2A" in d.columns:
                vals[s_name] = float(d.loc[g, "MAT2A"])
        print(f"  {g:12s} " + "  ".join(f"{k}={v:+.4f}" for k, v in vals.items()))
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
