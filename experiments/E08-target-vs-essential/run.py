"""E08 — 本届靶基因 vs 必需基因：在同一细胞系里量化必需性错配的折扣。

为什么这是 TAP 框架的阻塞前置（见 docs/08-framework.md 第 7 节）：
    E07 在必需基因面板上测出跨系可迁移性，但本届 300 个靶基因与该面板**零重叠**
    （0/300，二项检验 p = 2.03e-17，主办方刻意排除）。E07 的数要打多少折未知。

为什么 K562GenomeWide 能回答：
    它一个文件里同时含必需扰动（~2,057）和非必需扰动（共 9,866），
    其中本届 272/300 个靶基因在内。**同细胞系、同实验、同 DESeq2 管线**，
    所以组间差异只能来自扰动本身的性质，没有批次/平台混淆。

四个测量：
    1. |R_p| 分布      —— 官方锚点反推 E|R_p| ≈ 288，非必需靶基因是否成立
    2. 同系可重复性     —— fission h，定出折扣系数
    3. 功效是否可比     —— lfcSE 中位数，决定要不要功效匹配
    4. 通用轴是否同一   —— 枢纽度向量在两组间的相关。这是 TAP 里
                          「u_c 可从必需数据估出、用于非必需靶基因」的直接检验

跑法：  ~/vcc2026/.venv/bin/python experiments/E08-target-vs-essential/run.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vcclab.crossline import fission, h_topk, rank_matrix  # noqa: E402
from vcclab.scorer import bh_adjust  # noqa: E402

DATA = ROOT / "data" / "nadig2025"
GW = DATA / "K562GW_p.csv.gz"
ALPHA = 0.05
SEED = 0
K_HUB = 288          # 枢纽度用的 top-K，取官方反推的 E|R_p|
N_OTHER = 1200       # 参照组：随机抽的其他非必需扰动
RNG = np.random.default_rng(SEED)


def gw_columns() -> list[str]:
    return [str(c) for c in pd.read_csv(GW, index_col=0, nrows=1).columns]


def read_gw(kind: str, cols: list[str]) -> pd.DataFrame:
    f = {"p": "K562GW_p", "lfc": "K562GW_lfc", "se": "K562GW_se"}[kind]
    df = pd.read_csv(DATA / f"{f}.csv.gz", index_col=0,
                     usecols=["Unnamed: 0"] + cols,
                     dtype={c: np.float32 for c in cols}, engine="c")
    df.index = df.index.astype(str)
    return df[cols]          # 必需：usecols 按文件列序返回（E07 踩过的坑）


def bh_reject(pvals: np.ndarray) -> np.ndarray:
    out = np.zeros(pvals.shape, bool)
    for j in range(pvals.shape[1]):
        col = pvals[:, j]
        ok = np.isfinite(col)
        if not ok.any():
            continue
        out[np.flatnonzero(ok), j] = bh_adjust(col[ok].astype(np.float64)) < ALPHA
    return out


def hubness(rank: np.ndarray, k: int = K_HUB) -> np.ndarray:
    """每基因的上榜频率：在多少比例的扰动里进了 top-k。这就是通用响应轴的经验刻画。"""
    return (rank < k).mean(1)


def main() -> None:
    t0 = time.time()
    print("=== E08 本届靶基因 vs 必需基因（同一细胞系 K562）===\n")

    gw = gw_columns()
    gw_set = set(gw)
    ess = {str(c) for c in pd.read_csv(DATA / "K562Essential_p.csv.gz",
                                      index_col=0, nrows=1).columns}
    vcc = {str(t) for t in pd.read_csv(Path.home() / "vcc2026" / "pert_counts.csv"
                                       )["target_gene"]} - {"non-targeting"}

    g_ess = sorted(gw_set & ess)
    g_vcc = sorted(gw_set & vcc)
    rest = sorted(gw_set - ess - vcc)
    g_oth = sorted(RNG.choice(rest, size=min(N_OTHER, len(rest)), replace=False))

    print(f"K562GenomeWide 扰动数 = {len(gw):,}")
    print(f"  必需组   E = {len(g_ess):,}")
    print(f"  本届靶组 V = {len(g_vcc):,}  ({len(g_vcc)/len(vcc):.1%} of 300)")
    print(f"  其他参照 O = {len(g_oth):,}  (从 {len(rest):,} 个随机抽)\n")

    groups = {"E 必需": g_ess, "V 本届靶": g_vcc, "O 其他非必需": g_oth}
    cols = sorted(set(g_ess) | set(g_vcc) | set(g_oth))

    t = time.time()
    p = read_gw("p", cols)
    lfc = read_gw("lfc", cols).reindex(index=p.index)
    se = read_gw("se", cols).reindex(index=p.index)
    print(f"读入 {p.shape[0]:,} 基因 × {p.shape[1]:,} 扰动  ({time.time()-t:.0f}s)\n")

    pv = p.to_numpy()
    b = lfc.to_numpy().astype(np.float64)
    s = se.to_numpy().astype(np.float64)
    colpos = {c: i for i, c in enumerate(cols)}

    with np.errstate(divide="ignore", invalid="ignore"):
        b1, b2, s1, s2 = fission(b, s, RNG, tau=1.0)
        p_wald = (2 * norm.sf(np.abs(b / s))).astype(np.float32)
        p_f1 = (2 * norm.sf(np.abs(b1 / s1))).astype(np.float32)
        p_f2 = (2 * norm.sf(np.abs(b2 / s2))).astype(np.float32)
    del b1, b2, s1, s2

    sig_theirs = bh_reject(pv)
    sig_f1, sig_f2 = bh_reject(p_f1), bh_reject(p_f2)
    r_f1, r_f2 = rank_matrix(p_f1), rank_matrix(p_f2)
    r_wald = rank_matrix(p_wald)

    # ---------------- 1 & 3：|R_p| 与功效 ----------------
    print("--- 1&3. |R_p| 分布与功效 ---")
    print(f"{'组':>14} {'n':>6} {'|R_p| 中位':>10} {'IQR':>16} "
          f"{'lfcSE 中位':>11} {'|lfc| 中位':>11}")
    stats = {}
    for name, gl in groups.items():
        idx = np.array([colpos[c] for c in gl])
        ns = sig_theirs[:, idx].sum(0)
        q1, q3 = np.percentile(ns, [25, 75])
        mse_ = float(np.median(s[:, idx]))
        mlfc = float(np.median(np.abs(b[:, idx])))
        stats[name] = dict(idx=idx, n_sig=ns, med=float(np.median(ns)),
                           se=mse_, lfc=mlfc)
        print(f"{name:>14} {len(gl):6d} {np.median(ns):10.0f} "
              f"{f'[{q1:.0f}, {q3:.0f}]':>16} {mse_:11.4f} {mlfc:11.4f}")
    print(f"\n  官方锚点反推 E|R_p| ≈ 288 (范围 207–365)")
    v_med, e_med = stats["V 本届靶"]["med"], stats["E 必需"]["med"]
    print(f"  V/E 的 |R_p| 比 = {v_med:.0f}/{e_med:.0f} = "
          f"{v_med/max(e_med,1e-9):.3f}")

    # ---------------- 2：同系可重复性 ----------------
    print("\n--- 2. 同系可重复性（fission 伪重复）---")
    print(f"{'组':>14} {'h_BH':>8} {'h_top25':>9} {'h_top100':>10} {'h_top288':>10}")
    hh = {}
    for name, gl in groups.items():
        idx = stats[name]["idx"]
        a_, b_ = sig_f1[:, idx], sig_f2[:, idx]
        na, nb = a_.sum(0).astype(float), b_.sum(0).astype(float)
        den = 0.5 * (na + nb)
        hbh = np.divide((a_ & b_).sum(0), den, out=np.full_like(den, np.nan),
                        where=den > 0)
        row = {f"top{k}": h_topk(r_f1[:, idx], r_f2[:, idx], k)
               for k in (25, 100, 288)}
        row["bh"] = float(np.nanmedian(hbh))
        hh[name] = row
        print(f"{name:>14} {row['bh']:8.3f} {row['top25']:9.3f} "
              f"{row['top100']:10.3f} {row['top288']:10.3f}")

    print("\n  折扣系数 = V / E：")
    for k in ("bh", "top25", "top100", "top288"):
        e_, v_ = hh["E 必需"][k], hh["V 本届靶"][k]
        print(f"    {k:>7}: {v_:.3f} / {e_:.3f} = {v_/max(e_,1e-9):.3f}")

    # ---------------- 4：通用轴是否同一 ----------------
    print("\n--- 4. 通用响应轴：枢纽度向量在组间是否同一 ---")
    hub = {n: hubness(r_wald[:, stats[n]["idx"]]) for n in groups}
    names = list(groups)
    for i, a in enumerate(names):
        for b_ in names[i + 1:]:
            rho = spearmanr(hub[a], hub[b_]).statistic
            top_a = set(np.argsort(hub[a])[::-1][:K_HUB])
            top_b = set(np.argsort(hub[b_])[::-1][:K_HUB])
            jac = len(top_a & top_b) / len(top_a | top_b)
            print(f"  {a:>14} vs {b_:<14} Spearman ρ = {rho:6.3f}   "
                  f"top-{K_HUB} Jaccard = {jac:.3f}")
    for n in names:
        print(f"  {n:>14} 枢纽度: 中位 {np.median(hub[n]):.4f}  "
              f"p95 {np.percentile(hub[n],95):.4f}  max {hub[n].max():.4f}")

    print(f"\n耗时 {time.time()-t0:.0f}s")

    out = Path(__file__).parent
    pd.DataFrame([dict(group=n, n_pert=len(groups[n]), median_Rp=stats[n]["med"],
                       median_lfcSE=stats[n]["se"], median_abs_lfc=stats[n]["lfc"],
                       **{f"h_{k}": v for k, v in hh[n].items()})
                  for n in names]).to_csv(out / "result.csv", index=False)
    np.savez_compressed(out / "hubness.npz", genes=np.array(p.index),
                        **{f"hub_{i}": hub[n] for i, n in enumerate(names)})


if __name__ == "__main__":
    main()
