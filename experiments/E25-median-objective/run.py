"""E25 — 中位数目标 vs 均值目标：我们一直在优化错的函数。

## 第一性原理

分数 = median_p jac_p（已验证：复现的平凡基线中位数 0.0190 落在官方 0.021-0.037 内）。
而 E23 的最好策略是「每箱取使**平均** jac 最大的 K」—— 优化的是均值。

中位数只取决于第 150 个顺序统计量，所以：
    中位数以上的扰动，改进价值 = 0
    远低于中位数的扰动，改进价值 = 0
    只有中位数附近的扰动有价值

正确形式是**覆盖问题**：
    max t   s.t.   #{p : jac_p(K_p) >= t} >= n/2

给定 t，每个扰动应选 K 最大化 P(jac_p >= t)，而不是 E[jac_p]。

## 顺带纠正 E24 的一个框架错误

E24 把「在 H1 上学 K 规则」判为泄漏，因为**评估也在 H1 上**。
但部署到 A/B/C 时，在 H1 上学规则**不是泄漏**，是跨 context 的超参迁移 —— 合法。
真正的问题是我们只有一个带标注的 context，**无法验证该迁移**。

所以可部署分数在 E24 的 0.25×（纯先验）与 E23 的 0.61×（H1 学 + H1 评）之间，
具体位置取决于 K 规则的跨 context 迁移性，而那是不可测的。本实验两种框架都报。

## 策略

统一排序：|beta_K562| / lfcSE_K562，MDE 有限的基因才入选（E18 选出的最好排序）。
5 折扰动级交叉验证，训练折上定 K 规则，留出折评估。

    M0  全报（baseline）
    M1  每箱取「平均 jac 最大」的 K        ← E23 的做法，优化均值
    M2  每箱取「中位 jac 最大」的 K        ← 换成中位数
    M3  覆盖策略：扫 t，每箱取「使 jac>=t 的比例最大」的 K，
        取覆盖率仍 >= 50% 的最大 t          ← 直接优化顺序统计量
    M4  oracle K_opt（上界）

主判据是**留出折 jac 的中位数**（= 官方口径）。

跑法：  ~/vcc2026/.venv/bin/python experiments/E25-median-objective/run.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import norm, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vcclab.detectability import mde  # noqa: E402
from vcclab.scorer import ControlRef, bh_adjust  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "E10-h1-erp"))
from run import VCC_CTRL_CELLS, VCC_PERT_CELLS, VCC_UMI, read_rows, thin, to_cpm  # noqa: E402

DATA = ROOT / "data"
H5 = DATA / "vcc2025" / "adata_Validation.h5ad"
MAPCSV = DATA / "external" / "ens2sym.csv"
OUT = Path(__file__).parent
ALPHA, Z_BH, SEED = 0.05, 3.184, 0
N_FOLD, N_BIN = 5, 4
R_ANCHOR, LEADER = 0.379, 0.1899
KGRID = np.unique(np.round(np.geomspace(1, 7000, 80)).astype(int))
T_GRID = np.linspace(0.005, 0.60, 120)
RULES = ("M0_all", "M1_mean", "M2_median", "M3_coverage", "M4_oracle")
LAB = {"M0_all": "M0 全报(baseline)", "M1_mean": "M1 每箱最大平均jac",
       "M2_median": "M2 每箱最大中位jac", "M3_coverage": "M3 覆盖策略",
       "M4_oracle": "M4 oracle K_opt"}


def main() -> None:
    t0 = time.time()
    print("=== E25 中位数目标 vs 均值目标 ===\n")

    with h5py.File(H5, "r") as f:
        h1g = [g.decode() if isinstance(g, bytes) else str(g)
               for g in f["var"]["_index"][:]]
        o = f["obs"]["target_gene"]
        cats = [c.decode() if isinstance(c, bytes) else str(c)
                for c in o["categories"][:]]
        codes = o["codes"][:]
    ntc = cats.index("non-targeting")
    rng = np.random.default_rng(SEED)

    rows = rng.choice(np.flatnonzero(codes == ntc), VCC_CTRL_CELLS, replace=False)
    p_tmp = OUT / "_ntc.h5ad"
    ad.AnnData(X=sp.csr_matrix(thin(read_rows(rows), VCC_UMI, rng)),
               var=pd.DataFrame(index=pd.Index(h1g))).write_h5ad(p_tmp)
    ref = ControlRef.load(p_tmp, h1g)
    gidx = np.asarray(ref.gidx)
    m = mde(ref, n_cells=VCC_PERT_CELLS, alpha=float(2 * norm.sf(Z_BH)),
            seed=SEED, tie_correct=True)
    gate_sym = np.array(h1g)[gidx]

    e2s = pd.read_csv(MAPCSV).dropna()
    gw_cols = [str(c) for c in pd.read_csv(DATA / "nadig2025" / "K562GW_p.csv.gz",
                                          index_col=0, nrows=1).columns]
    perts = sorted(set(gw_cols) & (set(cats) - {"non-targeting"}))

    def rd(n):
        d = pd.read_csv(DATA / "nadig2025" / f"{n}.csv.gz", index_col=0,
                        usecols=["Unnamed: 0"] + perts,
                        dtype={c: np.float32 for c in perts}, engine="c")
        d.index = d.index.astype(str)
        return d[perts]

    lfc, se, pv = rd("K562GW_lfc"), rd("K562GW_se"), rd("K562GW_p")
    se, pv = se.reindex(index=lfc.index), pv.reindex(index=lfc.index)
    sym = lfc.index.map(dict(zip(e2s.ensembl, e2s.symbol)))
    kp = pd.notna(sym)
    lfc, se, pv, sym = lfc[kp], se[kp], pv[kp], pd.Index(sym[kp])
    dd = ~sym.duplicated()
    lfc, se, pv, sym = lfc[dd], se[dd], pv[dd], sym[dd]
    common = pd.Index(gate_sym).intersection(sym)
    gi_h1, gi_k5 = pd.Index(gate_sym).get_indexer(common), sym.get_indexer(common)
    G = len(common)
    B, S, P = (x.to_numpy().astype(np.float64)[gi_k5] for x in (lfc, se, pv))
    thr = m[gi_h1]
    ok = np.isfinite(thr) & (thr > 0)
    n_src = np.array([int((bh_adjust(P[np.isfinite(P[:, j]), j]) < ALPHA).sum())
                      if np.isfinite(P[:, j]).any() else 0 for j in range(len(perts))])

    t = time.time()
    CURVE, NR = [], np.zeros(len(perts), int)
    for j, name in enumerate(perts):
        idx = np.flatnonzero(codes == cats.index(name))
        if len(idx) > VCC_PERT_CELLS:
            idx = rng.choice(idx, VCC_PERT_CELLS, replace=False)
        padj, _ = ref.de_table(to_cpm(thin(read_rows(idx), VCC_UMI, rng)),
                               tie_correct=True)
        real = (padj < ALPHA)[gi_h1]
        NR[j] = int(real.sum())
        b, s = B[:, j], S[:, j]
        good = np.isfinite(b) & np.isfinite(s) & (s > 0) & ok
        order = np.argsort(np.where(good, np.abs(b) / np.maximum(s, 1e-9), -np.inf))[::-1]
        if NR[j] == 0:
            CURVE.append(None); continue
        hit = np.cumsum(real[order])
        CURVE.append(hit / (NR[j] + np.arange(1, G + 1) - hit))
    print(f"jac(K) 曲线算完 ({time.time()-t:.0f}s)  G={G:,}  "
          f"|R_p| 中位 {np.median(NR):.0f}")
    p_tmp.unlink(missing_ok=True)
    valid = np.flatnonzero(NR > 0)
    KOPT = np.array([int(np.argmax(CURVE[j])) + 1 if CURVE[j] is not None else 1
                     for j in range(len(perts))])
    CG = np.array([[CURVE[j][k - 1] if CURVE[j] is not None else 0.0 for k in KGRID]
                   for j in range(len(perts))])          # 扰动 × KGRID

    folds = rng.permutation(len(perts)) % N_FOLD
    recs = []
    for f_ in range(N_FOLD):
        tr = np.array([j for j in valid if folds[j] != f_])
        te = np.array([j for j in valid if folds[j] == f_])
        if len(te) == 0:
            continue
        # 分箱只用源侧可得量（n_src），不看靶侧
        x = np.log10(n_src[tr] + 1.0)
        edges = np.quantile(x, np.linspace(0, 1, N_BIN + 1))[1:-1]
        bin_tr = np.digitize(x, edges)
        bin_te = np.digitize(np.log10(n_src[te] + 1.0), edges)

        best = {r: {} for r in ("M1_mean", "M2_median", "M3_coverage")}
        for b_ in range(N_BIN):
            mem = tr[bin_tr == b_]
            if len(mem) == 0:
                for r in best:
                    best[r][b_] = G
                continue
            sub = CG[mem]                                  # 成员 × KGRID
            best["M1_mean"][b_] = int(KGRID[int(np.argmax(sub.mean(0)))])
            best["M2_median"][b_] = int(KGRID[int(np.argmax(np.median(sub, 0)))])
            # 覆盖策略：找覆盖率仍 >= 50% 的最大 t，取该 t 下覆盖率最大的 K
            chosen, best_t = G, -1.0
            for tt in T_GRID:
                cov = (sub >= tt).mean(0)                  # 每个 K 的覆盖率
                if cov.max() >= 0.5:
                    best_t = tt
                    chosen = int(KGRID[int(np.argmax(cov))])
            best["M3_coverage"][b_] = chosen if best_t > 0 else best["M1_mean"][b_]

        for i, j in enumerate(te):
            c = CURVE[j]
            ks = {"M0_all": G, "M4_oracle": int(KOPT[j])}
            for r in ("M1_mean", "M2_median", "M3_coverage"):
                ks[r] = int(np.clip(best[r][int(bin_te[i])], 1, G))
            rec = dict(target_gene=perts[j], n_real=int(NR[j]), n_src=int(n_src[j]),
                       fold=f_)
            for r, k in ks.items():
                rec[f"jac_{r}"] = float(c[int(np.clip(k, 1, G)) - 1])
                rec[f"K_{r}"] = int(np.clip(k, 1, G))
            recs.append(rec)

    df = pd.DataFrame(recs)
    df.to_csv(OUT / "result.csv", index=False)
    base = df.jac_M0_all.median()
    sc = lambda v: (v - base) / (R_ANCHOR - base)

    print(f"\n{'='*84}\n结论（n = {len(df)}，官方口径 = 中位数）\n{'='*84}")
    print(f"{'策略':>22} {'jac 中位':>9} {'jac 均值':>9} {'缩放分':>9} "
          f"{'vs 榜首':>8} {'K 中位':>8}")
    for r in RULES:
        v = df[f"jac_{r}"]
        print(f"{LAB[r]:>22} {v.median():9.4f} {v.mean():9.4f} {sc(v.median()):9.4f} "
              f"{sc(v.median())/LEADER:7.2f}× {df[f'K_{r}'].median():8.0f}")

    print(f"\n--- 配对检验（vs M1 优化均值 = E23 的做法）---")
    for r in ("M2_median", "M3_coverage", "M4_oracle"):
        d = df[f"jac_{r}"] - df.jac_M1_mean
        p = wilcoxon(df[f"jac_{r}"], df.jac_M1_mean).pvalue if d.abs().sum() else 1.0
        print(f"  {LAB[r]:>22}: Δ中位 {d.median():+.4f}  胜 {(d>0).sum():2d}/{len(df)}"
              f"  p={p:.5f}")

    b1 = max(("M1_mean", "M2_median", "M3_coverage"),
             key=lambda r: df[f"jac_{r}"].median())
    print(f"\n{'='*84}")
    print(f"最好（非 oracle）: {LAB[b1]}   中位 jac {df[f'jac_{b1}'].median():.4f}"
          f"   缩放分 {sc(df[f'jac_{b1}'].median()):.4f} = 榜首的 "
          f"{sc(df[f'jac_{b1}'].median())/LEADER:.0%}")
    print(f"E23 的均值目标（M1）: {sc(df.jac_M1_mean.median())/LEADER:.0%}")
    gain = sc(df[f'jac_{b1}'].median()) / max(sc(df.jac_M1_mean.median()), 1e-9)
    print(f"→ 换成中位数/覆盖目标的净收益: {gain:.2f}×")
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
