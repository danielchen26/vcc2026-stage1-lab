"""E24 — 量化 E23 的信息泄漏：把同 context 标定换成先验标定后还剩多少。

## 发现的泄漏

E23 的 5 折交叉验证是在 **H1 内部**做的：训练折用同一个 context 里其他扰动的
**真实 R_p** 来学
    (a) K_opt ~ n_src 的回归系数
    (b) 每个箱里「平均 jac 最大的 K」

而本届 context A/B/C **一个标注扰动都没有**（三个 h5ad 全是 non-targeting 对照，
已在 E07 核实）。所以 E23 的「榜首 0.61×」不可部署。

## 本实验：换成三种**不泄漏**的标定

    P0  E23 原样（同 context 标定）           上界，含泄漏
    P1  先验常数 K = 288                    用 F8 的 E|R_p|，完全不看靶侧
    P2  先验比例 K = c * n_src              c 取自**另一个数据集**的关系
                                            （这里用 CD4T 的 n_de/n_src 比值中位）
    P3  先验分区 + 先验 K                    按 n_src 分三档，每档给一个先验 K
                                            （档位边界与 K 值都不看靶侧 R_p）
    P4  全报                                baseline

P1–P3 完全不使用 H1 的任何 R_p 信息 —— 这才是 A/B/C 上真实能做的。

## 判读（一致的中位数标尺，已与官方公布值对齐）

    官方基线 jac 0.021-0.037（我实测全报中位 0.0190 ✓）
    官方 replicate 0.399（我实测 0.379 ✓）
    榜首缩放分 0.1899
    缩放分 = (jac - baseline) / (replicate - baseline)

跑法：  ~/vcc2026/.venv/bin/python experiments/E24-leakage/run.py
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
MAPCSV = (DATA / "external" / "ens2sym.csv") if (DATA / "external" / "ens2sym.csv").exists() else Path("/tmp/ens2sym.csv")
CD4 = DATA / "external" / "cd4_de.csv"
OUT = Path(__file__).parent
ALPHA, Z_BH, SEED = 0.05, 3.184, 0
N_FOLD, N_BIN = 5, 4
R_ANCHOR, LEADER = 0.379, 0.1899
ERP_PRIOR = 288                      # F8 的 E|R_p|，完全先验
KGRID = np.unique(np.round(np.geomspace(1, 7000, 60)).astype(int))


def main() -> None:
    t0 = time.time()
    print("=== E24 量化 E23 的信息泄漏 ===\n")

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

    # 先验比例 c：**只用 CD4T 数据**（另一个数据集）估 n_target/n_source 的关系
    cd = pd.read_csv(CD4)
    r_ = cd[cd.culture_condition == "Rest"].drop_duplicates("target_contrast_gene_name")
    cmap = dict(zip(r_.target_contrast_gene_name.astype(str), r_.n_total_de_genes))
    n_cd4 = np.array([cmap.get(p_, np.nan) for p_ in perts], float)
    both = np.isfinite(n_cd4) & (n_src > 0)
    c_prior = float(np.median(n_cd4[both] / n_src[both]))
    print(f"先验比例 c = median(n_cd4 / n_src) = {c_prior:.2f}"
          f"   （只用 CD4T，不看 H1）")

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
        k = np.arange(1, G + 1)
        CURVE.append(hit / (NR[j] + k - hit))
    valid = np.flatnonzero(NR > 0)
    KOPT = np.array([int(np.argmax(CURVE[j])) + 1 if CURVE[j] is not None else 1
                     for j in range(len(perts))])
    print(f"jac(K) 曲线算完 ({time.time()-t:.0f}s)  G={G:,}  有效扰动 {len(valid)}")
    p_tmp.unlink(missing_ok=True)

    folds = rng.permutation(len(perts)) % N_FOLD
    recs = []
    for f_ in range(N_FOLD):
        tr = np.array([j for j in valid if folds[j] != f_])
        te = np.array([j for j in valid if folds[j] == f_])
        if len(te) == 0:
            continue
        # P0：泄漏版（E23 的做法）
        x = np.log10(n_src[tr] + 1.0)
        co = np.linalg.lstsq(np.column_stack([np.ones_like(x), x]),
                             np.log10(KOPT[tr].clip(min=1)), rcond=None)[0]
        nh_tr = 10 ** (co[0] + co[1] * x)
        edges = np.quantile(np.log10(nh_tr), np.linspace(0, 1, N_BIN + 1))[1:-1]
        bin_tr = np.digitize(np.log10(nh_tr), edges)
        bestk = {}
        for b_ in range(N_BIN):
            mem = tr[bin_tr == b_]
            bestk[b_] = (int(KGRID[int(np.argmax(
                [np.mean([CURVE[j][k - 1] for j in mem]) for k in KGRID]))])
                if len(mem) else G)

        for j in te:
            c = CURVE[j]
            xj = np.log10(n_src[j] + 1.0)
            nh = 10 ** (co[0] + co[1] * xj)
            k_leak = int(np.clip(bestk[int(np.digitize(np.log10(nh), edges))], 1, G))
            # P1 先验常数
            k_p1 = min(ERP_PRIOR, G)
            # P2 先验比例（c 来自 CD4T）
            k_p2 = int(np.clip(round(c_prior * max(n_src[j], 1)), 1, G))
            # P3 先验分区：n_src 分三档，K 用先验的 E|R_p| 缩放
            ns = n_src[j]
            k_p3 = (max(int(0.1 * ERP_PRIOR), 1) if ns < 5 else
                    ERP_PRIOR if ns < 100 else G)
            rec = dict(target_gene=perts[j], n_real=int(NR[j]), n_src=int(ns),
                       k_opt=int(KOPT[j]))
            for tag, k in (("P0_leak", k_leak), ("P1_const", k_p1),
                           ("P2_ratio", k_p2), ("P3_regime", k_p3),
                           ("P4_all", G), ("P5_oracle", int(KOPT[j]))):
                kk = int(np.clip(k, 1, G))
                rec[f"K_{tag}"] = kk
                rec[f"jac_{tag}"] = float(c[kk - 1])
            recs.append(rec)

    df = pd.DataFrame(recs)
    df.to_csv(OUT / "result.csv", index=False)

    base = df.jac_P4_all.median()
    print(f"\n{'='*84}\n结论（n = {len(df)}，一致的中位数标尺）\n{'='*84}")
    print(f"全报（baseline）中位 jac = {base:.4f}"
          f"   官方公布区间 0.021-0.037 → {'吻合' if 0.015 <= base <= 0.045 else '不吻合'}")
    print(f"\n{'规则':>26} {'泄漏?':>6} {'jac 中位':>9} {'缩放分':>9} {'vs 榜首':>9} {'K 中位':>8}")
    order = ("P4_all", "P1_const", "P2_ratio", "P3_regime", "P0_leak", "P5_oracle")
    lab = {"P4_all": "P4 全报（baseline）", "P1_const": "P1 先验常数 K=288",
           "P2_ratio": "P2 先验比例 c×n_src", "P3_regime": "P3 先验分区",
           "P0_leak": "P0 同 context 标定", "P5_oracle": "P5 oracle K_opt"}
    leak = {"P0_leak": "**是**", "P5_oracle": "**是**"}
    for r_ in order:
        v = df[f"jac_{r_}"].median()
        sc = (v - base) / (R_ANCHOR - base)
        print(f"{lab[r_]:>26} {leak.get(r_,'否'):>6} {v:9.4f} {sc:9.4f} "
              f"{sc/LEADER:8.2f}× {df[f'K_{r_}'].median():8.0f}")

    print(f"\n--- 配对检验（vs P4 全报）---")
    for r_ in ("P1_const", "P2_ratio", "P3_regime", "P0_leak"):
        d = df[f"jac_{r_}"] - df.jac_P4_all
        p = wilcoxon(df[f"jac_{r_}"], df.jac_P4_all).pvalue if d.abs().sum() else 1.0
        print(f"  {lab[r_]:>26}: Δ中位 {d.median():+.4f}  胜 {(d>0).sum():2d}/{len(df)}"
              f"  p={p:.5f}")

    bestnl = max(("P1_const", "P2_ratio", "P3_regime"),
                 key=lambda r: df[f"jac_{r}"].median())
    vn = df[f"jac_{bestnl}"].median()
    vl = df.jac_P0_leak.median()
    sn = (vn - base) / (R_ANCHOR - base)
    sl = (vl - base) / (R_ANCHOR - base)
    print(f"\n{'='*84}")
    print(f"不泄漏的最好: {lab[bestnl]}  缩放分 {sn:.4f} = 榜首的 {sn/LEADER:.0%}")
    print(f"泄漏版:       缩放分 {sl:.4f} = 榜首的 {sl/LEADER:.0%}")
    print(f"→ 泄漏贡献了 {1 - sn/sl:.0%} 的表现" if sl > 0 else "")
    print(f"\n>>> {'泄漏不致命，方法可部署' if sn >= 0.7*sl else '泄漏是致命的，必须先解决标定'} <<<")
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
