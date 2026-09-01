"""E20 — 去掉预言机：真实 K 规则下的 Jaccard，回答「到底好不好用」。

## 为什么前面的结论都还不能用

E16–E19 所有 h 都用了**预言机规模** K = |R_p^H1|，即假装已知答案集有多大。
真实比赛里 K 必须自己定。而且：

    h   = |R ∩ R̂| / |R|        **完全不惩罚多报** —— 全报就 h = 1
    jac = |R ∩ R̂| / |R ∪ R̂|    官方用的口径，多报会被惩罚

所以 h + 预言机 K 是双重乐观。本实验两个都去掉。

## K 规则（都不看靶侧答案）

    K1  预言机 K = |R_p^H1|              上界参照，实际拿不到
    K2  固定 K = 253                    E10 实测的 |R_p| 中位
    K3  K = |R_p^K562|                  源侧自己的显著数（BH, alpha=0.05）
    K4  K = a * |R_p^K562|              a 在训练扰动上标定（5 折）
    K5  K = G（全报）                    退化基线，官方 baseline 的做法

排序统一用 E18 选出的最好统计量 D：|beta_K562| / lfcSE_K562，MDE 有限的基因才入选。

## 判读参照（全部实测）

    jac 同系 replicate 锚点  0.379   E11
    jac 追平线              0.068   = h/(2-h)，h = 0.127
    jac 零生物学地板         ~0.012  E13 的 h=0.022 折算

跑法：  ~/vcc2026/.venv/bin/python experiments/E20-realistic-k/run.py
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
MAPCSV = Path("/tmp/ens2sym.csv")
OUT = Path(__file__).parent
ALPHA, Z_BH, SEED = 0.05, 3.184, 0
N_FOLD = 5
JAC_REPLICATE, JAC_TIE = 0.379, 0.127 / (2 - 0.127)
RULES = ("K1_oracle", "K2_fixed253", "K3_source", "K4_calib", "K5_all")
LAB = {"K1_oracle": "K1 预言机（拿不到）", "K2_fixed253": "K2 固定 253",
       "K3_source": "K3 源侧显著数", "K4_calib": "K4 标定 a×源侧",
       "K5_all": "K5 全报（退化基线）"}


def main() -> None:
    t0 = time.time()
    print("=== E20 真实 K 规则下的 Jaccard ===")
    print(f"参照：jac 同系锚点 {JAC_REPLICATE} · 追平线 {JAC_TIE:.4f} · 地板 ~0.012\n")

    with h5py.File(H5, "r") as f:
        h1_genes = [g.decode() if isinstance(g, bytes) else str(g)
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
               var=pd.DataFrame(index=pd.Index(h1_genes))).write_h5ad(p_tmp)
    ref = ControlRef.load(p_tmp, h1_genes)
    gidx = np.asarray(ref.gidx)
    m = mde(ref, n_cells=VCC_PERT_CELLS, alpha=float(2 * norm.sf(Z_BH)),
            seed=SEED, tie_correct=True)
    gate_sym = np.array(h1_genes)[gidx]

    e2s = pd.read_csv(MAPCSV).dropna()
    pfile = DATA / "nadig2025" / "K562GW_p.csv.gz"
    gw_cols = [str(c) for c in pd.read_csv(pfile, index_col=0, nrows=1).columns]
    perts = sorted(set(gw_cols) & (set(cats) - {"non-targeting"}))

    def read_gw(name: str) -> pd.DataFrame:
        d = pd.read_csv(DATA / "nadig2025" / f"{name}.csv.gz", index_col=0,
                        usecols=["Unnamed: 0"] + perts,
                        dtype={c: np.float32 for c in perts}, engine="c")
        d.index = d.index.astype(str)
        return d[perts]

    lfc = read_gw("K562GW_lfc")
    se = read_gw("K562GW_se").reindex(index=lfc.index)
    pv = read_gw("K562GW_p").reindex(index=lfc.index)
    sym = lfc.index.map(dict(zip(e2s.ensembl, e2s.symbol)))
    kp = pd.notna(sym)
    lfc, se, pv, sym = lfc[kp], se[kp], pv[kp], pd.Index(sym[kp])
    dd = ~sym.duplicated()
    lfc, se, pv, sym = lfc[dd], se[dd], pv[dd], sym[dd]
    common = pd.Index(gate_sym).intersection(sym)
    gi_h1, gi_k5 = pd.Index(gate_sym).get_indexer(common), sym.get_indexer(common)
    G = len(common)
    B = lfc.to_numpy().astype(np.float64)[gi_k5]
    S = se.to_numpy().astype(np.float64)[gi_k5]
    P = pv.to_numpy().astype(np.float64)[gi_k5]
    thr = m[gi_h1]
    ok = np.isfinite(thr) & (thr > 0)
    print(f"共同基因 G = {G:,}  扰动 = {len(perts)}")

    # 源侧自己的显著数
    n_src = np.zeros(len(perts), int)
    for j in range(len(perts)):
        c = P[:, j]
        f_ = np.isfinite(c)
        if f_.any():
            n_src[j] = int((bh_adjust(c[f_]) < ALPHA).sum())
    print(f"源侧(K562) 显著数 中位 {np.median(n_src):.0f}  范围 [{n_src.min()}, {n_src.max()}]")

    # 靶侧真集
    t = time.time()
    REAL = np.zeros((G, len(perts)), bool)
    for j, name in enumerate(perts):
        idx = np.flatnonzero(codes == cats.index(name))
        if len(idx) > VCC_PERT_CELLS:
            idx = rng.choice(idx, VCC_PERT_CELLS, replace=False)
        padj, _ = ref.de_table(to_cpm(thin(read_rows(idx), VCC_UMI, rng)),
                               tie_correct=True)
        REAL[:, j] = (padj < ALPHA)[gi_h1]
    NR = REAL.sum(0)
    print(f"靶侧(H1) |R_p| 中位 {np.median(NR):.0f}  ({time.time()-t:.0f}s)")
    print(f"源靶显著数 Spearman: ", end="")
    from scipy.stats import spearmanr
    print(f"{spearmanr(n_src, NR).statistic:.3f}")

    # 标定 a（5 折）
    folds = rng.permutation(len(perts)) % N_FOLD
    a_fold = np.zeros(N_FOLD)
    for f_ in range(N_FOLD):
        tr = np.flatnonzero(folds != f_)
        r = NR[tr] / np.maximum(n_src[tr], 1)
        a_fold[f_] = float(np.median(r))
    print(f"标定系数 a = |R_p^H1|/|R_p^K562| 各折中位: "
          + " ".join(f"{x:.2f}" for x in a_fold))

    recs = []
    print(f"\n{'扰动':>10} {'|R_p|':>6} {'K源':>6} " +
          " ".join(f"{LAB[r][:8]:>9}" for r in RULES))
    for j, name in enumerate(perts):
        n_real = int(NR[j])
        if n_real == 0:
            continue
        b, s = B[:, j], S[:, j]
        good = np.isfinite(b) & np.isfinite(s) & (s > 0) & ok
        if good.sum() < 500:
            continue
        score = np.where(good, np.abs(b) / np.maximum(s, 1e-9), -np.inf)
        order = np.argsort(score)[::-1]
        a = a_fold[folds[j]]
        ks = {"K1_oracle": n_real, "K2_fixed253": 253,
              "K3_source": max(int(n_src[j]), 1),
              "K4_calib": max(int(round(a * n_src[j])), 1), "K5_all": G}
        rec = dict(target_gene=name, n_real=n_real, n_src=int(n_src[j]))
        for r_, k in ks.items():
            k = min(max(k, 1), G)
            sel = order[:k]
            inter = int(REAL[sel, j].sum())
            rec[f"h_{r_}"] = inter / n_real
            rec[f"jac_{r_}"] = inter / (n_real + k - inter)
            rec[f"K_{r_}"] = k
        recs.append(rec)
        print(f"{name:>10} {n_real:6d} {int(n_src[j]):6d} " +
              " ".join(f"{rec['jac_'+r]:9.4f}" for r in RULES))

    df = pd.DataFrame(recs)
    df.to_csv(OUT / "result.csv", index=False)
    p_tmp.unlink(missing_ok=True)

    print(f"\n{'='*80}\n结论（n = {len(df)}）\n{'='*80}")
    print(f"{'K 规则':>22} {'jac 中位':>9} {'jac 均值':>9} {'h 中位':>8} "
          f"{'K 中位':>7} {'vs 追平线':>10}")
    for r_ in RULES:
        jm = df[f"jac_{r_}"].median()
        print(f"{LAB[r_]:>22} {jm:9.4f} {df[f'jac_{r_}'].mean():9.4f} "
              f"{df[f'h_{r_}'].median():8.3f} {df[f'K_{r_}'].median():7.0f} "
              f"{jm/JAC_TIE:9.2f}×")

    print(f"\n--- 配对检验（各规则 vs K5 全报=官方 baseline 的做法）---")
    for r_ in RULES[:-1]:
        d = df[f"jac_{r_}"] - df.jac_K5_all
        p = wilcoxon(df[f"jac_{r_}"], df.jac_K5_all).pvalue if d.abs().sum() else 1.0
        print(f"  {LAB[r_]:>22}: Δjac {d.median():+.4f}  "
              f"胜 {(d>0).sum():2d}/{len(df)}  p={p:.5f}")

    usable = [r for r in RULES[1:-1] if df[f"jac_{r}"].median() >= JAC_TIE]
    best = max(RULES[1:-1], key=lambda r: df[f"jac_{r}"].median())
    jb = df[f"jac_{best}"].median()
    print(f"\n{'='*80}")
    print(f"不用预言机的最好规则: {LAB[best]}   jac = {jb:.4f}")
    print(f"  追平线 {JAC_TIE:.4f} → {jb/JAC_TIE:.2f}×   "
          f"同系锚点 {JAC_REPLICATE} → {jb/JAC_REPLICATE:.0%}")
    print(f"\n>>> {'有可用方法：' if usable else '尚无可用方法：'}"
          f"{LAB[best]} {'超过' if jb >= JAC_TIE else '未达'}追平线 <<<")
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
