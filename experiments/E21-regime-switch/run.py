"""E21 — 分区制 K 选择：唯一剩下的已定位路径。

## E20 留下的三个事实

    分层         n     jac 预言机K   jac 固定253   jac 全报
    |R_p|<50    17     0.0645       0.0040      0.0006
    50-500      13     0.0350       0.0315      0.0190
    >500        17     0.2057       0.0476      0.3129   ← 全报比预言机 K 还好

    源侧显著数 vs 靶侧 |R_p| 的 Spearman = 0.722   → 分区可预测
    K 选择贡献了 67% 的损失（预言机 0.0957 → 固定 253 的 0.0315）

最优策略是**分区制**：高响应全报、低响应少报。而且分区可判。

## 本实验

排序固定用 E18 选出的 D（|beta_K562|/lfcSE，MDE 有限的基因才入选）。
在**扰动维度**做 5 折交叉验证，训练折上学「最优 K 是源侧显著数的什么函数」：

    1. 对训练扰动，算出 jac(K) 曲线的极大点 K_opt（给定我们的排序与真集）
    2. 拟合 log(K_opt) ~ 线性函数 of log(n_src)     ← 只用源侧可得的量
    3. 在留出扰动上用预测的 K，评估 jac

对照：
    R1  预测 K（本实验）
    R2  全报 K=G                 = 官方 baseline 的做法，mean jac 0.1198
    R3  固定 253
    R4  oracle K_opt             上界，拿不到
    R5  oracle |R_p|             E20 的 K1

## 判读标尺（已修正）

先前用 jac = h/(2-h) 把追平线折算成 0.0678 是**错的** —— 该恒等式只在
|R̂| = |R| 时成立，而全报时 h=1.0 而 jac≈0.12，两者不可换算。
用锚点重推：baseline raw jac ≈ 0.12、replicate 0.379、榜首缩放分 0.1899

    追平线 raw jac ≈ 0.12 + 0.1899 × (0.379 - 0.12) = 0.169

**且必须看均值**（分数按扰动平均），不是中位数。

跑法：  ~/vcc2026/.venv/bin/python experiments/E21-regime-switch/run.py
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
JAC_BASELINE, JAC_REPLICATE, LEADER_SCALED = 0.12, 0.379, 0.1899
JAC_TIE = JAC_BASELINE + LEADER_SCALED * (JAC_REPLICATE - JAC_BASELINE)
RULES = ("R1_pred", "R2_all", "R3_fixed", "R4_oracleK", "R5_oracleN")
LAB = {"R1_pred": "R1 预测 K（本实验）", "R2_all": "R2 全报（=baseline）",
       "R3_fixed": "R3 固定 253", "R4_oracleK": "R4 oracle K_opt",
       "R5_oracleN": "R5 oracle |R_p|"}


def jac_curve(real: np.ndarray, order: np.ndarray, n_real: int) -> np.ndarray:
    """jac(K) 对 K=1..G 的完整曲线。"""
    hit = np.cumsum(real[order])
    k = np.arange(1, len(order) + 1)
    return hit / (n_real + k - hit)


def main() -> None:
    t0 = time.time()
    print("=== E21 分区制 K 选择 ===")
    print(f"标尺（已修正）：baseline raw jac {JAC_BASELINE} · replicate {JAC_REPLICATE}")
    print(f"  → 追平线 raw jac = {JAC_TIE:.4f}，**看均值不看中位数**\n")

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
    gw_cols = [str(c) for c in pd.read_csv(DATA / "nadig2025" / "K562GW_p.csv.gz",
                                          index_col=0, nrows=1).columns]
    perts = sorted(set(gw_cols) & (set(cats) - {"non-targeting"}))

    def read_gw(n: str) -> pd.DataFrame:
        d = pd.read_csv(DATA / "nadig2025" / f"{n}.csv.gz", index_col=0,
                        usecols=["Unnamed: 0"] + perts,
                        dtype={c: np.float32 for c in perts}, engine="c")
        d.index = d.index.astype(str)
        return d[perts]

    lfc = read_gw("K562GW_lfc")
    se = read_gw("K562GW_se").reindex(index=lfc.index)
    pv = read_gw("K562GW_p").reindex(index=lfc.index)
    sym = lfc.index.map(dict(zip(e2s.ensembl, e2s.symbol)))
    kk = pd.notna(sym)
    lfc, se, pv, sym = lfc[kk], se[kk], pv[kk], pd.Index(sym[kk])
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

    n_src = np.zeros(len(perts), int)
    for j in range(len(perts)):
        c = P[:, j]
        f_ = np.isfinite(c)
        if f_.any():
            n_src[j] = int((bh_adjust(c[f_]) < ALPHA).sum())

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
    print(f"G = {G:,}  扰动 = {len(perts)}  靶侧 |R_p| 中位 {np.median(NR):.0f}"
          f"  ({time.time()-t:.0f}s)")

    # ---- 每扰动的排序与 jac(K) 曲线 ----
    ORD, CURVE, KOPT = [], [], np.zeros(len(perts), int)
    for j in range(len(perts)):
        b, s = B[:, j], S[:, j]
        good = np.isfinite(b) & np.isfinite(s) & (s > 0) & ok
        score = np.where(good, np.abs(b) / np.maximum(s, 1e-9), -np.inf)
        order = np.argsort(score)[::-1]
        ORD.append(order)
        if NR[j] == 0:
            CURVE.append(None); KOPT[j] = 1; continue
        c = jac_curve(REAL[:, j], order, int(NR[j]))
        CURVE.append(c)
        KOPT[j] = int(np.argmax(c)) + 1
    valid = NR > 0
    print(f"有效扰动 {valid.sum()}   K_opt 中位 {np.median(KOPT[valid]):.0f}"
          f"  范围 [{KOPT[valid].min()}, {KOPT[valid].max()}]")
    print(f"log10(K_opt) vs log10(n_src+1) 相关: "
          f"{np.corrcoef(np.log10(KOPT[valid]), np.log10(n_src[valid]+1))[0,1]:.3f}")

    # ---- 5 折：学 K_opt ~ n_src ----
    folds = rng.permutation(len(perts)) % N_FOLD
    recs = []
    for f_ in range(N_FOLD):
        tr = np.flatnonzero((folds != f_) & valid)
        te = np.flatnonzero((folds == f_) & valid)
        x = np.log10(n_src[tr] + 1.0)
        y = np.log10(KOPT[tr])
        A = np.column_stack([np.ones_like(x), x])
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        for j in te:
            xj = np.log10(n_src[j] + 1.0)
            k_pred = int(np.clip(round(10 ** (coef[0] + coef[1] * xj)), 1, G))
            c = CURVE[j]
            ks = {"R1_pred": k_pred, "R2_all": G, "R3_fixed": min(253, G),
                  "R4_oracleK": KOPT[j], "R5_oracleN": int(min(NR[j], G))}
            rec = dict(target_gene=perts[j], n_real=int(NR[j]),
                       n_src=int(n_src[j]), k_pred=k_pred, k_opt=int(KOPT[j]))
            for r_, k in ks.items():
                rec[f"jac_{r_}"] = float(c[max(k, 1) - 1])
            recs.append(rec)

    df = pd.DataFrame(recs)
    df.to_csv(OUT / "result.csv", index=False)
    p_tmp.unlink(missing_ok=True)

    print(f"\n{'='*80}\n结论（n = {len(df)}，5 折扰动级交叉验证）\n{'='*80}")
    print(f"{'规则':>24} {'jac 均值':>10} {'jac 中位':>10} {'K 中位':>8} {'vs baseline':>12} {'vs 追平线':>10}")
    for r_ in RULES:
        v = df[f"jac_{r_}"]
        print(f"{LAB[r_]:>24} {v.mean():10.4f} {v.median():10.4f} "
              f"{'—' if r_ in ('R2_all',) else f'{df.k_pred.median():.0f}' if r_=='R1_pred' else '—':>8} "
              f"{v.mean()/JAC_BASELINE:11.2f}× {v.mean()/JAC_TIE:9.2f}×")

    print(f"\n--- 配对检验（vs R2 全报 = baseline）---")
    for r_ in ("R1_pred", "R3_fixed", "R4_oracleK", "R5_oracleN"):
        d = df[f"jac_{r_}"] - df.jac_R2_all
        p = wilcoxon(df[f"jac_{r_}"], df.jac_R2_all).pvalue if d.abs().sum() else 1.0
        print(f"  {LAB[r_]:>24}: Δjac 均值 {d.mean():+.4f}  中位 {d.median():+.4f}  "
              f"胜 {(d>0).sum():2d}/{len(df)}  p={p:.5f}")

    m1, m2 = df.jac_R1_pred.mean(), df.jac_R2_all.mean()
    d = df.jac_R1_pred - df.jac_R2_all
    p = wilcoxon(df.jac_R1_pred, df.jac_R2_all).pvalue
    print(f"\n{'='*80}")
    ok_beat = (m1 > m2) and (p < 0.05)
    print(f"R1 预测 K: 均值 jac {m1:.4f} vs baseline {m2:.4f}  "
          f"({m1/m2:.2f}×, p={p:.5f})")
    print(f"追平线 {JAC_TIE:.4f} → {m1/JAC_TIE:.2f}×")
    print(f"\n>>> {'有好结果：显著超过 baseline' if ok_beat else '仍无好结果：未显著超过 baseline'} <<<")
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
