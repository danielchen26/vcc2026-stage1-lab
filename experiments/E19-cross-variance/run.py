"""E19 — 填上框架 §6 缺失的跨系方差项，做扰动级交叉验证。

## E18 留下的问题

E18 证实了 E17 的诊断：跨系排序改用 |beta|/lfcSE + MDE 硬门限，
机会校正后从 0.0327 提到 0.0501（1.53×，30/47 胜，p < 1e-5）。

但同系的增益是 2.15×，跨系只有 1.53×。机制清楚：

    同系：主导噪声 = 测量噪声，lfcSE 正好刻画它 → 归一化很有效
    跨系：主导噪声 = **生物学差异**，lfcSE 完全不刻画它 → 归一化只治了小头

正确的归一化应该是

    z = |beta_source| / sqrt(lfcSE^2 + sigma_cross^2)

其中 sigma_cross(g) 是基因 g 的效应量在细胞系间的散布 —— 这正是
docs/08-framework.md §6 里标为「必须外推」的那一项，一直没填。

## 怎样非循环地估 sigma_cross

sigma_cross 主要是**基因特异**的（反映该基因跨细胞系行为差异有多大），
与具体扰动关系较弱。所以可以在**扰动维度**上做交叉验证：

    用训练扰动估   sigma_cross(g) = std_p [ beta_H1(g,p) - beta_K562(g,p) ]
    在留出扰动上评估

这对**被评估的扰动**是非循环的。（对细胞系仍是同一对，所以这验证的是
「该函数形式是否有用」，不是「能否泛化到新细胞系」—— 后者需要第三个细胞系。）

同时报一个更保守的变体：把 sigma_cross 建成基因特征的函数
（表达量、lfcSE、跨扰动上榜频率），而不是逐基因自由参数 —— 这样即使
逐基因估计有过拟合，函数形式仍可迁移。

## 对照

    D.  |beta|/lfcSE + MDE 门限            E18 的最好结果
    F.  |beta|/sqrt(lfcSE^2+sigma_x^2)     逐基因 sigma_cross
    G.  同上但 sigma_cross 用特征回归        更保守
    H.  只用 sigma_cross 归一（去掉 lfcSE）  检验哪一项在起作用

跑法：  ~/vcc2026/.venv/bin/python experiments/E19-cross-variance/run.py
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
from vcclab.scorer import ControlRef  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "E10-h1-erp"))
from run import VCC_CTRL_CELLS, VCC_PERT_CELLS, VCC_UMI, read_rows, thin, to_cpm  # noqa: E402

DATA = ROOT / "data"
H5 = DATA / "vcc2025" / "adata_Validation.h5ad"
MAPCSV = Path("/tmp/ens2sym.csv")
OUT = Path(__file__).parent
ALPHA, Z_BH, SEED, EPS = 0.05, 3.184, 0, 1.0
N_FOLD = 5
H_TIE, H_CEIL = 0.127, 0.500
KEYS = ("D_lfcse", "F_sigma_gene", "G_sigma_feat", "H_sigma_only")
LAB = {"D_lfcse": "D. |b|/lfcSE (E18)", "F_sigma_gene": "F. 加逐基因 sigma_x",
       "G_sigma_feat": "G. 加特征回归 sigma_x", "H_sigma_only": "H. 只用 sigma_x"}


def main() -> None:
    t0 = time.time()
    print("=== E19 填上跨系方差项（扰动级 5 折交叉验证）===\n")

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
    ctrl_raw = thin(read_rows(rows), VCC_UMI, rng)
    p_tmp = OUT / "_ntc.h5ad"
    ad.AnnData(X=sp.csr_matrix(ctrl_raw),
               var=pd.DataFrame(index=pd.Index(h1_genes))).write_h5ad(p_tmp)
    ref = ControlRef.load(p_tmp, h1_genes)
    gidx = np.asarray(ref.gidx)
    m = mde(ref, n_cells=VCC_PERT_CELLS, alpha=float(2 * norm.sf(Z_BH)),
            seed=SEED, tie_correct=True)
    gate_sym = np.array(h1_genes)[gidx]
    ctrl_cpm = np.asarray(ctrl_raw[:, gidx].multiply(
        1e6 / np.maximum(np.asarray(ctrl_raw.sum(1)), 1.0)).mean(0)).ravel()

    e2s = pd.read_csv(MAPCSV).dropna()
    gw_cols = [str(c) for c in pd.read_csv(DATA / "nadig2025" / "K562GW_p.csv.gz",
                                          index_col=0, nrows=1).columns]
    perts = sorted(set(gw_cols) & (set(cats) - {"non-targeting"}))

    def read_gw(kind: str) -> pd.DataFrame:
        fn = {"lfc": "K562GW_lfc", "se": "K562GW_se"}[kind]
        d = pd.read_csv(DATA / "nadig2025" / f"{fn}.csv.gz", index_col=0,
                        usecols=["Unnamed: 0"] + perts,
                        dtype={c: np.float32 for c in perts}, engine="c")
        d.index = d.index.astype(str)
        return d[perts]

    lfc, se = read_gw("lfc"), None
    se = read_gw("se").reindex(index=lfc.index)
    sym = lfc.index.map(dict(zip(e2s.ensembl, e2s.symbol)))
    k = pd.notna(sym)
    lfc, se, sym = lfc[k], se[k], pd.Index(sym[k])
    dd = ~sym.duplicated()
    lfc, se, sym = lfc[dd], se[dd], sym[dd]
    common = pd.Index(gate_sym).intersection(sym)
    gi_h1, gi_k5 = pd.Index(gate_sym).get_indexer(common), sym.get_indexer(common)
    G = len(common)
    B = lfc.to_numpy().astype(np.float64)[gi_k5]
    S = se.to_numpy().astype(np.float64)[gi_k5]
    thr = m[gi_h1]
    ok = np.isfinite(thr) & (thr > 0)
    print(f"共同基因 G = {G:,}   扰动 = {len(perts)}")

    # ---- 一次性算出 H1 的 beta 与真集 ----
    t = time.time()
    H1B = np.full((G, len(perts)), np.nan)
    REAL = np.zeros((G, len(perts)), bool)
    NR = np.zeros(len(perts), int)
    for j, name in enumerate(perts):
        idx = np.flatnonzero(codes == cats.index(name))
        if len(idx) > VCC_PERT_CELLS:
            idx = rng.choice(idx, VCC_PERT_CELLS, replace=False)
        cpm = to_cpm(thin(read_rows(idx), VCC_UMI, rng))
        padj, _ = ref.de_table(cpm, tie_correct=True)
        REAL[:, j] = (padj < ALPHA)[gi_h1]
        NR[j] = int(REAL[:, j].sum())
        pm = cpm[:, gidx].mean(0)
        H1B[:, j] = np.log2((pm + EPS) / (ctrl_cpm + EPS))[gi_h1]
    print(f"H1 的 beta 与真集算完 ({time.time()-t:.0f}s)  |R_p| 中位 {np.median(NR):.0f}")

    # ---- 特征（给 G 用）----
    logexpr = np.log10(ctrl_cpm[gi_h1] + 1.0)
    med_se = np.nanmedian(S, axis=1)
    feats = np.column_stack([np.ones(G), logexpr, np.log10(np.maximum(med_se, 1e-6)),
                             logexpr ** 2])

    folds = rng.permutation(len(perts)) % N_FOLD
    recs = []
    print(f"\n{'扰动':>10} {'|R_p|':>6} " + " ".join(f"{LAB[x][:10]:>11}" for x in KEYS))
    for f_ in range(N_FOLD):
        tr, te = np.flatnonzero(folds != f_), np.flatnonzero(folds == f_)
        R = H1B[:, tr] - B[:, tr]                       # 逐基因跨系残差
        with np.errstate(invalid="ignore"):
            sig_gene = np.nanstd(R, axis=1)
        finite = np.isfinite(sig_gene) & (sig_gene > 0)
        # 特征回归版（对 log sigma 做最小二乘）
        coef = np.linalg.lstsq(feats[finite], np.log(sig_gene[finite]), rcond=None)[0]
        sig_feat = np.exp(feats @ coef)
        sg = np.where(finite, sig_gene, np.nanmedian(sig_gene[finite]))

        for j in te:
            n_real = NR[j]
            if n_real == 0:
                continue
            b, s = B[:, j], S[:, j]
            good = np.isfinite(b) & np.isfinite(s) & (s > 0) & ok
            if good.sum() < 500:
                continue
            sc = {}
            sc["D_lfcse"] = np.where(good, np.abs(b) / np.maximum(s, 1e-9), -np.inf)
            sc["F_sigma_gene"] = np.where(
                good, np.abs(b) / np.sqrt(s ** 2 + sg ** 2), -np.inf)
            sc["G_sigma_feat"] = np.where(
                good, np.abs(b) / np.sqrt(s ** 2 + sig_feat ** 2), -np.inf)
            sc["H_sigma_only"] = np.where(
                good, np.abs(b) / np.maximum(sg, 1e-9), -np.inf)
            rec = dict(target_gene=perts[j], n_real=n_real, chance=n_real / G, fold=f_)
            for x in KEYS:
                top = np.argsort(sc[x])[::-1][:n_real]
                rec[f"h_{x}"] = float(REAL[top, j].sum()) / n_real
            recs.append(rec)
            print(f"{perts[j]:>10} {n_real:6d} " +
                  " ".join(f"{rec['h_'+x]:11.3f}" for x in KEYS))

    df = pd.DataFrame(recs)
    df.to_csv(OUT / "result.csv", index=False)
    p_tmp.unlink(missing_ok=True)

    print(f"\n{'='*78}\n结论（n = {len(df)}，5 折扰动级交叉验证）\n{'='*78}")
    print(f"{'统计量':>24} {'h 中位':>8} {'机会校正后':>11} {'vs D':>7}")
    hd = df.h_D_lfcse.median()
    ccd = ((df.h_D_lfcse - df.chance) / (1 - df.chance)).median()
    for x in KEYS:
        v = df[f"h_{x}"]
        cc = ((v - df.chance) / (1 - df.chance)).median()
        print(f"{LAB[x]:>24} {v.median():8.3f} {cc:11.4f} {cc/ccd:6.2f}×")
    print(f"\n--- 配对检验（vs D）---")
    for x in KEYS[1:]:
        d = df[f"h_{x}"] - df.h_D_lfcse
        p = wilcoxon(df[f"h_{x}"], df.h_D_lfcse).pvalue if d.abs().sum() else 1.0
        print(f"  {LAB[x]:>24}: Δh {d.median():+.4f}  胜 {(d>0).sum():2d}/{len(df)}  p={p:.5f}")

    best = max(KEYS, key=lambda x: ((df[f"h_{x}"] - df.chance) / (1 - df.chance)).median())
    hb = df[f"h_{best}"].median()
    print(f"\n最好: {LAB[best]}   h = {hb:.3f}   相对追平线 {hb/H_TIE:.2f}×"
          f"   相对同系上限 {hb/H_CEIL:.0%}")
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
