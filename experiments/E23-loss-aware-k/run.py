"""E23 — 损失感知的 K 策略：不追求预测准，追求期望 jac 最大。

## 为什么换思路

E22 用双源把 K_opt 的预测 Spearman 从 0.545 提到 0.664，但**中位倍数误差仍 8.98×**。
按 9 倍误差选 K 不可能好。

而 E21 的分层暴露了一个决定性的不对称：

    真实 |R_p| > 500 时：少报 0.2057 vs 全报 0.3129  →  少报亏 1.5×
    真实 |R_p| < 50  时：少报 0.1429 vs 全报 0.0006  →  全报亏 240×

**不对称约 150:1。** 所以在不确定下最优的不是「预测准」，而是**损失感知**：
直接在训练扰动上按预测的 n_hat 分箱，学「该箱里使平均 jac 最大的 K」。
这是对目标函数的直接经验风险最小化，天然把不对称算进去，不需要 K 预测准。

## 规则

    S1  全报（baseline）
    S2  E21 的做法：回归预测 K_opt 后直接用
    S3  **分箱查表**：按 n_hat 分箱，每箱取训练折上平均 jac 最大的 K
    S4  分箱查表 + 双源 n_hat（E22 的特征）
    S5  oracle K_opt（上界）

n_hat 来自源侧可得的量（K562 显著数 + CD4T Rest 的 n_total_de_genes），
5 折扰动级交叉验证，训练折内再不看留出折的任何信息。

跑法：  ~/vcc2026/.venv/bin/python experiments/E23-loss-aware-k/run.py
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
CD4 = DATA / "external" / "cd4_de.csv"
OUT = Path(__file__).parent
ALPHA, Z_BH, SEED = 0.05, 3.184, 0
N_FOLD, N_BIN = 5, 4
JAC_BASELINE, JAC_REPLICATE, LEADER = 0.12, 0.379, 0.1899
JAC_TIE = JAC_BASELINE + LEADER * (JAC_REPLICATE - JAC_BASELINE)
RULES = ("S1_all", "S2_regress", "S3_bin1src", "S4_bin2src", "S5_oracle")
LAB = {"S1_all": "S1 全报（baseline）", "S2_regress": "S2 回归预测 K",
       "S3_bin1src": "S3 分箱查表（单源）", "S4_bin2src": "S4 分箱查表（双源）",
       "S5_oracle": "S5 oracle K_opt"}
KGRID = np.unique(np.round(np.geomspace(1, 7000, 60)).astype(int))


def main() -> None:
    t0 = time.time()
    print("=== E23 损失感知的 K 策略 ===")
    print(f"追平线 raw jac = {JAC_TIE:.4f}（看均值）\n")

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
    gwp = DATA / "nadig2025" / "K562GW_p.csv.gz"
    gw_cols = [str(c) for c in pd.read_csv(gwp, index_col=0, nrows=1).columns]
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
    cd = pd.read_csv(CD4)
    r_ = cd[cd.culture_condition == "Rest"].drop_duplicates("target_contrast_gene_name")
    cd_map = dict(zip(r_.target_contrast_gene_name.astype(str), r_.n_total_de_genes))
    n_cd4 = np.array([cd_map.get(p_, np.nan) for p_ in perts], float)
    print(f"G = {G:,}  扰动 {len(perts)}  CD4T 覆盖 {np.isfinite(n_cd4).sum()}")

    t = time.time()
    CURVE = []
    NR = np.zeros(len(perts), int)
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
        order = np.argsort(np.where(good, np.abs(b) / np.maximum(s, 1e-9),
                                    -np.inf))[::-1]
        if NR[j] == 0:
            CURVE.append(None); continue
        hit = np.cumsum(real[order])
        k = np.arange(1, G + 1)
        CURVE.append(hit / (NR[j] + k - hit))
    print(f"jac(K) 曲线算完 ({time.time()-t:.0f}s)  |R_p| 中位 {np.median(NR):.0f}")
    valid = np.flatnonzero(NR > 0)
    KOPT = np.array([int(np.argmax(CURVE[j])) + 1 if CURVE[j] is not None else 1
                     for j in range(len(perts))])
    p_tmp.unlink(missing_ok=True)

    def fit_pred(tr, te, use_cd4):
        def X(ii):
            c = [np.ones(len(ii)), np.log10(n_src[ii] + 1.0)]
            if use_cd4:
                c.append(np.log10(np.nan_to_num(n_cd4[ii], nan=np.nanmedian(n_cd4)) + 1.0))
            return np.column_stack(c)
        co = np.linalg.lstsq(X(tr), np.log10(KOPT[tr].clip(min=1)), rcond=None)[0]
        return 10 ** (X(te) @ co)

    folds = rng.permutation(len(perts)) % N_FOLD
    recs = []
    for f_ in range(N_FOLD):
        tr = np.array([j for j in valid if folds[j] != f_])
        te = np.array([j for j in valid if folds[j] == f_])
        if len(te) == 0:
            continue
        # 分箱：按训练折的 n_hat 分位切 N_BIN 箱，每箱取平均 jac 最大的 K
        for tag, use_cd4 in (("S3_bin1src", False), ("S4_bin2src", True)):
            nh_tr = fit_pred(tr, tr, use_cd4)
            edges = np.quantile(np.log10(nh_tr), np.linspace(0, 1, N_BIN + 1))[1:-1]
            bin_tr = np.digitize(np.log10(nh_tr), edges)
            best_k_bin = {}
            for b_ in range(N_BIN):
                mem = tr[bin_tr == b_]
                if len(mem) == 0:
                    best_k_bin[b_] = G
                    continue
                mean_jac = np.array([np.mean([CURVE[j][k - 1] for j in mem])
                                     for k in KGRID])
                best_k_bin[b_] = int(KGRID[int(np.argmax(mean_jac))])
            nh_te = fit_pred(tr, te, use_cd4)
            bin_te = np.digitize(np.log10(nh_te), edges)
            for i, j in enumerate(te):
                r = next((x for x in recs if x["_j"] == j), None)
                if r is None:
                    r = dict(_j=j, target_gene=perts[j], n_real=int(NR[j]),
                             n_src=int(n_src[j]), n_cd4=float(n_cd4[j]),
                             k_opt=int(KOPT[j]))
                    recs.append(r)
                k = int(np.clip(best_k_bin[int(bin_te[i])], 1, G))
                r[f"K_{tag}"] = k
                r[f"jac_{tag}"] = float(CURVE[j][k - 1])
        # S2 回归预测 K 直接用；S1 全报；S5 oracle
        nh = fit_pred(tr, te, True)
        for i, j in enumerate(te):
            r = next(x for x in recs if x["_j"] == j)
            k2 = int(np.clip(round(nh[i]), 1, G))
            r["K_S2_regress"], r["jac_S2_regress"] = k2, float(CURVE[j][k2 - 1])
            r["K_S1_all"], r["jac_S1_all"] = G, float(CURVE[j][G - 1])
            r["K_S5_oracle"] = int(KOPT[j])
            r["jac_S5_oracle"] = float(CURVE[j][KOPT[j] - 1])

    df = pd.DataFrame(recs).drop(columns=["_j"])
    df.to_csv(OUT / "result.csv", index=False)

    print(f"\n{'='*80}\n结论（n = {len(df)}，5 折扰动级交叉验证）\n{'='*80}")
    print(f"{'规则':>22} {'jac 均值':>10} {'jac 中位':>10} {'K 中位':>9} "
          f"{'vs baseline':>12} {'vs 追平线':>10}")
    for r_ in RULES:
        v = df[f"jac_{r_}"]
        print(f"{LAB[r_]:>22} {v.mean():10.4f} {v.median():10.4f} "
              f"{df[f'K_{r_}'].median():9.0f} {v.mean()/df.jac_S1_all.mean():11.2f}× "
              f"{v.mean()/JAC_TIE:9.2f}×")

    print(f"\n--- 配对检验（vs S1 全报 = baseline）---")
    for r_ in RULES[1:]:
        d = df[f"jac_{r_}"] - df.jac_S1_all
        p = wilcoxon(df[f"jac_{r_}"], df.jac_S1_all).pvalue if d.abs().sum() else 1.0
        print(f"  {LAB[r_]:>22}: Δ均值 {d.mean():+.4f}  Δ中位 {d.median():+.4f}  "
              f"胜 {(d>0).sum():2d}/{len(df)}  p={p:.5f}")

    best = max(RULES[1:-1], key=lambda r: df[f"jac_{r}"].mean())
    mb, m1 = df[f"jac_{best}"].mean(), df.jac_S1_all.mean()
    p = wilcoxon(df[f"jac_{best}"], df.jac_S1_all).pvalue
    win = mb > m1 and p < 0.05
    print(f"\n{'='*80}")
    print(f"最好（非 oracle）: {LAB[best]}   jac 均值 {mb:.4f} vs baseline {m1:.4f}"
          f"  ({mb/m1:.2f}×, p={p:.5f})")
    print(f"  相对追平线 {JAC_TIE:.4f} → {mb/JAC_TIE:.2f}×")
    print(f"\n>>> {'有好结果：显著超过 baseline' if win else '仍未显著超过 baseline'} <<<")
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
