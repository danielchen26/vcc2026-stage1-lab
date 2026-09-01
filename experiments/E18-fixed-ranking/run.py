"""E18 — 用修正后的排序统计量重测跨系 h：验证 E17 定位的修法。

## E17 定位的缺陷

E17 拆解出（同系、对半劈开、预言机规模）：
    ① |beta|                 h = 0.232
    ② |beta| / MDE           h = 0.212   与 ① 无差异（p = 0.52）**MDE 归一化无用**
    ⑤ |beta| / SE            h = 0.500   2.15×，达官方 z（0.551）的 91%

而 E14 的 TAP 用 `p_exceed(beta, se, MDE, pi0, tau)` 排序，跨系得 h = 0.145。
`p_exceed` 同时用了**有用的 beta/se** 和**无用的 MDE** —— 后者把信号稀释了。

**修法**：跨系排序改用纯 Wald z = |beta| / lfcSE，不掺 MDE。
K562 GWPS 发布的 DESeq2 lfcSE 就是这个 SE，数据已在手。

## 本实验

与 E14 完全同一套设置（源 = K562 GWPS，靶 = H1 官方真集，共同基因 7,016，
47 个扰动，预言机规模 K = |R_p|），只换排序统计量：

    A. p_exceed（E14 的做法，应复现 0.145）
    B. |beta_K562|（裸）
    C. |beta_K562| / lfcSE_K562        ← E17 预测的赢家
    D. |beta_K562| / lfcSE，且用 MDE 做**硬门限**而非连续权重
       （只在 MDE 有限的基因里排序，但不按 MDE 加权）
    E. 1/MDE（零生物学地板）

参照：同系上限 0.500–0.551 · 追平线 0.127 · 零生物学地板 0.022

跑法：  ~/vcc2026/.venv/bin/python experiments/E18-fixed-ranking/run.py
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
from vcclab.callset import fit_prior, p_exceed  # noqa: E402
from vcclab.detectability import mde  # noqa: E402
from vcclab.scorer import ControlRef  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "E10-h1-erp"))
from run import VCC_CTRL_CELLS, VCC_PERT_CELLS, VCC_UMI, read_rows, thin, to_cpm  # noqa: E402

DATA = ROOT / "data"
H5 = DATA / "vcc2025" / "adata_Validation.h5ad"
MAPCSV = Path("/tmp/ens2sym.csv")
OUT = Path(__file__).parent
ALPHA = 0.05
Z_BH = 3.184
SEED = 0
H_FLOOR, H_TIE = 0.022, 0.127
H_CEIL_MEAN, H_CEIL_OFFICIAL = 0.500, 0.551
KEYS = ("p_exceed", "abs_beta", "wald_z", "wald_z_gated", "inv_mde")
LAB = {"p_exceed": "A. p_exceed（E14）", "abs_beta": "B. |beta| 裸",
       "wald_z": "C. |beta|/lfcSE", "wald_z_gated": "D. C + MDE 硬门限",
       "inv_mde": "E. 1/MDE（地板）"}


def main() -> None:
    t0 = time.time()
    print("=== E18 修正排序统计量后的跨系 h ===\n")

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
    m = mde(ref, n_cells=VCC_PERT_CELLS, alpha=float(2 * norm.sf(Z_BH)),
            seed=SEED, tie_correct=True)
    gate_sym = np.array(h1_genes)[np.asarray(ref.gidx)]
    print(f"H1: gate = {ref.G:,}  MDE 中位 = {np.nanmedian(m):.4f}")

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

    lfc = read_gw("lfc")
    se = read_gw("se").reindex(index=lfc.index)
    sym = lfc.index.map(dict(zip(e2s.ensembl, e2s.symbol)))
    keep = pd.notna(sym)
    lfc, se, sym = lfc[keep], se[keep], pd.Index(sym[keep])
    dd = ~sym.duplicated()
    lfc, se, sym = lfc[dd], se[dd], sym[dd]

    common = pd.Index(gate_sym).intersection(sym)
    gi_h1 = pd.Index(gate_sym).get_indexer(common)
    gi_k5 = sym.get_indexer(common)
    G = len(common)
    B = lfc.to_numpy().astype(np.float64)[gi_k5]
    S = se.to_numpy().astype(np.float64)[gi_k5]
    thr = m[gi_h1]
    ok_thr = np.isfinite(thr) & (thr > 0)
    print(f"共同基因 G = {G:,}   扰动 = {len(perts)}\n")

    print(f"{'扰动':>10} {'|R_p|':>6} " + " ".join(f"{LAB[k][:9]:>10}" for k in KEYS))
    recs = []
    for j, name in enumerate(perts):
        ci = cats.index(name)
        idx = np.flatnonzero(codes == ci)
        if len(idx) > VCC_PERT_CELLS:
            idx = rng.choice(idx, VCC_PERT_CELLS, replace=False)
        padj, _ = ref.de_table(to_cpm(thin(read_rows(idx), VCC_UMI, rng)),
                               tie_correct=True)
        real = (padj < ALPHA)[gi_h1]
        n_real = int(real.sum())
        if n_real == 0:
            continue
        b, s = B[:, j], S[:, j]
        good = np.isfinite(b) & np.isfinite(s) & (s > 0)
        if good.sum() < 500:
            continue

        sc = {k: np.full(G, -np.inf) for k in KEYS}
        pi0, tau = fit_prior(b[good], s[good])
        gg = good & ok_thr
        sc["p_exceed"][gg] = p_exceed(b[gg], s[gg], thr[gg], pi0, tau)
        sc["abs_beta"][good] = np.abs(b[good])
        sc["wald_z"][good] = np.abs(b[good]) / s[good]
        sc["wald_z_gated"][gg] = np.abs(b[gg]) / s[gg]
        sc["inv_mde"][ok_thr] = 1.0 / thr[ok_thr]

        rec = dict(target_gene=name, n_real=n_real)
        for k in KEYS:
            top = np.argsort(sc[k])[::-1][:n_real]
            rec[f"h_{k}"] = float(real[top].sum()) / n_real
        rec["chance"] = n_real / G
        recs.append(rec)
        print(f"{name:>10} {n_real:6d} " + " ".join(f"{rec['h_'+k]:10.3f}" for k in KEYS))

    df = pd.DataFrame(recs)
    df.to_csv(OUT / "result.csv", index=False)
    p_tmp.unlink(missing_ok=True)

    print(f"\n{'='*78}\n结论（n = {len(df)}，预言机规模 K = |R_p|）\n{'='*78}")
    print(f"{'排序统计量':>22} {'h 中位':>8} {'IQR':>18} {'机会校正后中位':>14} {'vs A':>7}")
    ha = df.h_p_exceed.median()
    for k in KEYS:
        v = df[f"h_{k}"]
        cc = ((v - df.chance) / (1 - df.chance)).median()
        q = np.percentile(v, [25, 50, 75])
        print(f"{LAB[k]:>22} {q[1]:8.3f} {f'[{q[0]:.3f}, {q[2]:.3f}]':>18} "
              f"{cc:14.4f} {q[1]/ha:6.2f}×")

    print(f"\n--- 配对检验（vs A. p_exceed）---")
    for k in KEYS[1:]:
        d = df[f"h_{k}"] - df.h_p_exceed
        p = wilcoxon(df[f"h_{k}"], df.h_p_exceed).pvalue if d.abs().sum() else 1.0
        print(f"  {LAB[k]:>22}: Δh {d.median():+.4f}  胜 {(d>0).sum():2d}/{len(df)}  p={p:.5f}")

    best = max(KEYS, key=lambda k: df[f"h_{k}"].median())
    hb = df[f"h_{best}"].median()
    print(f"\n{'='*78}")
    print(f"最好的统计量: {LAB[best]}   h = {hb:.3f}")
    print(f"  参照: 地板 {H_FLOOR} · 追平线 {H_TIE} · "
          f"同系均值型上限 {H_CEIL_MEAN} · 同系官方上限 {H_CEIL_OFFICIAL}")
    print(f"  相对追平线 {hb/H_TIE:.2f}×   相对同系均值型上限 {hb/H_CEIL_MEAN:.0%}")
    print(f"  E14 原 p_exceed 为 {ha:.3f} → 提升 {hb/ha:.2f}×")
    print(f"\n>>> {'超过追平线' if hb >= H_TIE else '仍低于追平线'}，"
          f"{'且' if hb >= 2*H_TIE else '但'}余量 {hb/H_TIE:.2f}× <<<")
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
