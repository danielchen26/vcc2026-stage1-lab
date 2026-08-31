"""E14 — 直接实测跨系 h：用 K562 的效应量预测 H1 的官方真集。

## 这是本项目唯一一次真正测出 h

此前所有 h 的数字都是锚点或地板/天花板：
    0.022  零生物学地板（E13 实测）
    0.127  追平榜首（E11 实测）
    0.550  同系重做天花板（E11 实测）
**但「跨细胞系预测能拿到多少 h」从未测过。** E12 试图从 LFC 相关系数折算，失败了
（假设 beta 与 MDE 独立，被 E13 实测的 rho=+0.148 否证，模拟高估 23 倍）。

现在可以直测，因为手里的数据刚好凑齐：
    靶：H1 validation 50 个扰动的**细胞级**数据 → 用官方机器算真集 R_p
    源：K562 GenomeWide 的 DESeq2 beta + lfcSE，**47/50 覆盖**
    桥：GTEx v10 的 Ensembl→symbol 映射（8,236/8,248 = 99.9%），共同基因 7,481

这是**真正的零样本跨细胞系预测**：源侧是 K562，靶侧是 H1，两者除了扰动基因名
之外没有任何共享信息，且靶侧用的是官方打分机器。

## TAP 在这里的具体形态

    P(g in R_p^H1) = p_exceed(beta_K562, se_K562, MDE_H1)
    召集 = 按该概率降序取前 K 个
    h = |R_p^H1 ∩ 召集| / |R_p^H1|

阈值 MDE_H1 完全由 H1 的对照细胞算出（TAP 的核心：阈值在目标侧精确已知）；
效应量从 K562 迁移。两侧唯一的接口就是基因名。

两种 K 都报，把「排序质量」和「定量质量」分开：
    K = |R_p|      预言机规模 —— 隔离出排序本身有多好
    K = best_k     实际会用的规模 —— 但 p_exceed 只含源侧测量误差、
                   不含跨系方差，所以会过度自信，K 偏大

三条对照：零生物学（只按 MDE 排序）· 随机 · 同系天花板 0.550。

跑法：  ~/vcc2026/.venv/bin/python experiments/E14-crossline-h/run.py
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
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vcclab.callset import best_k, fit_prior, identifiability, p_exceed  # noqa: E402
from vcclab.detectability import mde  # noqa: E402
from vcclab.scorer import ControlRef  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "E10-h1-erp"))
from run import VCC_CTRL_CELLS, VCC_PERT_CELLS, VCC_UMI, read_rows, thin, to_cpm  # noqa: E402

DATA = ROOT / "data"
H5 = DATA / "vcc2025" / "adata_Validation.h5ad"
MAP = Path("/tmp/ens2sym.csv")
OUT = Path(__file__).parent
ALPHA = 0.05
Z_BH = 3.184
SEED = 0
H_FLOOR, H_TIE, H_CEIL = 0.022, 0.127, 0.550


def main() -> None:
    t0 = time.time()
    print("=== E14 直接实测跨系 h（K562 → H1，官方打分机器）===\n")

    with h5py.File(H5, "r") as f:
        h1_genes = [g.decode() if isinstance(g, bytes) else str(g)
                    for g in f["var"]["_index"][:]]
        o = f["obs"]["target_gene"]
        cats = [c.decode() if isinstance(c, bytes) else str(c)
                for c in o["categories"][:]]
        codes = o["codes"][:]
    ntc = cats.index("non-targeting")
    rng = np.random.default_rng(SEED)

    # ---- 靶侧：H1 的对照参考与逐基因阈值 ----
    rows = rng.choice(np.flatnonzero(codes == ntc), VCC_CTRL_CELLS, replace=False)
    p_tmp = OUT / "_ntc.h5ad"
    ad.AnnData(X=sp.csr_matrix(thin(read_rows(rows), VCC_UMI, rng)),
               var=pd.DataFrame(index=pd.Index(h1_genes))).write_h5ad(p_tmp)
    ref = ControlRef.load(p_tmp, h1_genes)
    m = mde(ref, n_cells=VCC_PERT_CELLS, alpha=float(2 * norm.sf(Z_BH)),
            seed=SEED, tie_correct=True)
    gate_sym = np.array(h1_genes)[np.asarray(ref.gidx)]
    print(f"H1: gate = {ref.G:,}  对照 = {ref.n_ctrl:,}  MDE 中位 = {np.nanmedian(m):.4f}")

    # ---- 源侧：K562 GWPS，Ensembl→symbol ----
    e2s = pd.read_csv(MAP).dropna()
    gw_cols = [str(c) for c in pd.read_csv(DATA / "nadig2025" / "K562GW_p.csv.gz",
                                          index_col=0, nrows=1).columns]
    perts = sorted(set(gw_cols) & (set(cats) - {"non-targeting"}))
    print(f"可测扰动 = {len(perts)} / {len(cats)-1}")

    def read_gw(kind: str) -> pd.DataFrame:
        f = {"lfc": "K562GW_lfc", "se": "K562GW_se"}[kind]
        d = pd.read_csv(DATA / "nadig2025" / f"{f}.csv.gz", index_col=0,
                        usecols=["Unnamed: 0"] + perts,
                        dtype={c: np.float32 for c in perts}, engine="c")
        d.index = d.index.astype(str)
        return d[perts]                      # 必须重排（E07 踩过的坑）

    t = time.time()
    lfc = read_gw("lfc")
    se = read_gw("se").reindex(index=lfc.index)
    sym = lfc.index.map(dict(zip(e2s.ensembl, e2s.symbol)))
    keep = pd.notna(sym)
    lfc, se = lfc[keep], se[keep]
    sym = pd.Index(sym[keep])
    dedup = ~sym.duplicated()
    lfc, se, sym = lfc[dedup], se[dedup], sym[dedup]
    print(f"K562: {lfc.shape[0]:,} 基因映射成 symbol  ({time.time()-t:.0f}s)")

    # ---- 共同基因（H1 gate ∩ K562 已映射）----
    common = pd.Index(gate_sym).intersection(sym)
    gi_h1 = pd.Index(gate_sym).get_indexer(common)      # H1 gate 内下标
    gi_k5 = sym.get_indexer(common)                     # K562 行下标
    G = len(common)
    print(f"共同基因 G = {G:,}\n")

    B = lfc.to_numpy().astype(np.float64)[gi_k5]
    S = se.to_numpy().astype(np.float64)[gi_k5]
    thr = m[gi_h1]
    ok_thr = np.isfinite(thr)
    print(f"{'扰动':>10} {'|R_p|':>6} {'h(K=|R_p|)':>11} {'h(best_k)':>10} "
          f"{'K':>6} {'h_零生物':>9} {'h_随机':>8} {'tau/se':>7}")

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
        good = np.isfinite(b) & np.isfinite(s) & (s > 0) & ok_thr
        if good.sum() < 500:
            continue
        pi0, tau = fit_prior(b[good], s[good])
        pg = np.zeros(G)
        pg[good] = p_exceed(b[good], s[good], thr[good], pi0, tau)

        order = np.argsort(pg)[::-1]
        h_eq = float(real[order[:n_real]].sum()) / n_real
        k_bk, _ = best_k(pg, float(n_real))
        h_bk = float(real[order[:k_bk]].sum()) / n_real
        # 对照
        zo = np.argsort(np.where(ok_thr, thr, np.inf))
        h_zero = float(real[zo[:n_real]].sum()) / n_real
        h_rand = float(real[rng.permutation(G)[:n_real]].sum()) / n_real
        ident = identifiability(s[good], tau)

        recs.append(dict(target_gene=name, n_real=n_real, h_eq=h_eq, h_bk=h_bk,
                         k_bk=k_bk, h_zero=h_zero, h_rand=h_rand,
                         tau_over_se=ident["tau_over_se"], pi0=pi0, tau=tau))
        print(f"{name:>10} {n_real:6d} {h_eq:11.3f} {h_bk:10.3f} {k_bk:6d} "
              f"{h_zero:9.3f} {h_rand:8.3f} {ident['tau_over_se']:7.2f}")

    df = pd.DataFrame(recs)
    df.to_csv(OUT / "result.csv", index=False)
    p_tmp.unlink(missing_ok=True)

    print(f"\n{'='*72}\n结论（n = {len(df)} 个扰动，共同基因 {G:,}）\n{'='*72}")
    print(f"{'方案':>26} {'h 中位':>8} {'IQR':>18} {'均值':>8}")
    for col, lab in (("h_eq", "TAP 跨系（K=|R_p|）"), ("h_bk", "TAP 跨系（best_k）"),
                     ("h_zero", "零生物学（只按 MDE）"), ("h_rand", "随机")):
        v = df[col].dropna()
        q = np.percentile(v, [25, 50, 75])
        print(f"{lab:>26} {q[1]:8.3f} {f'[{q[0]:.3f}, {q[2]:.3f}]':>18} {v.mean():8.3f}")

    he = float(df.h_eq.median())
    print(f"\n三个实测参照：地板 {H_FLOOR}（E13）· 追平线 {H_TIE}（E11）· "
          f"天花板 {H_CEIL}（E11）")
    print(f"\nTAP 跨系实测 h = {he:.3f}")
    print(f"  相对地板   {he/H_FLOOR:6.1f}×")
    print(f"  相对追平线 {he/H_TIE:6.2f}×")
    print(f"  相对天花板 {he/H_CEIL:6.2f}×  （即恢复了同系重做的 {he/H_CEIL:.0%}）")
    verdict = ("超过追平线 —— 方向成立" if he >= H_TIE
               else "低于追平线 —— 单靠 K562 单源不够")
    print(f"\n>>> {verdict} <<<")
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
