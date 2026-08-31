"""E13 — 实测「零生物学」基线：只按可检测性排序能拿到多少 h。

## 为什么必须实测

E12 试图把「跨系 LFC 相关 rho」转换成「集合重叠 h」，但模拟在 rho = 0 处
给出 h = 0.506 —— 源侧零信息却拿到一半重叠，荒谬。

根因：E12 假设 beta 与 MDE 独立。于是按 |beta_S|/MDE 排序即使在 beta_S 纯噪声时
也会偏向低 MDE 基因，而低 MDE 基因恰是最可能越阈的，白拿大量重叠。
真实数据里**效应量与表达量相关**（低表达基因 MDE 高，|LFC| 也大），
这个白拿会被大幅抵消。抵消多少，**只能测，不能假设**。

这个数同时是所有 go/no-go 判断的地板：任何方法都必须显著超过它才算有生物学价值。
先前记录的「零生物学上限 h <= 0.130」是理论扫描得来的，现在用主办方自己的数据实测。

## 方法

在 2025 H1、降采样到本届确切条件下，对每个扰动：
  1. 真集 R_p = de_table 判定的显著基因（官方机器）
  2. 零生物学召集：**只按 MDE 升序**取 top-K（不看任何扰动信息）
  3. h_zero = |R_p ∩ 召集| / |R_p|

三种 K 的取法都报，避免 K 的选择掩盖结论：
  K = |R_p|（理想情况，等规模）· K = 253（E10 实测中位）· K 由 best_k 定

同时测 |LFC| 与 MDE 的实际关系 —— 这正是 E12 缺的那块。

跑法：  ~/vcc2026/.venv/bin/python experiments/E13-zero-biology/run.py
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
from scipy.stats import norm, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vcclab.detectability import mde  # noqa: E402
from vcclab.scorer import ControlRef  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "E10-h1-erp"))
from run import VCC_CTRL_CELLS, VCC_PERT_CELLS, VCC_UMI, read_rows, thin, to_cpm  # noqa: E402

H5 = ROOT / "data" / "vcc2025" / "adata_Validation.h5ad"
OUT = Path(__file__).parent
ALPHA = 0.05
Z_BH = 3.184
SEED = 0
H_TIE = 0.127            # E11 实测
PRIOR_CEILING = 0.130    # 先前理论扫描记录的零生物学上限


def main() -> None:
    t0 = time.time()
    print("=== E13 实测零生物学基线 ===")
    print(f"先前理论扫描记录的上限 h <= {PRIOR_CEILING}；追平线 h = {H_TIE}（E11 实测）\n")

    with h5py.File(H5, "r") as f:
        genes = [g.decode() if isinstance(g, bytes) else str(g)
                 for g in f["var"]["_index"][:]]
        o = f["obs"]["target_gene"]
        cats = [c.decode() if isinstance(c, bytes) else str(c)
                for c in o["categories"][:]]
        codes = o["codes"][:]
    ntc = cats.index("non-targeting")
    rng = np.random.default_rng(SEED)

    rows = rng.choice(np.flatnonzero(codes == ntc), VCC_CTRL_CELLS, replace=False)
    ctrl = thin(read_rows(rows), VCC_UMI, rng)
    p = OUT / "_ntc.h5ad"
    ad.AnnData(X=sp.csr_matrix(ctrl),
               var=pd.DataFrame(index=pd.Index(genes))).write_h5ad(p)
    ref = ControlRef.load(p, genes)
    print(f"ControlRef: gate = {ref.G:,}  对照 = {ref.n_ctrl:,}")

    t = time.time()
    m = mde(ref, n_cells=VCC_PERT_CELLS, alpha=float(2 * norm.sf(Z_BH)),
            seed=SEED, tie_correct=True)
    finite = np.isfinite(m)
    print(f"MDE: 可算 {finite.sum():,}/{ref.G:,}  中位 {np.nanmedian(m):.4f}  "
          f"({time.time()-t:.0f}s)")
    # 零生物学排序：MDE 升序（不可算的排最后）
    key = np.where(finite, m, np.inf)
    zero_rank = np.argsort(key)

    recs = []
    print(f"\n{'扰动':>12} {'|R_p|':>7} {'h(K=|R_p|)':>11} {'h(K=253)':>9} "
          f"{'rho(|lfc|,MDE)':>15}")
    for ci, name in enumerate(cats):
        if ci == ntc:
            continue
        idx = np.flatnonzero(codes == ci)
        if len(idx) > VCC_PERT_CELLS:
            idx = rng.choice(idx, VCC_PERT_CELLS, replace=False)
        M = thin(read_rows(idx), VCC_UMI, rng)
        padj, lfc = ref.de_table(to_cpm(M), tie_correct=True)
        real = padj < ALPHA
        n_real = int(real.sum())
        if n_real == 0:
            continue
        h_eq = float(real[zero_rank[:n_real]].sum()) / n_real
        h_253 = float(real[zero_rank[:253]].sum()) / n_real
        ok = finite & np.isfinite(lfc)
        rho = spearmanr(np.abs(lfc[ok]), m[ok]).statistic
        recs.append(dict(target_gene=name, n_real=n_real, h_eq=h_eq, h_253=h_253,
                         rho_lfc_mde=rho))
        print(f"{name:>12} {n_real:7d} {h_eq:11.3f} {h_253:9.3f} {rho:15.3f}")

    df = pd.DataFrame(recs)
    df.to_csv(OUT / "result.csv", index=False)
    p.unlink(missing_ok=True)

    print(f"\n{'='*66}\n结论（n = {len(df)} 个扰动）\n{'='*66}")
    for col, lab in (("h_eq", "K = |R_p|（等规模，理想）"), ("h_253", "K = 253")):
        v = df[col].dropna()
        q = np.percentile(v, [25, 50, 75])
        print(f"  {lab:24s} h 中位 {q[1]:.3f}  IQR [{q[0]:.3f}, {q[2]:.3f}]  "
              f"最大 {v.max():.3f}")
    hm = float(df.h_eq.median())
    print(f"\n  |LFC| 与 MDE 的 Spearman rho: 中位 {df.rho_lfc_mde.median():.3f}"
          f"  （E12 假设为 0）")
    print(f"\n  先前理论扫描记录的上限 {PRIOR_CEILING} vs 实测 {hm:.3f}")
    if hm > PRIOR_CEILING * 1.5:
        print(f"  → **理论扫描低估了零生物学基线 {hm/PRIOR_CEILING:.1f} 倍**")
    print(f"\n  追平线 h = {H_TIE}")
    if hm >= H_TIE:
        print(f"  >>> 零生物学基线本身就 >= 追平线 <<<")
        print(f"      含义：榜首分数可能主要来自可检测性结构，而非生物学。")
        print(f"      任何方法必须显著超过 {hm:.3f} 才算有真实预测价值。")
    else:
        print(f"  >>> 零生物学基线 {hm:.3f} < 追平线 {H_TIE}，需要真实生物学信号 <<<")
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
