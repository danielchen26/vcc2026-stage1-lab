"""E17 — 把 2.37 倍拆开：leverage 来自秩变换，还是逐基因方差归一？

## 为什么这个拆解决定一切

E16 测得（同系、对半劈开、预言机规模）：
    ① |beta|（裸对数倍数变化）    h = 0.232
    ② |beta| / MDE               h = 0.212   与 ① 无差异（p = 0.52）
    ③ 官方 p_adj                 h = 0.551   25/25 全胜，2.37×

官方统计量是 z = (U - n1 n2 / 2) / sd，其中
    U  = sum_i psi_g(v_i)     秩和（我们 400 个细胞在对照 ECDF 上的位置）
    sd = 含并列校正的秩和标准差（逐基因不同）

**2.37 倍可能来自两个完全不同的地方，含义相反：**

  (A) **秩变换**本身 —— 若是，则必须拿到源侧的**细胞级**数据才能迁移，
      而 K562 只发布了 DESeq2 汇总统计 → 当前路线封顶。

  (B) **逐基因方差归一** —— 若是，则 DESeq2 的 beta/lfcSE **已经是**这个东西，
      TAP 的 p_exceed 用的就是它 → 损失纯粹来自跨系，路线不封顶。

## 本实验的统计量阶梯

全部只用半 A 构造，靶 = 半 B 独立算出的官方 R_p，K = |R_p(B)|：

    ① |beta|                      裸对数倍数变化
    ② |beta| / MDE                按可检出阈值归一
    ③ |U - n1 n2 / 2|             **秩位移，不做 sd 归一**  ← 拆解关键
    ④ |U - n1 n2 / 2| / sd        官方 z（应复现 0.551）
    ⑤ |beta| / SE_pooled          **最好的均值型统计量**（Welch t）  ← 拆解关键
    ⑥ 1/MDE                       零生物学地板

判读：
    ③ ≈ ④ ≈ 0.55  →  秩变换是关键，sd 归一无所谓
    ③ ≈ 0.23、④ ≈ 0.55  →  sd 归一是关键
    ⑤ ≈ 0.55  →  **均值型路线不封顶**，DESeq2 的 beta/lfcSE 够用，损失纯在跨系
    ⑤ ≈ 0.23  →  均值型路线结构性封顶，必须拿细胞级源数据

跑法：  ~/vcc2026/.venv/bin/python experiments/E17-statistic-decomposition/run.py
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

H5 = ROOT / "data" / "vcc2025" / "adata_Validation.h5ad"
OUT = Path(__file__).parent
ALPHA = 0.05
Z_BH = 3.184
SEED = 0
EPS = 1.0
KEYS = ("abs_beta", "beta_over_mde", "rank_raw", "official_z", "beta_over_se", "inv_mde")
LAB = {"abs_beta": "① |beta|", "beta_over_mde": "② |beta|/MDE",
       "rank_raw": "③ 秩位移（无 sd）", "official_z": "④ 官方 z",
       "beta_over_se": "⑤ |beta|/SE（均值型上限）", "inv_mde": "⑥ 1/MDE（地板）"}


def main() -> None:
    t0 = time.time()
    print("=== E17 把 2.37 倍拆成「秩变换」与「方差归一」===\n")

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
    ctrl_raw = thin(read_rows(rows), VCC_UMI, rng)
    p_tmp = OUT / "_ntc.h5ad"
    ad.AnnData(X=sp.csr_matrix(ctrl_raw),
               var=pd.DataFrame(index=pd.Index(genes))).write_h5ad(p_tmp)
    ref = ControlRef.load(p_tmp, genes)
    gidx = np.asarray(ref.gidx)
    G, n2 = len(gidx), ref.n_ctrl
    m = mde(ref, n_cells=VCC_PERT_CELLS, alpha=float(2 * norm.sf(Z_BH)),
            seed=SEED, tie_correct=True)
    m_safe = np.where(np.isfinite(m) & (m > 0), m, np.inf)

    # 对照侧：CPM 均值（给 beta 用）与 log2(CPM+1) 的均值/方差（给 SE 用）
    inv = 1e6 / np.maximum(np.asarray(ctrl_raw.sum(1)), 1.0)
    Cc = np.asarray(ctrl_raw[:, gidx].multiply(inv).todense(), dtype=np.float32)
    ctrl_cpm = Cc.mean(0)
    Cl = np.log2(Cc + 1.0, out=Cc)
    ctrl_lmu, ctrl_lvar = Cl.mean(0), Cl.var(0, ddof=1)
    del Cc, Cl
    print(f"gate = {G:,}  对照 = {n2:,}  MDE 中位 {np.nanmedian(m):.4f}")

    def stats_from(sel: np.ndarray) -> dict:
        M = thin(read_rows(sel), VCC_UMI, rng)
        cpm_full = to_cpm(M)
        n1 = cpm_full.shape[0]
        sub = cpm_full[:, gidx]
        pm = sub.mean(0)
        beta = np.log2((pm + EPS) / (ctrl_cpm + EPS))
        L = np.log2(sub + 1.0)
        lmu, lvar = L.mean(0), L.var(0, ddof=1)
        se_pool = np.sqrt(np.maximum(lvar / n1 + ctrl_lvar / n2, 1e-12))
        U = np.empty(G)
        T = np.empty(G)
        for j in range(G):
            v = cpm_full[:, gidx[j]].astype(np.float64)
            U[j] = ref.psi(j, v).sum()
            allv = np.concatenate([np.zeros(ref._nzero[j], np.float32),
                                   ref._sorted[j], v.astype(np.float32)])
            _, c = np.unique(allv, return_counts=True)
            T[j] = np.sum(c.astype(np.float64) ** 3 - c)
        N = n1 + n2
        sd = np.sqrt(n1 * n2 / 12.0 * ((N + 1) - T / (N * (N - 1.0))))
        dev = np.abs(U - n1 * n2 / 2.0)
        return {"abs_beta": np.abs(beta), "beta_over_mde": np.abs(beta) / m_safe,
                "rank_raw": dev, "official_z": dev / sd,
                "beta_over_se": np.abs(lmu - ctrl_lmu) / se_pool,
                "inv_mde": 1.0 / m_safe, "_cpm": cpm_full}

    need = 2 * VCC_PERT_CELLS
    print(f"\n{'扰动':>10} {'|R_p(B)|':>9} " + " ".join(f"{LAB[k][:9]:>10}" for k in KEYS))
    recs = []
    for ci, name in enumerate(cats):
        if ci == ntc:
            continue
        idx = np.flatnonzero(codes == ci)
        if len(idx) < need:
            continue
        pick = rng.permutation(idx)[:need]
        sa = stats_from(pick[:VCC_PERT_CELLS])
        padj_b, _ = ref.de_table(to_cpm(thin(read_rows(pick[VCC_PERT_CELLS:]),
                                             VCC_UMI, rng)), tie_correct=True)
        real = padj_b < ALPHA
        n_real = int(real.sum())
        if n_real < 5:
            continue
        rec = dict(target_gene=name, n_real=n_real)
        for k in KEYS:
            top = np.argsort(sa[k])[::-1][:n_real]
            rec[f"h_{k}"] = float(real[top].sum()) / n_real
        recs.append(rec)
        print(f"{name:>10} {n_real:9d} " + " ".join(f"{rec['h_'+k]:10.3f}" for k in KEYS))

    df = pd.DataFrame(recs)
    df.to_csv(OUT / "result.csv", index=False)
    p_tmp.unlink(missing_ok=True)

    print(f"\n{'='*76}\n结论（n = {len(df)}，预言机规模 K = |R_p(B)|）\n{'='*76}")
    print(f"{'统计量':>28} {'h 中位':>9} {'IQR':>18} {'相对 ①':>9}")
    base = df.h_abs_beta.median()
    for k in KEYS:
        v = df[f"h_{k}"]
        q = np.percentile(v, [25, 50, 75])
        print(f"{LAB[k]:>28} {q[1]:9.3f} {f'[{q[0]:.3f}, {q[2]:.3f}]':>18} {q[1]/base:8.2f}×")

    print(f"\n--- 配对检验（vs ① |beta|）---")
    for k in KEYS[1:]:
        d = df[f"h_{k}"] - df.h_abs_beta
        p = wilcoxon(df[f"h_{k}"], df.h_abs_beta).pvalue if d.abs().sum() else 1.0
        print(f"  {LAB[k]:>28}: Δh {d.median():+.4f}  胜 {(d>0).sum():2d}/{len(df)}  p={p:.5f}")

    h1, h3, h4, h5 = (df.h_abs_beta.median(), df.h_rank_raw.median(),
                      df.h_official_z.median(), df.h_beta_over_se.median())
    print(f"\n{'='*76}\n拆解\n{'='*76}")
    print(f"  ① 裸 |beta|            {h1:.3f}")
    print(f"  ③ 秩位移（无 sd 归一）   {h3:.3f}   → 秩变换单独贡献 {h3/h1:.2f}×")
    print(f"  ④ 官方 z               {h4:.3f}   → 再加 sd 归一 {h4/h3:.2f}×")
    print(f"  ⑤ |beta|/SE（均值型）   {h5:.3f}   → 均值型 + 方差归一 {h5/h1:.2f}×")
    if h5 >= 0.8 * h4:
        print(f"\n>>> **均值型路线不封顶**：|beta|/SE 达到官方 z 的 {h5/h4:.0%} <<<")
        print(f"    DESeq2 的 beta/lfcSE 就是这个统计量 → 损失纯粹来自跨系，")
        print(f"    不需要源侧细胞级数据。修 TAP 的排序统计量即可。")
    else:
        print(f"\n>>> **均值型路线结构性封顶**：|beta|/SE 只到官方 z 的 {h5/h4:.0%} <<<")
        print(f"    秩位移里有均值统计量抓不到的信息 → 必须拿到源侧**细胞级**数据。")
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
