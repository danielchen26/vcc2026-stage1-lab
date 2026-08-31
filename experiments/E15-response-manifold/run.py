"""E15 — 检验重构框架的核心前提：扰动响应是否落在对照群体的自然变异张成的空间里。

## 为什么要重构

E14 实测出三件事，并排看指向同一个结构：
  1. |R_p| 最大到 5,398 / 10,000 基因 —— 半个转录组「显著变化」
  2. 官方检验是 400 vs 18,400 细胞的 Wilcoxon，400→1090 细胞使计数涨 4.0 倍
  3. 高 |R_p| 层，只按可检测性排序（+0.0655）比跨系迁移（+0.0411）更好

半个转录组显著不可能是「这些基因被特异性调控」，而是**细胞整体状态位移**。
那么越阈基因的身份由「位移方向 × 基因载荷 × 基因阈值」决定，
**不由扰动的特异性靶点决定**。载荷是细胞系特异的，所以迁移基因级效应量在这一层没用
—— 这正是 E14 测到的。

## 重构

    beta[g,p,T] = sum_k a[k,p] * v[k,g,T] + eps[g,p]
                  ^^^^^^^^^^   ^^^^^^^^^^
                  可迁移的振幅   靶侧可估的载荷

TAP 迁移的是**乘积**（10,000 个基因级效应量，跨系相关仅 0.339）。
应该只迁移真正保守的因子（约 20 个振幅标量），载荷用靶侧自己的对照数据估。

## 本实验只验一件事（前提，不是收益）

**扰动响应 beta 是否落在对照群体自然变异张成的低维空间里？**

若是，则 v_k 可以从每个 context 的 18,400 个对照细胞精确估出，完全不需要迁移。
若不是，整个重构无效，必须另想。

做法（只用 H1，无需新数据）：
  1. 对 18,400 个 NTC 细胞做 PCA（log1p CPM，gate 内基因）→ 载荷 V (基因 × k)
  2. 每个扰动算 pseudobulk beta = log2(mean CPM 扰动 / mean CPM 对照)
  3. 把 beta 投影到 V 张成的空间：a = pinv(V) beta，beta_hat = V a
     报 R^2 = 1 - ||beta - beta_hat||^2 / ||beta||^2 随 k 的曲线
  4. 对照：用**随机正交基**做同样的投影 —— 任何 k 维子空间都能解释 k/G 的方差，
     必须扣掉这个平凡部分，否则 R^2 是自欺

  同时报「响应方向与前几个 PC 的夹角余弦」，看是否集中在少数轴上。

跑法：  ~/vcc2026/.venv/bin/python experiments/E15-response-manifold/run.py
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
from sklearn.utils.extmath import randomized_svd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vcclab.scorer import ControlRef  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "E10-h1-erp"))
from run import VCC_CTRL_CELLS, VCC_PERT_CELLS, VCC_UMI, read_rows, thin  # noqa: E402

H5 = ROOT / "data" / "vcc2025" / "adata_Validation.h5ad"
OUT = Path(__file__).parent
KS = (1, 2, 5, 10, 20, 50, 100, 200)
N_PC = 200
SEED = 0
EPS = 1.0


def logcpm(M: sp.csr_matrix, gidx: np.ndarray) -> np.ndarray:
    """行归一到 1e6 后 log1p，只取 gate 内基因。返回稠密 (细胞 × gate)。"""
    A = np.asarray(M[:, gidx].todense(), dtype=np.float32)
    s = np.asarray(M.sum(1), dtype=np.float32)
    np.divide(A, np.maximum(s, 1.0), out=A)
    A *= 1e6
    return np.log1p(A, out=A)


def main() -> None:
    t0 = time.time()
    print("=== E15 扰动响应是否落在对照的自然变异空间里 ===\n")

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
    G = len(gidx)
    print(f"gate = {G:,} 基因   对照 = {VCC_CTRL_CELLS:,} 细胞")

    t = time.time()
    C = logcpm(ctrl_raw, gidx)
    mu = C.mean(0)
    C -= mu
    print(f"对照 log1pCPM 中心化完成 ({time.time()-t:.0f}s)  形状 {C.shape}")

    t = time.time()
    _, S, Vt = randomized_svd(C, n_components=N_PC, random_state=SEED)
    var_frac = S ** 2 / (C ** 2).sum()
    print(f"PCA {N_PC} 个成分 ({time.time()-t:.0f}s)   "
          f"累计解释对照方差: k=10 {var_frac[:10].sum():.1%} · "
          f"k=50 {var_frac[:50].sum():.1%} · k=200 {var_frac.sum():.1%}")
    del C
    V = Vt.T                                    # (G × N_PC)，列正交

    # 随机正交基对照：任何 k 维子空间平凡地解释 k/G
    Q = np.linalg.qr(rng.standard_normal((G, N_PC)).astype(np.float32))[0]

    ctrl_mean_cpm = np.asarray(ctrl_raw[:, gidx].multiply(
        1e6 / np.maximum(np.asarray(ctrl_raw.sum(1)), 1.0)).mean(0)).ravel()

    print(f"\n{'扰动':>10} {'|beta|中位':>10} " +
          " ".join(f"R²k={k:<4d}" for k in KS) + "   随机k=20")
    recs = []
    for ci, name in enumerate(cats):
        if ci == ntc:
            continue
        idx = np.flatnonzero(codes == ci)
        if len(idx) > VCC_PERT_CELLS:
            idx = rng.choice(idx, VCC_PERT_CELLS, replace=False)
        M = thin(read_rows(idx), VCC_UMI, rng)
        pm = np.asarray(M[:, gidx].multiply(
            1e6 / np.maximum(np.asarray(M.sum(1)), 1.0)).mean(0)).ravel()
        beta = np.log2((pm + EPS) / (ctrl_mean_cpm + EPS)).astype(np.float32)
        tot = float(beta @ beta)
        if tot <= 0:
            continue
        r2 = {}
        for k in KS:
            a = V[:, :k].T @ beta
            r2[k] = float(a @ a) / tot          # 正交基 → 直接是投影能量占比
        aq = Q[:, :20].T @ beta
        r2_rand = float(aq @ aq) / tot
        recs.append(dict(target_gene=name, n_cells=len(idx),
                         beta_l2=tot ** 0.5, med_abs_beta=float(np.median(np.abs(beta))),
                         **{f"r2_k{k}": r2[k] for k in KS}, r2_rand20=r2_rand))
        print(f"{name:>10} {np.median(np.abs(beta)):10.4f} " +
              " ".join(f"{r2[k]:7.3f}" for k in KS) + f"   {r2_rand:8.4f}")

    df = pd.DataFrame(recs)
    df.to_csv(OUT / "result.csv", index=False)
    p_tmp.unlink(missing_ok=True)

    print(f"\n{'='*78}\n结论（n = {len(df)} 个扰动，gate {G:,}）\n{'='*78}")
    print(f"{'k':>6} {'R² 中位':>9} {'R² 均值':>9} {'随机基线 k/G':>13} {'超出倍数':>9}")
    for k in KS:
        v = df[f"r2_k{k}"]
        triv = k / G
        print(f"{k:6d} {v.median():9.3f} {v.mean():9.3f} {triv:13.5f} "
              f"{v.median()/triv:9.1f}×")
    r20 = df.r2_k20.median()
    print(f"\n  随机 20 维正交基实测 R² 中位 = {df.r2_rand20.median():.5f}"
          f"  （理论 20/{G} = {20/G:.5f}）✓")
    print(f"  对照 PCA 前 20 维 R² 中位 = {r20:.3f}"
          f"  →  是随机的 {r20/df.r2_rand20.median():.0f} 倍")

    verdict = ("前提成立：响应确实落在对照自然变异的低维空间里"
               if r20 >= 0.5 else
               "前提部分成立，需要更多维" if r20 >= 0.25 else
               "**前提不成立**：响应不在对照变异空间内，重构无效")
    print(f"\n>>> {verdict} <<<")
    print(f"\n  含义：若成立，则每个 context 的载荷 v_k 可从其 18,400 个对照细胞精确估出，")
    print(f"  跨系只需迁移 ~20 个振幅标量，而不是 {G:,} 个基因级效应量。")
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
