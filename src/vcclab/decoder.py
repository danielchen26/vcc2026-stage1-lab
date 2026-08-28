"""Stage 2 构造器: 把「响应基因集 + 目标 lfc」无损翻译成 400 个整数计数细胞.

从 `~/vcc2026/vcc_local.py` 的 `ControlRef.design` / `hamilton` 原样移植,
**数值行为不变**, 只把方法改成以 `ref` 为第一参的函数.

已验证 (context_A, 250 个意图显著基因):
  实际 250 个显著, 召回 100%, 精确率 100%, 假阳性 0, 方向一致 100%, 0.28 s/组.

易错点 (错一个就静默失分):
  1. CPM 是成分数据: 目标 profile 的列均值之和必须重归一到 1e6, 否则每细胞行和
     无解 (会抛 ValueError).
  2. 整数化用最大余数法 (Hamilton) 逐行补平到恰好 1e6, 这样 counts 恰等于 CPM,
     打分器的归一化成为恒等映射.
"""

from __future__ import annotations

import numpy as np

from .scorer import TS_CELL, ControlRef

__all__ = ["hamilton", "design_cells"]


def hamilton(row: np.ndarray, total: int = 1_000_000) -> np.ndarray:
    """最大余数法取整, 使行和恰为 total. counts == CPM 的前提."""
    fl = np.floor(row)
    need = int(total - fl.sum())
    if need > 0:
        fl[np.argpartition(-(row - fl), need - 1)[:need]] += 1
    elif need < 0:
        nz = np.flatnonzero(fl > 0)
        fl[nz[np.argpartition(row[nz] - fl[nz], -need - 1)[: -need]]] -= 1
    return fl


def design_cells(
    ref: ControlRef,
    r_set,
    lfc,
    n_cells: int = 400,
    shift: float = 0.10,
    seed: int = 0,
) -> np.ndarray:
    """Stage 2: 给定响应基因集 (gate 内下标) 与目标 lfc, 构造 n_cells 个整数
    计数细胞. 关键约束: CPM 是成分数据, 目标 profile 必须重归一到 1e6.

    null 背景用真实对照细胞自举 -> psi_bar 自动校准, 稀疏度/过散天然正确.
    响应基因用二点分布 (0, s), 对非零比例 f 二分, 使平均对照分位数命中
    0.5 +- shift. 显著性(psi_bar) 与方向(一阶矩) 完全解耦."""
    rg = np.random.default_rng(seed)
    r_set = np.asarray(r_set)
    lfc = np.asarray(lfc, dtype=float)

    lf = np.zeros(ref.n_genes)
    lf[ref.gidx[r_set]] = lfc
    tgt = ref.m_full * 2.0**lf
    tgt *= TS_CELL / tgt.sum()

    V = np.asarray(
        ref._cpm_csr[rg.choice(ref.n_ctrl, n_cells, replace=False)].todense()
    )
    V *= tgt / np.maximum(V.mean(0), 1e-12)

    for j, l in zip(r_set, lfc):
        mu = tgt[ref.gidx[j]]
        ut = 0.5 + np.sign(l) * shift
        lo, hi = 1.0 / n_cells, 1.0
        for _ in range(24):                       # psi_bar 对 f 单调 -> 二分
            f = 0.5 * (lo + hi)
            k = max(1, int(round(f * n_cells)))
            col = np.zeros(n_cells)
            col[:k] = mu * n_cells / k
            if ref.psi(j, col).mean() / ref.n_ctrl < ut:
                lo = f
            else:
                hi = f
        k = max(1, int(round(0.5 * (lo + hi) * n_cells)))
        col = np.zeros(n_cells)
        col[rg.permutation(n_cells)[:k]] = mu * n_cells / k
        V[:, ref.gidx[j]] = col

    V *= (TS_CELL / V.sum(1))[:, None]
    return np.vstack([hamilton(V[i]) for i in range(n_cells)]).astype(np.float32)
