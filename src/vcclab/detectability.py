"""最小可检出效应 (Minimum Detectable Effect, MDE).

对 gate 内基因 j: 从对照里自举 n_cells 个细胞的该基因 CPM 值, 全体乘 2**lfc 后
取整, 求使**单基因双侧 Wilcoxon 检验**刚好达到显著的最小 |lfc|.

显著判据 (与 `vcclab.scorer.ControlRef.de_table` 的 z 检验等价):

    U        = sum_i psi_j(v_i) = n1 * psi_bar
    E[U]     = n1 * n2 / 2
    |U - E[U]| > z_{1-alpha/2} * sigma
  <=> |psi_bar / n_ctrl - 0.5| > z_{1-alpha/2} * sigma / (n1 * n2)

`tie_correct=True` 时 sigma 逐基因逐步重算 (并列结构随基因与 lfc 变):

    sigma = sqrt(n1*n2/12 * ((N+1) - T/(N*(N-1)))),  N = n1 + n2
    T     = sum_t (t^3 - t)  over 合并样本的并列组

否则 sigma = sqrt(n1*n2*(N+1)/12).

## 为什么能向 9929 个基因 * 20 次求值 扩展

1. T 对「对照并列组」与「样本值」可加分解. 记 c_c(v) = #{ctrl == v},
   c_s(v) = #{sample == v}, 则合并样本只改动那些被样本命中的组:

       T = T_ctrl + sum_{v in unique(sample)} [ (c_c+c_s)^3-(c_c+c_s) - (c_c^3-c_c) ]

   T_ctrl 与 lfc 无关 -> 每基因预计算一个标量
   (`ControlRef.tie_cube_sum_ctrl`). 于是每次求值只碰 <= n_cells 个元素,
   不再对 18800 个元素重排.
2. `ControlRef._sorted[j]` 存成 float32, 而 v 是 float64; 直接 searchsorted 会
   让 numpy 每次调用都把整列升位成 float64 (18400 * 8 B 的拷贝, x2). 这里每基因
   只升位一次然后复用 -> 实测全 gate 的挂钟时间减半.

## 实测参考值 (context_A, 全 9929 个 gate 内基因, n_cells=400, alpha=0.05)

                          q05     q25     med     q75     q95    挂钟
  tie_correct=False      0.011   0.097   0.184   0.375   1.099   15.1 s
  tie_correct=True       0.000   0.093   0.180   0.362   1.054   25.2 s
  独立参考实现            0.0326  0.1072  0.1930  0.3777  1.0852  10.4 s

q25 / med / q75 / q95 与独立参考实现都在 6% 内. q05 结构性退化, **不是 bug**:
lfc = 0 时 dev = psi_bar/n2 - 0.5 的 sd = (1/sqrt(12))/sqrt(n_cells) = 0.014434,
而阈值是 0.028596, 所以理论上 2*sf(0.028596/0.014434) = 4.76% 的基因「自举样本
本身就已显著」-> MDE 恰为 0.0 (实测 4.05%). 5% 分位正好压在这块零质量的边缘,
跨自举种子在 0.000-0.024 之间摆动 (6 个种子实测). 需要「纯效应量」而不含自举
噪声时, 应把 dev 改成相对 lfc=0 基线的增量 (同条件下 q05 升到 0.077).

正确性由测试钉住: psi 与 scipy.stats.mannwhitneyu 逐位相等, 并列 T 的可加分解与
`de_table` 内联的 np.unique 版本逐位相等, 返回值确为显著性边界 (x1.02 显著 /
x0.98 不显著).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .scorer import ControlRef

__all__ = ["mde"]


def _psi(col: np.ndarray, nz: float, v: np.ndarray) -> np.ndarray:
    """`ControlRef.psi` 的等价快路径; `col` 须已是 float64 的非零升序对照列."""
    lo = np.searchsorted(col, v, "left")
    hi = np.searchsorted(col, v, "right")
    out = nz + lo + 0.5 * (hi - lo)
    zero = v == 0
    if zero.any():
        out = np.where(zero, 0.5 * nz, out)
    return out


def _tie_cube_sum(col: np.ndarray, nz: float, t_ctrl: float, v: np.ndarray) -> float:
    """合并样本 (对照 n2 个 + 样本 v) 的 sum_t (t^3 - t).

    精确等于 `de_table` 内联的 `np.unique(concat(...))` 版本, 见
    tests/test_vcclab.py::test_tie_decomposition_matches_de_table.
    """
    vs, cs = np.unique(v, return_counts=True)
    cc = (
        np.searchsorted(col, vs, "right") - np.searchsorted(col, vs, "left")
    ).astype(np.float64)
    if vs[0] == 0.0:
        cc[0] = nz                       # 零值那组不在 col 里 (稀疏存储)
    cs = cs.astype(np.float64)
    tot = cc + cs
    return float(t_ctrl + np.sum(tot**3 - tot) - np.sum(cc**3 - cc))


def mde(
    ref: ControlRef,
    gene_idx=None,
    n_cells: int = 400,
    alpha: float = 0.05,
    hi: float = 4.0,
    n_bisect: int = 18,
    seed: int = 0,
    tie_correct: bool = True,
) -> np.ndarray:
    """每基因的最小可检出 |log2 fold change|.

    Parameters
    ----------
    ref : ControlRef
    gene_idx : array-like[int] | None
        gate 内下标. None = 全部 gate 内基因.
    n_cells : int
        自举样本量 (= 提交格式里每个扰动的细胞数).
    alpha : float
        双侧显著水平.
    hi : float
        搜索上界; |lfc| > hi 仍不显著的基因返回 np.nan.
    n_bisect : int
        二分步数. 18 步 -> 分辨率 hi / 2**18 ~ 1.5e-5.
    seed : int
        自举种子. 一次有放回抽取 n_cells 个对照细胞, 所有基因共用同一批细胞
        (逐基因边缘分布不变, 且与 `decoder.design_cells` 的背景一致).
    tie_correct : bool
        True 时逐基因逐步重算并列校正 sigma.

    Returns
    -------
    np.ndarray
        长度与 `gene_idx` 一致 (None 时为 ref.G) 的 |lfc| 数组.
        lfc = 0 就已显著 (自举噪声本身越界) 记 0.0; |lfc| > hi 仍不显著记 np.nan.
    """
    gidx_gate = (
        np.arange(ref.G) if gene_idx is None else np.asarray(gene_idx, dtype=np.int64)
    )
    n1, n2 = n_cells, ref.n_ctrl
    N = n1 + n2
    z = norm.isf(alpha / 2.0)
    thr_flat = z * np.sqrt(n1 * n2 * (N + 1) / 12.0) / (n1 * n2)
    tie_lead = z / (n1 * n2) * np.sqrt(n1 * n2 / 12.0)
    tie_den = N * (N - 1.0)

    if tie_correct and not hasattr(ref, "_tie_ctrl"):
        ref._tie_ctrl = np.array(
            [ref.tie_cube_sum_ctrl(j) for j in range(ref.G)], dtype=np.float64
        )

    rg = np.random.default_rng(seed)
    cells = rg.choice(n2, n_cells, replace=True)
    V = np.asarray(
        ref._cpm_csr[cells][:, ref.gidx[gidx_gate]].todense(), dtype=np.float64
    )

    out = np.full(gidx_gate.size, np.nan)
    for k in range(gidx_gate.size):
        j = int(gidx_gate[k])
        col = ref._sorted[j].astype(np.float64)     # 每基因升位一次, 供 20 次求值复用
        nz = float(ref._nzero[j])
        base = V[:, k]
        t_ctrl = float(ref._tie_ctrl[j]) if tie_correct else 0.0

        def significant(lfc: float) -> bool:
            v = np.rint(base * 2.0**lfc)
            dev = abs(_psi(col, nz, v).mean() / n2 - 0.5)
            if not tie_correct:
                return dev > thr_flat
            T = _tie_cube_sum(col, nz, t_ctrl, v)
            return dev > tie_lead * np.sqrt((N + 1) - T / tie_den)

        if significant(0.0):
            out[k] = 0.0
            continue
        if not significant(hi):
            continue
        lo_l, hi_l = 0.0, hi
        for _ in range(n_bisect):        # dev 对 lfc 单调 -> 二分
            mid = 0.5 * (lo_l + hi_l)
            if significant(mid):
                hi_l = mid
            else:
                lo_l = mid
        out[k] = hi_l
    return out
