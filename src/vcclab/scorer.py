"""cell-eval2 Wilcoxon DE 的精确复刻 (官方打分器的充分统计量).

本模块从 `~/vcc2026/vcc_local.py` 原样移植, **数值行为不变**.

已验证 (2026-08-27, context_A, cell-eval2 0.16.0, preset vcc2026):
  - gate 基因集      : 9929, 与官方完全一致
  - log2_fold_change : 最大绝对差 1.0e-5 (float32 存储噪声)
  - p_adj            : log10 中位绝对差 0.0000
  - 显著集 R̂         : 3/3 个扰动对称差 = 0  (需 tie correction)
  - 速度             : 41x 官方 scanpy CPU 后端

核心洞察: 官方打分器对每个基因只读两个标量 —— (a) 400 个预测细胞的均值,
(b) 这 400 个值在 18400 个对照细胞里的平均中位秩. 对照组在整场比赛固定不变,
所以 Wilcoxon 秩和有闭式:

    psi_g(v) = #{c: x_cg < v} + 0.5 * #{c: x_cg = v}
    U_g      = sum_i psi_g(v_i)

与 `scipy.stats.mannwhitneyu` 逐位相等 (3/3 验证, 含 400 细胞全相同的退化情形).
预排序对照列后每基因只需 400 次 searchsorted.

易错点 (错一个就静默失分):
  1. BH 校正必须是 `np.minimum.accumulate(q[::-1])[::-1]` (min over j>=i).
     写成 maximum 会让发现数从 250 变成 0, 且不报错.
  2. Wilcoxon 必须做并列校正:
     sigma_tie = sqrt(n1*n2/12 * ((N+1) - sum_t(t^3-t)/(N*(N-1)))), N = n1+n2.
     实测中位 104,477 vs 未校正 107,384. 不做校正每组会少判 1-2 个基因.
  3. DE 只测「对照均值 CPM > 5」的基因 (gate). 实测 context A 9929, B 9626, C 10124.

构造侧的两个易错点 (CPM 重归一 / Hamilton 取整) 见 `vcclab.decoder`.
"""

from __future__ import annotations

import os

import numpy as np
from scipy import sparse
from scipy.stats import norm

TS_BULK = 5e4      # pds / mse 的 pseudobulk target sum
TS_CELL = 1e6      # DE 的 per-cell target sum
GATE_CPM = 5.0     # filter_gene_min_cpm_cell
ALPHA = 0.05       # p_adj_threshold, Benjamini-Hochberg
EPS = 1e-9         # fold-change epsilon

__all__ = [
    "TS_BULK",
    "TS_CELL",
    "GATE_CPM",
    "ALPHA",
    "EPS",
    "bh_adjust",
    "ControlRef",
]


def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up. 注意是 min_{j>=i}, 不是 max."""
    m = len(p)
    order = np.argsort(p)
    q = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


class ControlRef:
    """一个 cell context 的参考对照组. 官方 manifest 的 ground_truth_cells
    = 300*400 + 18400 证实发布的对照细胞就是打分用的比较组."""

    def __init__(self, h5ad_path, gene_names):
        import h5py

        with h5py.File(h5ad_path, "r") as f:
            X = sparse.csr_matrix(
                (f["X/data"][:], f["X/indices"][:], f["X/indptr"][:]),
                shape=tuple(f["X"].attrs["shape"]),
            )
            var = np.array(f["var/_index/values"][:], dtype=object).astype(str)
        if not np.array_equal(var, np.asarray(gene_names)):
            raise ValueError("var 顺序与 gene_names.csv 不一致")

        self.n_ctrl, self.n_genes = X.shape
        lib = np.asarray(X.sum(1)).ravel()
        cpm = X.multiply((TS_CELL / lib)[:, None]).tocsc()

        self.m_full = np.asarray(cpm.mean(0)).ravel()          # 全基因对照均值 CPM
        pb = np.asarray(X.sum(0)).ravel()
        self.b_ctrl = np.log1p(TS_BULK * pb / pb.sum())        # pds/mse 的对照 pseudobulk
        self.gidx = np.flatnonzero(self.m_full > GATE_CPM)     # DE gate
        self.m_gate = self.m_full[self.gidx]
        self.G = len(self.gidx)

        sub = cpm[:, self.gidx].tocsc()
        self._sorted = [
            np.sort(sub.data[sub.indptr[j] : sub.indptr[j + 1]]).astype(np.float32)
            for j in range(self.G)
        ]
        self._nzero = np.array(
            [self.n_ctrl - a.size for a in self._sorted], dtype=np.int64
        )
        self._cpm_csr = cpm.tocsr()

    @classmethod
    def load(cls, h5ad_path, gene_names) -> "ControlRef":
        """主入口. `h5ad_path` 支持 `~` 展开; `gene_names` 为长度 n_genes 的名字序列
        (顺序必须与 h5ad 的 var 一致, 否则抛 ValueError)."""
        return cls(os.path.expanduser(str(h5ad_path)), gene_names)

    def psi(self, j: int, v: np.ndarray) -> np.ndarray:
        """psi_g(v) = #{ctrl < v} + 0.5 * #{ctrl == v}. Wilcoxon 的充分统计量:
        U_g = sum_i psi_g(v_i)  (对 scipy.mannwhitneyu 逐位验证)."""
        col, nz = self._sorted[j], self._nzero[j]
        lo = np.searchsorted(col, v, "left")
        hi = np.searchsorted(col, v, "right")
        out = nz + lo + 0.5 * (hi - lo)
        out[v == 0] = 0.5 * nz
        return out

    def tie_cube_sum_ctrl(self, j: int) -> float:
        """sum_t (t^3 - t) over 对照列自身的并列组 (含 0 值那一组).

        这是 de_table 里 T[j] 的对照部分; 与样本部分可加分解, 见 detectability.mde.
        """
        col = self._sorted[j]
        nz = float(self._nzero[j])
        total = nz**3 - nz
        if col.size:
            bnd = np.flatnonzero(
                np.concatenate(([True], col[1:] != col[:-1], [True]))
            )
            c = np.diff(bnd).astype(np.float64)
            total += float(np.sum(c**3 - c))
        return total

    def de_table(self, counts: np.ndarray, tie_correct: bool = True):
        """cell-eval2 wilcoxon DE 的精确复刻. counts 行和须为 1e6.
        返回 (p_adj, log2fc), 均在 gate 内, 长度 self.G."""
        n1 = counts.shape[0]
        n2 = self.n_ctrl
        N = n1 + n2
        U = np.empty(self.G)
        T = np.empty(self.G)
        for j in range(self.G):
            v = counts[:, self.gidx[j]].astype(np.float64)
            U[j] = self.psi(j, v).sum()
            allv = np.concatenate(
                [np.zeros(self._nzero[j], np.float32), self._sorted[j], v.astype(np.float32)]
            )
            _, c = np.unique(allv, return_counts=True)
            T[j] = np.sum(c.astype(np.float64) ** 3 - c)
        if tie_correct:
            sd = np.sqrt(n1 * n2 / 12.0 * ((N + 1) - T / (N * (N - 1.0))))
        else:
            sd = np.full(self.G, np.sqrt(n1 * n2 * (N + 1) / 12.0))
        p = 2 * norm.sf(np.abs((U - n1 * n2 / 2) / sd))
        lfc = np.log2((counts[:, self.gidx].mean(0) + EPS) / (self.m_gate + EPS))
        return bh_adjust(p), lfc
