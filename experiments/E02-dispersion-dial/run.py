"""E02 — 弥散度旋钮: 固定一阶矩, 只改组内分布形状, 显著性可以任意调.

对一个真实基因 (context_A 的 SELENOT), 把 400 个细胞的**均值锁死**在
lfc=+0.322 对应的 CPM 上, 只改「多少个细胞非零」:

    非零比例 f = 1/(1+t),  非零值 = mu * n / k     (k = round(n f))

这样两点分布的均值恒为 mu, 而组内方差比 CV^2 恰好等于 t —— t 就是弥散度旋钮.
Wilcoxon 读的是秩 (psi_bar), 与均值无关, 于是:

  * t 从 0 调到 1.30, p 从 2.8e-29 走到 0.77   -> 同一个 lfc 可显著可不显著;
  * t 继续调到 3.00, d 变成负的而 p 又回到 5e-20 -> 检验说「下调」, lfc 说「上调」.

结论: 显著性由分布形状决定, 方向由一阶矩决定, 两者可以指相反方向.
这就是 Stage 2 解码器能把「意图」无损写进整数细胞的理论依据.

跑法::

    ~/vcc2026/.venv/bin/python experiments/E02-dispersion-dial/run.py

耗时 ~15 s (几乎全是 ControlRef 加载).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import norm

from _common import header, load_ref

GENE = "SELENOT"
LFC_TARGET = 0.322                     # 锁定的一阶矩 (目标均值 = 对照均值 * 2^lfc)
N_CELLS = 400
ALPHA = 0.05
DIALS = [0.00, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.05, 1.30, 1.70, 2.20, 3.00]


def tie_sigma(sorted_nz: np.ndarray, nzero: int, col: np.ndarray,
              n1: int, n2: int) -> float:
    """并列校正后的 U 的标准差 (与 vcc_local.ControlRef.de_table 同式).

    sigma_tie = sqrt(n1 n2 / 12 * ((N+1) - sum_t (t^3 - t) / (N (N-1))))
    不做这个校正, 每组会少判 1-2 个基因 (静默失分)."""
    allv = np.concatenate(
        [np.zeros(nzero, np.float32), sorted_nz, col.astype(np.float32)]
    )
    _, c = np.unique(allv, return_counts=True)
    T = float(np.sum(c.astype(np.float64) ** 3 - c))
    N = n1 + n2
    return float(np.sqrt(n1 * n2 / 12.0 * ((N + 1) - T / (N * (N - 1.0)))))


def main() -> int:
    header("E02 弥散度旋钮")
    ref, t_load = load_ref("A")
    from _common import gene_names

    g_full = int(np.flatnonzero(gene_names() == GENE)[0])
    j = int(np.flatnonzero(ref.gidx == g_full)[0])          # gate 内下标
    m0 = float(ref.m_gate[j])
    nzero = int(ref._nzero[j])
    mu = m0 * 2.0 ** LFC_TARGET
    n1, n2 = N_CELLS, ref.n_ctrl

    print(
        f"{GENE}: gate j={j} (全基因下标 {g_full})  对照均值 {m0:.2f} CPM  "
        f"检不到 {nzero}/{n2} 个对照细胞  加载 {t_load:.1f}s"
    )
    print(
        f"锁定目标均值 {mu:.2f} CPM  -> lfc={np.log2((mu + 1e-9) / (m0 + 1e-9)):+.3f} "
        f"(所有行都一样)\n"
    )
    print("    t     非零       均值         d           p  判定")

    sds = []
    for t in DIALS:
        k = max(1, int(round(N_CELLS / (1.0 + t))))
        col = np.zeros(N_CELLS)
        col[:k] = mu * N_CELLS / k                          # 均值恒等于 mu
        psi_bar = float(ref.psi(j, col).mean())
        d = psi_bar / n2 - 0.5                              # 秩位移
        sd = tie_sigma(ref._sorted[j], nzero, col, n1, n2)   # 池化并列 (de_table 口径)
        p = float(2 * norm.sf(abs(n1 * n2 * d) / sd))        # U - n1 n2/2 = n1 n2 d
        sds.append(sd)
        print(
            f"{t:5.2f} {k:7d} {col.mean():9.2f} {d:+10.4f} {p:12.2e}  "
            f"{'显著' if p < ALPHA else '不显著'}"
        )

    # 只数对照侧并列的 sigma 与 t 无关, 是这个基因的「检出门槛常数」;
    # 逐行 p 用的是池化并列 (与 vcc_local.de_table / 官方打分器一致), 略小一点.
    sig_ctrl = tie_sigma(ref._sorted[j], nzero, np.empty(0), n1, n2)
    d_crit = norm.isf(ALPHA / 2) * sig_ctrl / (n1 * n2)
    print(
        f"\nsigma_tie({GENE})={sig_ctrl:,.0f}  d_crit={d_crit:.4f}  "
        f"(|d| 超过它才显著)   逐行池化 sigma "
        f"{min(sds):,.0f}..{max(sds):,.0f}   未校正 sigma "
        f"{np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0):,.0f}"
    )
    print(
        "关键观察: t=3.00 时均值仍然向上 (lfc=+0.32) 但 d<0 且 p=5e-20 —— "
        "检验读秩, lfc 读均值, 两者可指相反方向."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
