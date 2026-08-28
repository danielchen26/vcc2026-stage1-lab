"""E01 — Wilcoxon 秩和的闭式约化, 与 scipy 逐位对齐.

    psi_g(v) = #{c : x_cg < v} + 0.5 * #{c : x_cg = v}
    U_g      = sum_i psi_g(v_i)

对照组在整场比赛固定不变 -> psi_g 是一张可以预先排好的表, 每个基因只需
400 次 searchsorted 就能得到官方 Wilcoxon 的 U 统计量.

跑法::

    ~/vcc2026/.venv/bin/python experiments/E01-psi-identity/run.py
    ~/vcc2026/.venv/bin/python experiments/E01-psi-identity/run.py --real   # 追加真实数据闭环

合成数据: n2=18,400 对照 / n1=400 处理, NB(r=2, p=2/(2+mu)), seed 0.
三种情形: null (mu=40 = 对照) / shifted (mu=60) / degenerate (400 个细胞全等于 44).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import mannwhitneyu

from _common import header

N_CTRL, N_TREAT = 18_400, 400
R_NB, MU_CTRL, MU_SHIFT = 2, 40.0, 60.0
DEGEN_VALUE = 44          # 点质量位置: 使 U 落在 shifted 情形附近


class PsiTable:
    """`vcc_local.ControlRef.psi` 的独立复刻 (同样的稀疏拆分与零值快路径).

    只依赖对照列, 所以可以离线预排序; 这正是 41x 加速的来源."""

    def __init__(self, ctrl: np.ndarray):
        ctrl = np.asarray(ctrl, dtype=np.float64)
        self.n = ctrl.size
        self.nzero = int((ctrl == 0).sum())
        self.sorted = np.sort(ctrl[ctrl != 0])

    def __call__(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        lo = np.searchsorted(self.sorted, v, "left")
        hi = np.searchsorted(self.sorted, v, "right")
        out = self.nzero + lo + 0.5 * (hi - lo)
        out[v == 0] = 0.5 * self.nzero
        return out


def check(name: str, ctrl: np.ndarray, treat: np.ndarray) -> bool:
    u_scipy = mannwhitneyu(treat, ctrl, method="asymptotic").statistic
    u_psi = PsiTable(ctrl)(treat).sum()
    ok = u_scipy == u_psi                      # 要求逐位相等, 不是 allclose
    print(
        f"{name:<12} U_scipy={u_scipy:12.1f}  U_psi={u_psi:12.1f}  match={ok}"
    )
    return bool(ok)


def real_data_closure() -> None:
    """闭环: 在真实 context_A 上, ControlRef.psi 对 scipy 的复核.

    这里用的是官方发布的对照 CPM 列 (含大量精确零) 与解码器构造的一个细胞块,
    覆盖合成数据碰不到的稀疏 / 并列结构."""
    from _common import load_ref

    ref, t_load = load_ref("A")
    print(f"\n[真实数据闭环] ControlRef 加载 {t_load:.1f}s  gate={ref.G}")
    rg = np.random.default_rng(0)
    for j in rg.choice(ref.G, 5, replace=False):
        ctrl = np.asarray(ref._cpm_csr[:, ref.gidx[j]].todense()).ravel()
        v = np.asarray(
            ref._cpm_csr[rg.choice(ref.n_ctrl, N_TREAT, replace=False),
                         ref.gidx[j]].todense()
        ).ravel() * 1.4
        u_scipy = mannwhitneyu(v, ctrl, method="asymptotic").statistic
        u_psi = ref.psi(int(j), v).sum()
        print(
            f"  gate j={j:<5d} 零 {int((ctrl == 0).sum()):5d}/{ref.n_ctrl}  "
            f"U_scipy={u_scipy:12.1f}  U_psi={u_psi:12.1f}  "
            f"match={u_scipy == u_psi}"
        )


def main() -> int:
    header("E01 psi 闭式恒等")
    rg = np.random.default_rng(0)
    ctrl = rg.negative_binomial(R_NB, R_NB / (R_NB + MU_CTRL), N_CTRL)
    cases = {
        "null draw": rg.negative_binomial(R_NB, R_NB / (R_NB + MU_CTRL), N_TREAT),
        "shifted": rg.negative_binomial(R_NB, R_NB / (R_NB + MU_SHIFT), N_TREAT),
        "degenerate": np.full(N_TREAT, DEGEN_VALUE),
    }
    hits = [check(name, ctrl, treat) for name, treat in cases.items()]
    print(f"\n{sum(hits)}/{len(hits)} 逐位相等")
    if "--real" in sys.argv:
        real_data_closure()
    return 0 if all(hits) else 1


if __name__ == "__main__":
    raise SystemExit(main())
