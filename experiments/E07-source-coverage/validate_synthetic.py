"""E07 度量代码的合成数据验证 —— 在真实数据到手前先确认代码是对的。

思路：造两个「细胞系」，各自有已知的真实响应基因集，两者按可控的重叠率
share 一部分。然后看 crossline.py 测出来的 h_cross 是否回收了我们设定的重叠率，
以及 h_replicate 是否落在合理范围。

这不是单元测试的替代，而是**端到端标定**：如果设定 60% 重叠而测出 0.15，
说明降采样/判定/统计某处有偏，真实数据上的结论就不可信。

跑法：  ~/vcc2026/.venv/bin/python experiments/E07-source-coverage/validate_synthetic.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vcclab.crossline import h_to_jac, overlap_h  # noqa: E402

N_GENES = 2000          # 缩小的基因轴，够跑通逻辑且几秒出结果
N_CTRL = 4000
N_CELLS = 400
MEDIAN_UMI = 20_000
ALPHA = 0.05


def make_control(rng, n_cells=N_CTRL):
    """造一个「细胞系」的对照：基因表达强度服从对数正态，细胞按多项抽样。"""
    rate = rng.lognormal(mean=2.0, sigma=1.6, size=N_GENES)
    rate /= rate.sum()
    lib = rng.negative_binomial(8, 8 / (8 + MEDIAN_UMI), size=n_cells) + 1
    return np.array([rng.multinomial(int(l), rate) for l in lib], dtype=np.float32), rate


def make_perturbed(rng, rate, resp_idx, lfc, n_cells=N_CELLS, mass_neutral=True):
    """把 resp_idx 上的表达按 2**lfc 改变，重归一后多项抽样。

    mass_neutral=True 时，改动后**只在响应基因内部**把总质量拉回原值，
    使非响应基因的比例完全不变。否则会有成分泄漏：改动 120 个基因会让其余
    1,900 个基因被整体平移，400 个细胞下该平移可检测 → 大量与设定重叠率
    无关的假阳性 → 度量饱和。（首次标定就是这样 FAIL 的。）
    """
    r = rate.copy()
    old = r[resp_idx].sum()
    r[resp_idx] *= 2.0 ** lfc
    if mass_neutral and r[resp_idx].sum() > 0:
        r[resp_idx] *= old / r[resp_idx].sum()
    r /= r.sum()
    lib = rng.negative_binomial(8, 8 / (8 + MEDIAN_UMI), size=n_cells) + 1
    return np.array([rng.multinomial(int(l), r) for l in lib], dtype=np.float32)


def de_set(ctrl, pert, gate, alpha=ALPHA):
    """最小的官方判定复刻：per-cell CPM → Wilcoxon(含并列校正) → 门内 BH。"""
    from scipy.stats import norm

    from vcclab.scorer import bh_adjust

    c = ctrl / ctrl.sum(1, keepdims=True) * 1e6
    p_ = pert / pert.sum(1, keepdims=True) * 1e6
    n1, n2 = p_.shape[0], c.shape[0]
    N = n1 + n2
    g = np.flatnonzero(gate)
    pv = np.empty(len(g))
    for k, j in enumerate(g):
        cc, pp = c[:, j], p_[:, j]
        allv = np.concatenate([cc, pp])
        order = np.argsort(allv, kind="stable")
        ranks = np.empty(N)
        ranks[order] = np.arange(1, N + 1)
        # 并列取中位秩
        uniq, inv, cts = np.unique(allv, return_inverse=True, return_counts=True)
        mean_rank = np.zeros(len(uniq))
        np.add.at(mean_rank, inv, ranks)
        mean_rank /= cts
        ranks = mean_rank[inv]
        R1 = ranks[n2:].sum()
        U = R1 - n1 * (n1 + 1) / 2
        T = float(np.sum(cts.astype(float) ** 3 - cts))
        sd = np.sqrt(n1 * n2 / 12.0 * ((N + 1) - T / (N * (N - 1.0))))
        pv[k] = 2 * norm.sf(abs((U - n1 * n2 / 2) / sd)) if sd > 0 else 1.0
    out = np.zeros(len(gate), bool)
    out[g] = bh_adjust(pv) < alpha
    return out


def main() -> None:
    rng = np.random.default_rng(0)
    t0 = time.time()
    print("=== E07 度量代码的合成标定 ===")
    print(f"基因 {N_GENES} · 对照 {N_CTRL} · 每扰动 {N_CELLS} 细胞 · 中位 {MEDIAN_UMI:,} UMI\n")

    ctrl_a, rate_a = make_control(rng)
    ctrl_b, rate_b = make_control(rng)
    gate_a = (ctrl_a / ctrl_a.sum(1, keepdims=True) * 1e6).mean(0) > 5
    gate_b = (ctrl_b / ctrl_b.sum(1, keepdims=True) * 1e6).mean(0) > 5
    gate = gate_a & gate_b
    print(f"两系共同 gate: {int(gate.sum()):,} 个基因\n")

    pool = np.flatnonzero(gate)
    n_resp = 120
    base = rng.choice(pool, n_resp, replace=False)   # 固定，使各档可比
    print(f"{'设定重叠率':>10} {'测出 h_cross':>13} {'回收率':>8} "
          f"{'|R_A|':>7} {'|R_B|':>7} {'h_rep(A)':>10} {'h_rep(B)':>10} {'比值':>8}")
    rows = []
    for share in (0.2, 0.4, 0.6, 0.8):
        n_shared = int(round(share * n_resp))
        rest = np.setdiff1d(pool, base)
        resp_a = base
        resp_b = np.concatenate([base[:n_shared],
                                rng.choice(rest, n_resp - n_shared, replace=False)])
        lfc_a = rng.choice([-1, 1], n_resp) * rng.uniform(0.5, 1.6, n_resp)
        lfc_b = lfc_a.copy()          # 共享部分方向一致（保守：只测集合重叠）

        pa = make_perturbed(rng, rate_a, resp_a, lfc_a)
        pb = make_perturbed(rng, rate_b, resp_b, lfc_b)
        ra, rb = de_set(ctrl_a, pa, gate), de_set(ctrl_b, pb, gate)

        # 同系 replicate：再独立造一次同样的扰动
        pa2 = make_perturbed(rng, rate_a, resp_a, lfc_a)
        pb2 = make_perturbed(rng, rate_b, resp_b, lfc_b)
        hra = overlap_h(ra, de_set(ctrl_a, pa2, gate))
        hrb = overlap_h(rb, de_set(ctrl_b, pb2, gate))

        h = overlap_h(ra, rb)
        ratio = h / np.nanmean([hra, hrb])
        rows.append((share, h, ratio))
        print(f"{share:10.1f} {h:13.3f} {h/share:8.2f} {int(ra.sum()):7d} "
              f"{int(rb.sum()):7d} {hra:10.3f} {hrb:10.3f} {ratio:8.3f}")

    shares = np.array([r[0] for r in rows])
    hs = np.array([r[1] for r in rows])
    ratios = np.array([r[2] for r in rows])
    corr = float(np.corrcoef(shares, hs)[0, 1])
    mono = bool(np.all(np.diff(hs) > 0)) and bool(np.all(np.diff(ratios) > 0))

    print(f"\n设定重叠率 vs 测出 h 的相关: {corr:.4f}")
    print(f"h 与比值都随设定值单调上升: {mono}")
    print(f"耗时 {time.time() - t0:.1f}s\n")

    print(f"（设定的真实响应基因数 = {n_resp}；若 |R| 远大于此说明仍有成分泄漏）")
    ok = corr > 0.98 and mono
    print("判定: " + ("PASS — 度量代码单调且高度线性，可用于真实数据"
                     if ok else "FAIL — 度量有偏，真实数据上的结论不可信，先排查"))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
