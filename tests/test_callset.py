"""callset 的标定：反卷积能否回收真实的 pi0 / tau，以及最优 K 的两个命题。

每条测试对应一个已经犯过或差点犯的错：
  - 不做反卷积直接过阈值 → 噪声底（E08 的 run_erp.py 得 981）
  - 用命题 2 的闭式定 K → p 曲线呈阶跃时误差 68%
  - se 极大的基因被吸进零成分 → pi0 偏高（低测序深度源数据的真实风险）
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm, pearsonr

from vcclab.callset import (best_k, call_set, expected_jaccard_curve, fit_prior,
                            p_exceed, stopping_threshold)


def _sim(rng, g, pi0, tau, se_lo=0.15, se_hi=0.25):
    se = rng.uniform(se_lo, se_hi, g)
    nz = rng.random(g) >= pi0
    beta = np.where(nz, rng.standard_normal(g) * tau, 0.0)
    return beta + rng.standard_normal(g) * se, se, beta, nz


@pytest.mark.parametrize("pi0_true,tau_true", [(0.99, 0.5), (0.95, 0.4), (0.80, 0.3),
                                               (0.50, 0.25)])
def test_fit_prior_recovers_truth(pi0_true, tau_true):
    """pi0 可辨识；tau 只在宽容差内可辨识 —— (pi0, tau) 沿一条脊互换。

    实测脊的平坦程度（pi0=0.99, 40k 基因, se=0.20）：
        tau=0.50 → pi0=0.9884, 非零 464, -logL=-43332.40   ← 真值附近
        tau=0.55 → pi0=0.9902, 非零 391, -logL=-43331.73
        tau=0.62 → pi0=0.9920, 非零 320, -logL=-43328.51
    整段只差约 4 个 log 单位，所以单报 tau 没有意义。
    **下游真正用的是 p_exceed，它在脊上是稳的** —— 由
    test_p_exceed_recovers_true_count 用紧容差钉住。
    """
    rng = np.random.default_rng(0)
    bh, se, _, _ = _sim(rng, 40_000, pi0_true, tau_true)
    pi0, tau = fit_prior(bh, se)
    assert pi0 == pytest.approx(pi0_true, abs=0.02)
    assert tau == pytest.approx(tau_true, rel=0.30)


def test_fit_prior_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="形状不一致"):
        fit_prior(np.zeros(10), np.ones(9))


def test_fit_prior_rejects_too_few_observations():
    with pytest.raises(ValueError, match="观测太少"):
        fit_prior(np.zeros(5), np.ones(5))


def test_p_exceed_beats_naive_thresholding_on_false_positives():
    """核心动机：朴素过阈值制造大量假阳性，反卷积不会。

    这正是 E08 run_erp.py 的 981 是怎么来的。
    """
    rng = np.random.default_rng(1)
    g = 8_000
    bh, se, beta, nz = _sim(rng, g, pi0=0.995, tau=0.5)
    thr = np.full(g, 0.30)

    naive_fp = int(((np.abs(bh) > thr) & ~nz).sum())
    pi0, tau = fit_prior(bh, se)
    p = p_exceed(bh, se, thr, pi0, tau)
    eb_fp = float(p[~nz].sum())          # 假阳性的期望个数

    assert naive_fp > 500, f"朴素法应制造大量假阳性，实得 {naive_fp}"
    assert eb_fp < naive_fp / 20, f"反卷积假阳性期望 {eb_fp:.1f} 应远小于 {naive_fp}"


def test_p_exceed_calibration_under_null():
    """全零真值时，越阈概率之和应接近 0（而不是接近 13% x G）。"""
    rng = np.random.default_rng(2)
    g = 20_000
    se = rng.uniform(0.15, 0.25, g)
    bh = rng.standard_normal(g) * se        # beta 全为 0
    pi0, tau = fit_prior(bh, se)
    p = p_exceed(bh, se, np.full(g, 0.30), pi0, tau)
    naive = int((np.abs(bh) > 0.30).sum())
    assert naive > 1_000
    assert p.sum() < 0.02 * g


def test_p_exceed_recovers_true_count():
    """E|R_p| = sum(p) 应回收真实的越阈基因数。"""
    rng = np.random.default_rng(3)
    g = 30_000
    bh, se, beta, _ = _sim(rng, g, pi0=0.97, tau=0.6)
    thr = np.full(g, 0.35)
    truth = int((np.abs(beta) > thr).sum())
    pi0, tau = fit_prior(bh, se)
    est = float(p_exceed(bh, se, thr, pi0, tau).sum())
    assert est == pytest.approx(truth, rel=0.15)


def test_p_exceed_monotone_in_threshold():
    rng = np.random.default_rng(4)
    bh, se, _, _ = _sim(rng, 3_000, pi0=0.95, tau=0.5)
    pi0, tau = fit_prior(bh, se)
    tots = [p_exceed(bh, se, np.full(3_000, t), pi0, tau).sum()
            for t in (0.1, 0.2, 0.4, 0.8, 1.6)]
    assert all(np.diff(tots) < 0)


def test_p_exceed_handles_per_gene_thresholds():
    """目标 context 的 MDE 逐基因不同 —— 这是 TAP 的核心，必须支持。"""
    rng = np.random.default_rng(5)
    g = 5_000
    bh, se, _, _ = _sim(rng, g, pi0=0.95, tau=0.5)
    pi0, tau = fit_prior(bh, se)
    lo = p_exceed(bh, se, np.full(g, 0.15), pi0, tau)
    hi = p_exceed(bh, se, np.full(g, 0.60), pi0, tau)
    mixed = p_exceed(bh, se, np.where(np.arange(g) % 2 == 0, 0.15, 0.60), pi0, tau)
    np.testing.assert_allclose(mixed[::2], lo[::2])
    np.testing.assert_allclose(mixed[1::2], hi[1::2])


def test_best_k_matches_brute_force():
    rng = np.random.default_rng(6)
    p = np.sort(rng.beta(0.4, 3.0, 4_000))[::-1]
    n = float(p.sum())
    k, j = best_k(p, n)
    ks, js = expected_jaccard_curve(p, n)
    assert j == pytest.approx(js.max())
    assert k == int(ks[int(np.argmax(js))])


def test_best_k_on_step_curve_equals_step_size():
    """阶跃型 p（= 已知 R_p）时最优 K 应等于台阶宽度。"""
    for n_step in (15, 100, 288):
        p = np.concatenate([np.full(n_step, 0.9), np.full(4_000 - n_step, 0.001)])
        k, j = best_k(p, float(p.sum()))
        assert abs(k - n_step) <= max(2, n_step // 20), f"{n_step}: 得 {k}"
        assert j > 0.7


def test_best_k_exceeds_n_when_uncertain():
    """命题 2 的设计推论：p 曲线平滑（预测不确定）时 K* > E|R_p|。"""
    rng = np.random.default_rng(7)
    g = 6_000
    p = 0.45 * np.exp(-np.arange(g) / 300.0)     # 平滑衰减
    n = float(p.sum())
    k, _ = best_k(p, n)
    assert k > n, f"K*={k} 应大于 E|R_p|={n:.0f}"


def test_closed_form_stopping_matches_numeric_on_smooth_curves():
    """命题 2 闭式在平滑曲线上应准（误差 <10%）。"""
    g = 6_000
    p = 0.8 * np.exp(-np.arange(g) / 260.0)
    k, j = best_k(p, float(p.sum()))
    assert p[k - 1] == pytest.approx(stopping_threshold(j), rel=0.10)


def test_closed_form_stopping_fails_on_step_curve():
    """闭式在阶跃曲线上必须失效 —— 这是「实现里不用闭式」的理由，钉住它。"""
    p = np.concatenate([np.full(288, 0.62), np.full(6_000 - 288, 0.004)])
    k, j = best_k(p, float(p.sum()))
    assert p[k - 1] / stopping_threshold(j) > 1.4


def test_official_anchor_reproduces_h_over_two():
    """J=0.399（官方 replicate 锚点）→ p* = 0.285 = h/2。"""
    j = 0.399
    h = 2 * j / (1 + j)
    assert stopping_threshold(j) == pytest.approx(h / 2, rel=1e-9)
    assert stopping_threshold(j) == pytest.approx(0.285, abs=0.001)


def test_low_depth_inflates_then_collapses_erp():
    """低测序深度的真实失效模式：E|R_p| **先高估、再崩到 0**，不是单调低估。

    实测（真值 322 个基因越阈 0.30，tau=0.5）：
        se=0.18  tau/se=2.78  →  315.5   ← 良态，准确
        se=0.36  tau/se=1.39  →  331.3
        se=0.54  tau/se=0.93  →  648.3   ← 开始高估
        se=0.90  tau/se=0.56  →  984.0   ← 严重高估
        se=1.44  tau/se=0.35  →    0.0   ← 彻底崩溃

    机制：tau/se 变小时两个成分的方差趋同（N(0,se^2) vs N(0,se^2+tau^2)），
    (pi0, tau) 变得不可辨识，估计可以往任何方向跑。

    **实践判据：tau/se 是可用性的守门指标，必须报出来。**
    K562 GWPS 的 tau/se = 0.495/0.199 = 2.49 落在良态区，所以它的 E|R_p| 可信
    （见 docs/02-findings.md F20 的更正记录 —— 我先前误以为低深度会压低该估计）。
    """
    rng = np.random.default_rng(8)
    g = 20_000
    nz = rng.random(g) >= 0.97
    beta = np.where(nz, rng.standard_normal(g) * 0.5, 0.0)
    thr = np.full(g, 0.30)
    truth = int((np.abs(beta) > thr).sum())

    def erp(mult):
        se = np.full(g, 0.18 * mult)
        bh = beta + rng.standard_normal(g) * se
        pi0, tau = fit_prior(bh, se)
        return float(p_exceed(bh, se, thr, pi0, tau).sum()), tau / se[0]

    e1, r1 = erp(1)
    e3, r3 = erp(3)
    e8, r8 = erp(8)

    assert r1 > 2.0 and e1 == pytest.approx(truth, rel=0.15), \
        f"tau/se={r1:.2f} 应良态: {e1:.0f} vs {truth}"
    assert e3 > 1.5 * truth, f"tau/se={r3:.2f} 应高估: {e3:.0f} vs {truth}"
    assert e8 < 0.5 * truth or e8 > 1.5 * truth, \
        f"tau/se={r8:.2f} 应失控（任一方向）: {e8:.0f} vs {truth}"


def test_call_set_end_to_end():
    rng = np.random.default_rng(9)
    g = 8_000
    bh, se, beta, _ = _sim(rng, g, pi0=0.96, tau=0.55)
    thr = rng.uniform(0.15, 0.60, g)
    idx, p, info = call_set(bh, se, thr)

    assert len(idx) == info["K"]
    assert len(np.unique(idx)) == len(idx)
    # 召集的必须是 p 最大的那批（命题 1）
    assert p[idx].min() >= np.sort(p)[::-1][info["K"] - 1] - 1e-12
    # 真实重叠应显著高于随机同规模集合
    truth = np.abs(beta) > thr
    rand = rng.permutation(g)[:info["K"]]
    assert truth[idx].sum() > 3 * max(truth[rand].sum(), 1)
    for key in ("pi0", "tau", "K", "E_J", "E_Rp", "p_stop_closed_form"):
        assert key in info


def test_call_set_accepts_precomputed_prior():
    rng = np.random.default_rng(10)
    bh, se, _, _ = _sim(rng, 4_000, pi0=0.95, tau=0.5)
    thr = np.full(4_000, 0.3)
    pi0, tau = fit_prior(bh, se)
    i1, p1, _ = call_set(bh, se, thr, pi0=pi0, tau=tau)
    i2, p2, _ = call_set(bh, se, thr)
    np.testing.assert_array_equal(i1, i2)
    np.testing.assert_allclose(p1, p2)


def test_expected_jaccard_defaults_to_self_consistent_n():
    rng = np.random.default_rng(11)
    p = np.sort(rng.beta(0.5, 4.0, 2_000))[::-1]
    k1, j1 = expected_jaccard_curve(p)
    k2, j2 = expected_jaccard_curve(p, float(p.sum()))
    np.testing.assert_allclose(j1, j2)
