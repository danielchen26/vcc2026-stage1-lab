"""E07 汇总统计路径的标定：fission 独立性、top-K 度量的回收率、列对齐回归。

这三条每一条都对应一个已经犯过或差点犯的错：
  1. fission 若不独立，同系 replicate 分母会被系统性高估；
  2. top-K 度量若不单调/不回收，真实数据上的数不可读；
  3. 列不对齐会给出恰好等于随机水平的假 NO-GO —— 已经中过一次。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, pearsonr

from vcclab.crossline import (chance_overlap, fission, h_topk, h_topk_shuffled,
                              rank_matrix, ratio_cc, ratio_vs_permutation)

DATA = Path(__file__).resolve().parents[1] / "data" / "nadig2025"
LINES = ("K562", "RPE1", "Jurkat", "HepG2")


def test_fission_is_independent_not_just_uncorrelated():
    """tau=1 时两半的相关应为 0，且各自 SE = se*sqrt(2)。"""
    rng = np.random.default_rng(0)
    n = 200_000
    se = np.full(n, 0.4)
    beta = rng.standard_normal(n) * 0.0          # 真值取 0，只看噪声结构
    bhat = beta + rng.standard_normal(n) * se
    b1, b2, s1, s2 = fission(bhat, se, rng, tau=1.0)

    assert abs(pearsonr(b1, b2).statistic) < 0.01
    assert s1 == pytest.approx(se * np.sqrt(2))
    assert s2 == pytest.approx(se * np.sqrt(2))
    # 经验方差应落在 se^2*2 附近
    assert b1.var() == pytest.approx(2 * 0.4**2, rel=0.03)
    assert b2.var() == pytest.approx(2 * 0.4**2, rel=0.03)


def test_fission_preserves_mean():
    """两半的均值仍是原估计，劈分不引入偏差。"""
    rng = np.random.default_rng(1)
    bhat = rng.standard_normal(50_000) * 2.0
    se = np.abs(rng.standard_normal(50_000)) * 0.3 + 0.1
    b1, b2, _, _ = fission(bhat, se, rng)
    np.testing.assert_allclose(0.5 * (b1 + b2), bhat, rtol=1e-12)


def test_fission_tau_asymmetry():
    """tau != 1 时两个 SE 必须不同 —— 这里曾经返回单个值，是 bug。"""
    rng = np.random.default_rng(2)
    n = 200_000
    se = np.full(n, 0.5)
    bhat = rng.standard_normal(n) * se          # beta=0 + 真实抽样噪声
    b1, b2, s1, s2 = fission(bhat, se, rng, tau=2.0)
    assert not np.allclose(s1, s2)
    assert s1 == pytest.approx(se * np.sqrt(5))
    assert s2 == pytest.approx(se * np.sqrt(1.25))
    assert abs(pearsonr(b1, b2).statistic) < 0.01


def test_fission_independence_needs_input_variance():
    """独立性不是代数恒等式，而是分布陈述：Cov = Var(beta_hat) - se^2。

    若输入方差为 0（例如误传了真值而非估计），两半退化成同一个 Z 的缩放，
    相关变成 -1，同系 replicate 分母会被彻底算错。这里把该失效模式钉住。
    """
    rng = np.random.default_rng(7)
    se = np.full(20_000, 0.5)
    b1, b2, _, _ = fission(np.zeros(20_000), se, rng, tau=1.0)
    assert pearsonr(b1, b2).statistic == pytest.approx(-1.0, abs=1e-6)


def test_rank_matrix_is_permutation_and_picks_smallest():
    rng = np.random.default_rng(3)
    p = rng.random((500, 7))
    r = rank_matrix(p)
    for j in range(p.shape[1]):
        assert sorted(r[:, j]) == list(range(500))
        k = 30
        assert set(np.flatnonzero(r[:, j] < k)) == set(np.argsort(p[:, j])[:k])


def test_rank_matrix_sends_nan_last():
    p = np.array([[0.5], [np.nan], [0.1], [np.nan], [0.9]])
    r = rank_matrix(p)
    assert r[2, 0] == 0 and r[0, 0] == 1 and r[4, 0] == 2
    assert set(r[[1, 3], 0]) == {3, 4}


def _make_pair(rng, g, k, n_pert, designed, n_hub=0):
    """造两个细胞系的 p 矩阵。

    designed: 扰动特异性共享比例（从对方 top-K 的补集里取非共享部分）。
    n_hub:    「常常上榜」的枢纽基因数 —— 两个细胞系共有，且在**每个**扰动里都上榜。
              它们不携带任何扰动特异性信息，是置换零假设要扣掉的东西。
    """
    hub = rng.permutation(g)[:n_hub]
    pool = np.setdiff1d(np.arange(g), hub, assume_unique=False)
    k_free = k - n_hub
    pa, pb = np.empty((g, n_pert)), np.empty((g, n_pert))
    for j in range(n_pert):
        perm = rng.permutation(pool)
        top_a, rest = perm[:k_free], perm[k_free:]
        n_shared = int(round(designed * k_free))
        top_b = np.concatenate([top_a[:n_shared], rest[: k_free - n_shared]])
        for mat, top in ((pa, top_a), (pb, top_b)):
            col = rng.uniform(0.2, 1.0, g)
            col[np.concatenate([hub, top])] = rng.uniform(0.0, 1e-4, n_hub + k_free)
            mat[:, j] = col
    return pa, pb


def test_permutation_null_removes_hub_genes():
    """枢纽基因把 h_cross 抬高，但置换基线同样被抬高，差值仍回收真实共享比例。

    这正是 K/G 基线失效、必须换置换基线的原因：设计上扰动特异性共享为 0 时，
    枢纽基因会让 h_cross 远高于 K/G，看起来像强迁移信号 —— 实际是假的。
    """
    rng = np.random.default_rng(11)
    g, k, n_pert, n_hub = 4000, 200, 300, 80
    pa, pb = _make_pair(rng, g, k, n_pert, designed=0.0, n_hub=n_hub)
    ra, rb = rank_matrix(pa), rank_matrix(pb)

    h = h_topk(ra, rb, k)
    h_null = h_topk_shuffled(ra, rb, k)
    # 枢纽基因独占 80/200 = 0.40，远高于 K/G = 0.05 —— 用 K/G 会误判为强信号
    assert h > 0.35
    assert h > 5 * chance_overlap(k, g)
    # 置换基线抓住了同一批枢纽基因
    assert h_null == pytest.approx(h, abs=0.02)
    # 扣掉后没有剩余的扰动特异性信号
    assert h - h_null == pytest.approx(0.0, abs=0.02)


@pytest.mark.parametrize("designed", [0.2, 0.5, 0.8])
def test_permutation_null_preserves_real_signal(designed):
    """有枢纽基因干扰时，h_cross - h_null 仍应回收扰动特异性共享比例。"""
    rng = np.random.default_rng(12)
    g, k, n_pert, n_hub = 4000, 200, 300, 60
    pa, pb = _make_pair(rng, g, k, n_pert, designed=designed, n_hub=n_hub)
    ra, rb = rank_matrix(pa), rank_matrix(pb)
    excess = h_topk(ra, rb, k) - h_topk_shuffled(ra, rb, k)
    # 共享的是 k_free = k - n_hub 个自由名额中的 designed 比例
    expected = designed * (k - n_hub) / k
    assert excess == pytest.approx(expected, abs=0.03)


def test_h_topk_shuffled_has_no_fixed_points():
    """错位必须无不动点，否则真配对会漏进基线，把信号算进基线里。"""
    rng = np.random.default_rng(13)
    g, k, n_pert = 2000, 100, 200
    pa, _ = _make_pair(rng, g, k, n_pert, designed=1.0)
    r = rank_matrix(pa)
    # 自己和自己比：真配对 h = 1.0，任何不动点都会把基线推高
    assert h_topk(r, r, k) == pytest.approx(1.0)
    assert h_topk_shuffled(r, r, k) < 0.15


def test_ratio_vs_permutation_units():
    assert ratio_vs_permutation(0.5, 0.1, 0.5, 0.1) == pytest.approx(1.0)
    assert ratio_vs_permutation(0.1, 0.1, 0.5, 0.1) == pytest.approx(0.0)
    assert np.isnan(ratio_vs_permutation(0.3, 0.1, 0.1, 0.1))


@pytest.mark.parametrize("designed", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_h_topk_recovers_designed_overlap(designed):
    """造两组 top-K 名单，按设定比例共享，看度量能否精确回收。

    生成器里 top_b 的非共享部分取自 top_a 的**补集**，构造上排除了偶然重叠，
    所以真值恰好是 designed，不含随机项。随机基线本身由
    test_h_topk_hits_chance_when_independent 单独钉住。
    """
    rng = np.random.default_rng(4)
    g, k, n_pert = 6000, 288, 400
    n_shared = int(round(designed * k))
    pa = np.empty((g, n_pert))
    pb = np.empty((g, n_pert))
    for j in range(n_pert):
        perm = rng.permutation(g)
        top_a = perm[:k]
        rest = perm[k:]
        top_b = np.concatenate([top_a[:n_shared], rest[: k - n_shared]])
        for mat, top in ((pa, top_a), (pb, top_b)):
            col = rng.uniform(0.2, 1.0, g)
            col[top] = rng.uniform(0.0, 1e-4, k)
            mat[:, j] = col
    h = h_topk(rank_matrix(pa), rank_matrix(pb), k)
    assert h == pytest.approx(designed, abs=1e-9)


def test_h_topk_hits_chance_when_independent():
    """两组 top-K 完全独立抽取时，h 应贴住随机基线 K/G。"""
    rng = np.random.default_rng(8)
    g, k, n_pert = 6000, 288, 400
    pa, pb = np.empty((g, n_pert)), np.empty((g, n_pert))
    for j in range(n_pert):
        for mat in (pa, pb):
            col = rng.uniform(0.2, 1.0, g)
            col[rng.permutation(g)[:k]] = rng.uniform(0.0, 1e-4, k)
            mat[:, j] = col
    h = h_topk(rank_matrix(pa), rank_matrix(pb), k)
    assert h == pytest.approx(chance_overlap(k, g), abs=0.015)


def test_h_topk_monotone_and_correlated():
    rng = np.random.default_rng(5)
    g, k, n_pert = 4000, 100, 200
    designs = np.linspace(0.0, 1.0, 6)
    got = []
    for d in designs:
        n_shared = int(round(d * k))
        pa, pb = np.empty((g, n_pert)), np.empty((g, n_pert))
        for j in range(n_pert):
            perm = rng.permutation(g)
            top_a, rest = perm[:k], perm[k:]
            top_b = np.concatenate([top_a[:n_shared], rest[: k - n_shared]])
            for mat, top in ((pa, top_a), (pb, top_b)):
                col = rng.uniform(0.2, 1.0, g)
                col[top] = rng.uniform(0.0, 1e-4, k)
                mat[:, j] = col
        got.append(h_topk(rank_matrix(pa), rank_matrix(pb), k))
    got = np.array(got)
    assert np.all(np.diff(got) > 0)
    assert pearsonr(designs, got).statistic > 0.999


def test_ratio_cc_units():
    """跨系恰好等于同系重复时比值为 1；恰好等于随机时为 0。"""
    k, g = 288, 6642
    ch = chance_overlap(k, g)
    assert ratio_cc(0.5, 0.5, k, g) == pytest.approx(1.0)
    assert ratio_cc(ch, 0.5, k, g) == pytest.approx(0.0)
    assert np.isnan(ratio_cc(0.1, ch, k, g))


def test_fission_null_calibration():
    """真值全为 0 时，两半在 BH 下的 top-K 重叠应贴住随机基线。"""
    rng = np.random.default_rng(6)
    g, n_pert, k = 3000, 150, 100
    se = np.abs(rng.standard_normal((g, n_pert))) * 0.2 + 0.1
    bhat = rng.standard_normal((g, n_pert)) * se        # beta = 0
    b1, b2, s1, s2 = fission(bhat, se, rng)
    p1 = 2 * norm.sf(np.abs(b1 / s1))
    p2 = 2 * norm.sf(np.abs(b2 / s2))
    h = h_topk(rank_matrix(p1), rank_matrix(p2), k)
    assert h == pytest.approx(chance_overlap(k, g), abs=0.02)


@pytest.mark.skipif(not (DATA / "K562Essential_p.csv.gz").exists(),
                    reason="需要 E07 数据（gitignored）")
def test_real_files_column_alignment():
    """列对齐回归：四个文件的原始列序不同，read_matrix 必须重排成同一顺序。

    这个 bug 已经中过一次 —— 不重排会得到恰好等于随机水平的假 NO-GO。
    """
    heads = {}
    for ln in LINES:
        d = pd.read_csv(DATA / f"{ln}Essential_p.csv.gz", index_col=0, nrows=1)
        heads[ln] = [str(c) for c in d.columns]
    shared = sorted(set.intersection(*(set(h) for h in heads.values())))
    assert len(shared) > 2000

    # 原始列序确实不同，否则这个测试没有意义
    orders = [[c for c in heads[ln] if c in set(shared)] for ln in LINES]
    assert any(orders[0] != o for o in orders[1:]), "文件列序竟然一致，测试失效"

    import sys
    sys.path.insert(0, str(DATA.parents[1] / "experiments" / "E07-source-coverage"))
    from run import read_matrix

    probe = shared[:40]
    for ln in LINES:
        got = read_matrix(ln, "p", probe)
        assert got.columns.tolist() == probe, f"{ln} 列未重排"
