"""decoder 的 lfc_all 通道，以及一个**实测的物理约束**。

## 结论先行：lfc 与显著性在本届条件下**不可独立设定**

docstring 里说「显著性(psi_bar) 与方向(一阶矩) 完全解耦」——那是指两点分布构造里
可以固定均值而调 psi_bar。但在 400 提交细胞 vs 18,400 对照细胞的 Wilcoxon 下：

    给非召集基因赋 lfc ~ N(0, 0.15^2)  →  1,327 个被判显著（意图 250）
    给非召集基因赋 lfc ~ N(0, 0.50^2)  →  5,083 个被判显著

且对**下调**基因，「有 lfc 但不显著」在两点分布下**不可达**：psi_bar 的上界是
P(ctrl < mu)，下调时该上界本身就 < 0.5，永远越不回不显著区。

**所以这是物理约束，不是代码缺陷。** 六个计分指标里 5 个读连续谱、1 个读显著集，
两者通过 Wilcoxon 的灵敏度耦合在一起 —— 目标函数存在真实的内部张力，
最优解是取舍而非兼得。这几条测试把该约束钉成已知行为。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vcclab.decoder import design_cells  # noqa: E402
from vcclab.scorer import ControlRef  # noqa: E402

VCC = Path.home() / "vcc2026"
ALPHA = 0.05


@pytest.fixture(scope="module")
def ref():
    import pandas as pd
    genes = pd.read_csv(VCC / "gene_names.csv")["gene_name"].tolist()
    return ControlRef.load(VCC / "context_A.h5ad", genes)


def _cpm(counts):
    s = counts.sum(1, keepdims=True)
    return np.divide(counts, s, out=np.zeros_like(counts), where=s > 0) * 1e6


def test_intended_set_is_always_fully_recalled(ref):
    """无论是否传 lfc_all，意图显著集的**召回**必须是 100%（漏 0 个）。

    这是解码器的核心保证，E03 已在无 lfc_all 时验证。加 lfc_all 后仍须成立。
    """
    rng = np.random.default_rng(0)
    r_set = np.sort(rng.choice(ref.G, 250, replace=False))
    lfc = rng.choice([-1.0, 1.0], 250) * rng.uniform(0.8, 1.5, 250)
    for la in (None, rng.standard_normal(ref.G) * 0.15):
        counts = design_cells(ref, r_set, lfc, n_cells=400, seed=1, lfc_all=la)
        padj, _ = ref.de_table(_cpm(counts), tie_correct=True)
        sig = set(np.flatnonzero(padj < ALPHA).tolist())
        missed = set(r_set.tolist()) - sig
        assert not missed, f"漏判 {len(missed)} 个意图基因（lfc_all={'有' if la is not None else '无'}）"


def test_lfc_all_realizes_intended_lfc(ref):
    """传入的 lfc_all 必须被实现（在成分居中的意义下），相关 > 0.9、斜率近 1。"""
    rng = np.random.default_rng(2)
    r_set = np.sort(rng.choice(ref.G, 100, replace=False))
    lfc_all = rng.standard_normal(ref.G) * 0.3
    counts = design_cells(ref, r_set, np.full(100, 1.2), n_cells=400, seed=3,
                          lfc_all=lfc_all)
    _, got = ref.de_table(_cpm(counts), tie_correct=True)
    other = np.setdiff1d(np.arange(ref.G), r_set)
    w = ref.m_gate / ref.m_gate.sum()
    intent = lfc_all - float(w @ lfc_all)          # 与 decoder 同一居中
    r = np.corrcoef(intent[other], got[other])[0, 1]
    slope = np.polyfit(intent[other], got[other], 1)[0]
    assert r > 0.9, f"实际 lfc 与意图相关只有 {r:.3f}"
    assert 0.7 < slope < 1.3, f"斜率 {slope:.3f} 偏离 1 太多"


@pytest.mark.parametrize("sd,lo", [(0.15, 300), (0.50, 2000)])
def test_lfc_coupling_is_real_and_monotone(ref, sd, lo):
    """**实测约束**：给非召集基因赋 lfc 会让大量基因变显著，且随 lfc 尺度单调增。

    这不是 bug —— Wilcoxon 在 400 vs 18,400 下的 d_crit 极小。钉住它，
    以免未来把「加了 lfc 就多出显著基因」当成回归。
    """
    rng = np.random.default_rng(4)
    r_set = np.sort(rng.choice(ref.G, 50, replace=False))
    counts = design_cells(ref, r_set, np.full(50, 1.0), n_cells=400, seed=5,
                          lfc_all=rng.standard_normal(ref.G) * sd)
    padj, _ = ref.de_table(_cpm(counts), tie_correct=True)
    extra = np.setdiff1d(np.flatnonzero(padj < ALPHA), r_set).size
    assert extra >= lo, (
        f"lfc 尺度 {sd} 只多出 {extra} 个显著基因，低于实测下界 {lo} —— "
        f"若确实改进了解耦机制，请更新本测试并记录")


def test_lfc_all_none_matches_old_behavior(ref):
    """不传 lfc_all 时行为与改动前逐位一致（回归保护）。"""
    rng = np.random.default_rng(6)
    r_set = np.sort(rng.choice(ref.G, 80, replace=False))
    lfc = rng.choice([-1.0, 1.0], 80)
    a = design_cells(ref, r_set, lfc, n_cells=200, seed=7)
    b = design_cells(ref, r_set, lfc, n_cells=200, seed=7, lfc_all=None)
    np.testing.assert_array_equal(a, b)


def test_lfc_all_shape_is_validated(ref):
    with pytest.raises(ValueError, match="lfc_all 形状"):
        design_cells(ref, np.arange(10), np.ones(10), n_cells=50,
                     lfc_all=np.zeros(ref.G + 1))


def test_lfc_all_nonfinite_is_treated_as_zero(ref):
    """源侧缺测（NaN）不应污染 profile，且行和必须恰为 1e6。"""
    rng = np.random.default_rng(8)
    r_set = np.sort(rng.choice(ref.G, 40, replace=False))
    la = rng.standard_normal(ref.G) * 0.2
    la[rng.choice(ref.G, ref.G // 3, replace=False)] = np.nan
    counts = design_cells(ref, r_set, np.ones(40), n_cells=200, seed=9, lfc_all=la)
    assert np.isfinite(counts).all()
    assert (counts.sum(1) == 1_000_000).all()
