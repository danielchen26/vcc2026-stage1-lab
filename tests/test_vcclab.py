"""vcclab 单元测试 + 一个真实数据集成测试.

跑法: `~/vcc2026/.venv/bin/python -m pytest tests/ -q`
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from scipy.stats import false_discovery_control, mannwhitneyu, norm

from vcclab import (
    ControlRef,
    bh_adjust,
    design_cells,
    gate_needs_nonparametric,
    hamilton,
    james_stein,
    ledoit_wolf_nu0,
    mde,
    p_detect,
    tail_aware_scale,
)
from vcclab.detectability import _psi, _tie_cube_sum
from vcclab.scorer import GATE_CPM, TS_CELL

CTX_A = Path("~/vcc2026/context_A.h5ad").expanduser()
GENE_CSV = Path("~/vcc2026/gene_names.csv").expanduser()


# ───────────────────────── 合成 ControlRef ─────────────────────────


def _synthetic_ref(n_ctrl=300, n_genes=40, seed=0, tmp_path=None) -> ControlRef:
    """造一个小的 h5ad, 走 ControlRef.__init__ 的真实路径 (含稀疏/零膨胀/并列)."""
    import h5py

    rg = np.random.default_rng(seed)
    # 小整数计数 -> 大量并列, 正好压 tie correction 与 psi 的退化分支
    X = rg.poisson(1.2, size=(n_ctrl, n_genes)).astype(np.float32)
    X[:, 0] = 0                     # 全零基因: 必须被 gate 排除
    X[rg.random((n_ctrl, n_genes)) < 0.3] = 0
    lib = X.sum(1)
    X[lib == 0, 1] = 1.0            # 保证每个细胞 lib > 0
    Xs = sparse.csr_matrix(X)
    genes = np.array([f"G{i}" for i in range(n_genes)])

    path = Path(tmp_path) / f"synth_{seed}.h5ad"
    with h5py.File(path, "w") as f:
        g = f.create_group("X")
        g.create_dataset("data", data=Xs.data.astype(np.float32))
        g.create_dataset("indices", data=Xs.indices.astype(np.int32))
        g.create_dataset("indptr", data=Xs.indptr.astype(np.int32))
        g.attrs["shape"] = np.array(Xs.shape, dtype=np.int64)
        f.create_group("var").create_group("_index").create_dataset(
            "values", data=np.array([s.encode() for s in genes])
        )
    return ControlRef.load(path, genes)


@pytest.fixture(scope="module")
def synth_ref(tmp_path_factory):
    return _synthetic_ref(tmp_path=tmp_path_factory.mktemp("h5ad"))


# ───────────────────────────── psi ─────────────────────────────


def test_psi_matches_scipy(synth_ref):
    """U = sum_i psi_j(v_i) 必须逐位等于 scipy.mannwhitneyu 的 statistic."""
    ref = synth_ref
    assert ref.G > 3
    rg = np.random.default_rng(7)
    ctrl = np.asarray(ref._cpm_csr.todense())

    checked = 0
    for j in range(min(ref.G, 8)):
        col = ctrl[:, ref.gidx[j]]
        cases = [
            rg.choice(col, 25, replace=True),                  # 从对照里抽 -> 大量并列
            np.full(25, 0.0),                                  # 退化: 全为 0
            np.full(25, float(np.median(col))),                # 退化: 全等于对照中位数
            np.full(25, 12345.0),                              # 退化: 全大于所有对照
            rg.uniform(0, col.max() + 1, 25),                   # 连续值 -> 无并列
        ]
        for v in cases:
            v = np.asarray(v, dtype=np.float64)
            u_psi = ref.psi(j, v).sum()
            u_scipy = mannwhitneyu(v, col, alternative="two-sided").statistic
            assert u_psi == pytest.approx(u_scipy, abs=1e-6), (j, v[:3])
            checked += 1
    assert checked >= 20


def test_psi_zero_case_is_exact(synth_ref):
    """v == 0 分支: psi 必须是 0.5 * #{ctrl == 0}, 不能落到 searchsorted 的通路."""
    ref = synth_ref
    ctrl = np.asarray(ref._cpm_csr.todense())
    for j in range(min(ref.G, 5)):
        nz = (ctrl[:, ref.gidx[j]] == 0).sum()
        assert ref.psi(j, np.zeros(3))[0] == pytest.approx(0.5 * nz)


def test_detectability_psi_fast_path_is_faithful(synth_ref):
    """detectability._psi (float64 预升位快路径) 必须与 ControlRef.psi 逐位相等."""
    ref = synth_ref
    rg = np.random.default_rng(31)
    ctrl = np.asarray(ref._cpm_csr.todense())
    for j in range(min(ref.G, 8)):
        col64 = ref._sorted[j].astype(np.float64)
        nz = float(ref._nzero[j])
        for v in (
            np.rint(rg.choice(ctrl[:, ref.gidx[j]], 40, replace=True) * 1.7),
            np.zeros(40),                       # 全零: 走特殊分支
            np.full(40, 9e9),                   # 全大于对照
        ):
            assert np.array_equal(_psi(col64, nz, v), ref.psi(j, v.copy()))


def test_tie_decomposition_matches_de_table(synth_ref):
    """detectability 的 T 可加分解必须与 de_table 内联的 np.unique 版本逐位相等."""
    ref = synth_ref
    rg = np.random.default_rng(3)
    ctrl = np.asarray(ref._cpm_csr.todense())
    for j in range(min(ref.G, 6)):
        col64 = ref._sorted[j].astype(np.float64)
        nz = float(ref._nzero[j])
        t_ctrl = ref.tie_cube_sum_ctrl(j)
        for mult in (1.3, 1.0, 0.0):            # 0.0 -> 样本全为 0, 压零值并列组
            v = np.rint(rg.choice(ctrl[:, ref.gidx[j]], 30, replace=True) * mult)
            allv = np.concatenate(
                [
                    np.zeros(ref._nzero[j], np.float32),
                    ref._sorted[j],
                    v.astype(np.float32),
                ]
            )
            _, c = np.unique(allv, return_counts=True)
            t_ref = float(np.sum(c.astype(np.float64) ** 3 - c))
            assert _tie_cube_sum(col64, nz, t_ctrl, v) == pytest.approx(
                t_ref, rel=1e-12
            )


# ───────────────────────────── BH ─────────────────────────────


def _bh_wrong(p):
    """故意写错的版本: max_{j>=i} 而不是 min_{j>=i}. 作为回归对照留在测试里."""
    m = len(p)
    order = np.argsort(p)
    q = p[order] * m / np.arange(1, m + 1)
    q = np.maximum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def test_bh_direction():
    rg = np.random.default_rng(11)
    # 250 个真信号 + 9750 个 null, 模拟一次真实 DE 的 p 值谱
    p = np.concatenate(
        [rg.uniform(0, 1e-8, 250), rg.uniform(0, 1, 9750)]
    )
    ours = bh_adjust(p)
    ref = false_discovery_control(p, method="bh")
    assert np.allclose(ours, ref, atol=1e-12)

    # 回归: 用 maximum.accumulate 会让发现数塌到 ~0
    n_ok = int((ours < 0.05).sum())
    n_bad = int((_bh_wrong(p) < 0.05).sum())
    assert n_ok >= 250
    assert n_bad < n_ok / 10, (n_ok, n_bad)


def test_bh_monotone_and_bounded():
    rg = np.random.default_rng(12)
    p = rg.uniform(0, 1, 500)
    q = bh_adjust(p)
    assert q.min() >= 0.0 and q.max() <= 1.0
    o = np.argsort(p)
    assert np.all(np.diff(q[o]) >= -1e-15)      # step-up -> 排序后单调不减


# ─────────────────────────── hamilton ───────────────────────────


def test_hamilton_row_sum():
    rg = np.random.default_rng(5)
    for seed in range(6):
        rg = np.random.default_rng(seed)
        row = rg.random(18_533) ** 4
        row[rg.random(18_533) < 0.85] = 0.0     # 单细胞式稀疏
        row *= TS_CELL / row.sum()
        out = hamilton(row)
        assert out.sum() == 1_000_000
        assert np.all(out >= 0)
        assert np.all(out == np.floor(out))
        # 每格与目标至多差 1: 最大余数法的定义性质
        assert np.max(np.abs(out - row)) < 1.0 + 1e-9


def test_hamilton_handles_over_and_under_shoot():
    # floor 之和已超过 total -> 走 need < 0 的减法分支
    row = np.full(4, 250_001.0)
    out = hamilton(row)
    assert out.sum() == 1_000_000 and np.all(out >= 0)
    # floor 之和远小于 total -> 走 need > 0 的加法分支
    row = np.full(4, 249_999.5)
    out = hamilton(row)
    assert out.sum() == 1_000_000 and np.all(out >= 0)


# ──────────────────────────── probit ────────────────────────────


def test_probit_bounds():
    rg = np.random.default_rng(9)
    mu = rg.normal(0, 2, 5000)
    sd = rg.uniform(0.05, 3.0, 5000)
    m = rg.uniform(0.02, 1.5, 5000)
    p = p_detect(mu, sd, m)
    assert p.shape == mu.shape
    assert np.all(p >= 0.0) and np.all(p <= 1.0)
    assert np.all(np.isfinite(p))

    # mu = 0, MDE 远大于 sigma -> 概率极小
    assert p_detect(0.0, 0.1, 1.0) < 1e-15
    # |mu| >> MDE -> 趋于 1, 双向对称
    assert p_detect(10.0, 0.1, 1.0) == pytest.approx(1.0, abs=1e-12)
    assert p_detect(-10.0, 0.1, 1.0) == pytest.approx(1.0, abs=1e-12)
    assert p_detect(3.0, 0.5, 1.0) == pytest.approx(p_detect(-3.0, 0.5, 1.0))
    # mu 恰在 MDE 上 -> ~1/2 (加上另一侧可忽略的尾)
    assert p_detect(1.0, 0.1, 1.0) == pytest.approx(0.5, abs=1e-6)
    # sigma = 0 -> 指示函数
    assert p_detect(1.5, 0.0, 1.0) == 1.0
    assert p_detect(0.5, 0.0, 1.0) == 0.0
    # 与闭式对照
    assert p_detect(0.3, 0.7, 0.45) == pytest.approx(
        norm.cdf((0.3 - 0.45) / 0.7) + norm.cdf((-0.45 - 0.3) / 0.7)
    )


def test_gate_needs_nonparametric():
    mu = np.array([0.20, 1.00, 5.00, 0.30])
    sd = np.array([0.10, 0.10, 0.10, 0.00])
    m = np.array([0.25, 0.25, 0.25, 0.25])
    g = gate_needs_nonparametric(mu, sd, m)
    assert g.dtype == np.bool_
    assert g.tolist() == [True, False, False, False]
    # k 放大 -> 中间带变宽, 单调
    assert int(gate_needs_nonparametric(mu, sd, m, k=100.0).sum()) >= int(g.sum())


# ─────────────────────────── shrinkage ───────────────────────────


def test_james_stein_shrinks():
    rg = np.random.default_rng(13)
    r_bar = rg.normal(0, 1, 1000)
    s2g = rg.uniform(0.1, 4.0, 1000)
    n_c = 400

    assert np.allclose(james_stein(r_bar, s2g, 0.0, n_c), 0.0)
    assert np.allclose(james_stein(r_bar, s2g, np.inf, n_c), r_bar)
    assert np.allclose(james_stein(r_bar, s2g, 1e-12, n_c), 0.0, atol=1e-8)
    assert np.allclose(james_stein(r_bar, s2g, 1e18, n_c), r_bar, rtol=1e-10)

    # 闭式一致 + 收缩强度对 sigma2_between 单调
    b1 = james_stein(r_bar, s2g, 0.01, n_c)
    b2 = james_stein(r_bar, s2g, 1.00, n_c)
    assert np.allclose(b2, r_bar * n_c * 1.0 / (s2g + n_c * 1.0))
    assert np.all(np.abs(b1) <= np.abs(b2) + 1e-12)
    assert np.all(np.abs(b2) <= np.abs(r_bar) + 1e-12)
    # 方向从不翻转
    assert np.all(np.sign(b2[r_bar != 0]) == np.sign(r_bar[r_bar != 0]))


def test_ledoit_wolf_nu0():
    rg = np.random.default_rng(17)
    p, n, n_c = 6, 60, 400
    A = rg.normal(0, 1, (p, p))
    sigma = A @ A.T / p + np.eye(p)
    R = rg.multivariate_normal(np.zeros(p), sigma, n)

    a, nu0 = ledoit_wolf_nu0(R, sigma, n_c)
    assert 0.01 <= a <= 0.99
    assert nu0 == pytest.approx(max((p + 1) + a * n_c / (1 - a), p + 2))
    assert nu0 >= p + 2

    # 目标完全等于样本协方差 -> 分母 ~0 -> 退回 0.5
    D = R - R.mean(0)
    S = D.T @ D / n
    a_deg, _ = ledoit_wolf_nu0(R, S, n_c)
    assert a_deg == pytest.approx(0.5) or a_deg == pytest.approx(0.99)

    # NaN (缺失) 不应产生 nan 结果
    R2 = R.copy()
    R2[rg.random(R2.shape) < 0.2] = np.nan
    a2, nu2 = ledoit_wolf_nu0(R2, sigma, n_c)
    assert np.isfinite(a2) and np.isfinite(nu2)

    # 目标离样本很远 -> 收缩很弱 (alpha 触下界)
    a_far, _ = ledoit_wolf_nu0(R, sigma * 1e6, n_c)
    assert a_far == pytest.approx(0.01)

    with pytest.raises(ValueError):
        ledoit_wolf_nu0(R, np.eye(p + 1), n_c)


def test_tail_aware_scale():
    rg = np.random.default_rng(19)
    # 轻尾高斯: tail_index ~ 1 -> 走 MAD 支, 恢复 sigma
    x = rg.normal(0, 2.0, 200_000)
    assert tail_aware_scale(x) == pytest.approx(2.0, rel=0.03)

    # 重尾: 少量极端值把 std/MAD 抬到 > 3 -> 换 p98-p02 支, 不再被离群值主导
    y = rg.normal(0, 1.0, 20_000)
    y[:40] = 500.0
    s = tail_aware_scale(y)
    assert s < 0.2 * np.std(y)
    assert s == pytest.approx(1.0, rel=0.15)

    # 逐行, 保持形状
    Z = rg.normal(0, 1.0, (7, 5000))
    out = tail_aware_scale(Z, axis=-1)
    assert out.shape == (7,)
    assert np.all(np.abs(out - 1.0) < 0.1)
    assert np.all(tail_aware_scale(Z, axis=0) > 0)

    # MAD == 0 (常数向量) 不得返回 0 之外的怪值, 也不得抛
    assert tail_aware_scale(np.zeros(100)) == pytest.approx(0.0)


# ───────────────────── 合成数据上的端到端 ─────────────────────


def test_design_cells_synthetic_roundtrip(synth_ref):
    """在合成 ref 上跑 Stage 2 -> DE 回读: 行和守恒, 方向正确, 无假阳性."""
    ref = synth_ref
    rg = np.random.default_rng(21)
    n_r = max(2, ref.G // 4)
    R = rg.choice(ref.G, n_r, replace=False)
    lfc = rg.choice([-1.0, 1.0], n_r) * rg.uniform(1.0, 2.5, n_r)

    C = design_cells(ref, R, lfc, n_cells=120, seed=1)
    assert C.shape == (120, ref.n_genes)
    assert np.all(C.sum(1) == 1_000_000)
    assert np.all(C >= 0) and np.all(C == np.floor(C))

    padj, lf = ref.de_table(C)
    assert padj.shape == (ref.G,) and lf.shape == (ref.G,)
    hit = np.flatnonzero(padj < 0.05)
    intended = np.zeros(ref.G, bool)
    intended[R] = True
    realized = np.zeros(ref.G, bool)
    realized[hit] = True
    assert np.all(np.sign(lf[R]) == np.sign(lfc))       # 方向 100%
    assert int((intended & realized).sum()) == n_r      # 召回 100%
    assert int((~intended & realized).sum()) == 0       # 假阳性 0


def test_mde_synthetic(synth_ref):
    """MDE 的定义性质: 非负; 有并列校正的阈值更严 -> MDE 不小于未校正版."""
    ref = synth_ref
    k = min(ref.G, 12)
    idx = np.arange(k)
    m_tc = mde(ref, idx, n_cells=120, seed=2, tie_correct=True)
    m_nc = mde(ref, idx, n_cells=120, seed=2, tie_correct=False)
    assert m_tc.shape == (k,) and m_nc.shape == (k,)
    ok = np.isfinite(m_tc) & np.isfinite(m_nc)
    assert ok.sum() >= k // 2
    assert np.all(m_tc[ok] >= 0.0)
    # 并列校正把 sigma 变小 -> 阈值变松 -> MDE 不会变大
    assert np.all(m_tc[ok] <= m_nc[ok] + 1e-6)
    # 默认 gene_idx=None 覆盖全部 gate
    assert mde(ref, n_cells=120, seed=2, tie_correct=False).shape == (ref.G,)


def test_mde_boundary_is_tight(synth_ref):
    """返回值就是显著性边界: 稍微加大 |lfc| 显著, 稍微减小则不显著."""
    ref = synth_ref
    n1, n2 = 120, ref.n_ctrl
    N = n1 + n2
    z = norm.isf(0.025)
    rg = np.random.default_rng(2)
    cells = rg.choice(n2, n1, replace=True)
    V = np.asarray(ref._cpm_csr[cells][:, ref.gidx].todense(), dtype=np.float64)
    sd = np.sqrt(n1 * n2 * (N + 1) / 12.0)
    thr = z * sd / (n1 * n2)
    vals = mde(ref, n_cells=n1, seed=2, tie_correct=False)

    def dev(j, lfc):
        v = np.rint(V[:, j] * 2.0**lfc)
        return abs(ref.psi(j, v).mean() / n2 - 0.5)

    tested = 0
    for j in np.flatnonzero(np.isfinite(vals) & (vals > 1e-3))[:6]:
        assert dev(j, vals[j] * 1.02) > thr
        assert dev(j, vals[j] * 0.98) <= thr
        tested += 1
    assert tested >= 1


# ─────────────────────── 真实数据集成测试 ───────────────────────


@pytest.mark.skipif(not CTX_A.exists(), reason="官方 context_A.h5ad 不在本机")
def test_context_a_integration():
    import pandas as pd

    genes = pd.read_csv(GENE_CSV)["gene_name"].to_numpy()
    ref = ControlRef.load(CTX_A, genes)
    assert ref.n_ctrl == 18_400
    assert ref.n_genes == 18_533
    assert ref.G == 9929
    assert abs(ref.m_full.sum() - 1e6) < 1
    assert np.all(ref.m_gate > GATE_CPM)
    assert ref.m_gate.shape == (9929,)
    assert np.array_equal(ref.m_gate, ref.m_full[ref.gidx])
    assert len(ref._sorted) == ref.G
    assert int(ref._nzero[0]) + ref._sorted[0].size == ref.n_ctrl
