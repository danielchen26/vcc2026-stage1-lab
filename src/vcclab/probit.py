"""可检出性概率: 把 Stage 1 的点预测 + 不确定度 折成「这个基因会被判显著吗」.

MDE 在本问题里扮演 AdaptiveEROP 里决策阈值 tau 的角色: 打分器只在 |效应| 超过
最小可检出效应时才把基因计入 R_hat, 于是「命中显著集」= 双侧越界事件.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = ["p_detect", "gate_needs_nonparametric"]


def p_detect(
    mu_hat: np.ndarray,
    sigma_tilde: np.ndarray,
    mde_vals: np.ndarray,
) -> np.ndarray:
    """双侧可检出概率 P(|effect| >= MDE), 效应服从 N(mu_hat, sigma_tilde^2).

        p = Phi((mu - MDE) / sigma) + Phi((-MDE - mu) / sigma)

    源形式: `compute_p_success_cdf` 于 AdaptiveEROP
    `~/Documents/Decision_science_codes/dssi-decsci-assay-prediction/src/Core/p_success.jl`
    —— 那里是单侧 `Phi(d * (tau - mu_tilde) / sigma_tilde)`; 本处 tau = ±MDE
    且方向未知, 两条单侧尾相加即得双侧形式.

    sigma_tilde <= 0 视为确定性预测: 退化为指示函数 |mu| >= MDE.
    结果 clip 到 [0, 1].
    """
    mu = np.asarray(mu_hat, dtype=np.float64)
    sd = np.asarray(sigma_tilde, dtype=np.float64)
    m = np.abs(np.asarray(mde_vals, dtype=np.float64))

    mu, sd, m = np.broadcast_arrays(mu, sd, m)
    safe = sd > 0
    sd_safe = np.where(safe, sd, 1.0)
    p = norm.cdf((mu - m) / sd_safe) + norm.cdf((-m - mu) / sd_safe)
    p = np.where(safe, p, (np.abs(mu) >= m).astype(np.float64))
    return np.clip(p, 0.0, 1.0)


def gate_needs_nonparametric(
    mu_hat: np.ndarray,
    sigma_tilde: np.ndarray,
    mde_vals: np.ndarray,
    k: float = 1.0,
) -> np.ndarray:
    """哪些基因落在 MDE 边界的「中间带」, 必须走非参处理.

        |mu - MDE| / sigma < k   ->   True  (非参)
        否则                     ->   False (解析 Phi 足够, 误差 < 2%)

    源: AdaptiveEROP `.../src/Pipeline/predict.jl` 的路由规则 ——
    `distance_to_threshold = abs(tau - mu_cond) / sigma_cond; in_middle_zone =
    distance_to_threshold < 1.0`; 远离阈值时注释明确写 "Gaussian sufficient
    (diff<2%)", 近阈值时才换成学到的经验残差分布 (empirical_cdf_mc).

    sigma_tilde <= 0 (确定性预测) 一律返回 False: 无分布形状可言, 解析支即精确解.
    """
    mu = np.asarray(mu_hat, dtype=np.float64)
    sd = np.asarray(sigma_tilde, dtype=np.float64)
    m = np.asarray(mde_vals, dtype=np.float64)
    mu, sd, m = np.broadcast_arrays(mu, sd, m)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.abs(mu - m) / sd
    return np.where(sd > 0, d < k, False)
