"""召集集合（call set）：从带噪的效应量估计到「该报哪些基因、报多少个」。

这是 TAP 框架的第 1 个模块。理论在 docs/08-framework.md 第 5 节，
两个承重命题已在 experiments/E09-framework-theory/verify.py 数值验证。

## 为什么不能直接拿 beta_hat 过阈值

beta_hat 是带噪估计，噪声 SD = se。真值 beta = 0 的基因也有
2*Phi(-t/se) 的概率越过阈值 t。在 se=0.199、t=0.30、8248 个基因下，
这是约 1080 个纯噪声假阳性 —— 实测确实得到 981（见 E08 的 run_erp.py）。
必须对测量误差做反卷积。

## 反卷积：两成分经验贝叶斯

    beta_g ~ pi0 * delta_0 + (1 - pi0) * N(0, tau^2)
    beta_hat_g | beta_g ~ N(beta_g, se_g^2)          se_g 已知

边缘似然（se 逐基因不同，所以不能只看 z 的方差）：

    L = prod_g [ pi0 * N(bh_g; 0, se_g^2) + (1-pi0) * N(bh_g; 0, se_g^2 + tau^2) ]

后验越阈概率（零成分贡献 0，因为 beta 恰为 0 而 t > 0）：

    P(|beta_g| > t | bh_g) = P(非零|bh_g) * [ Phi((-t-mu)/sqrt(v)) + Phi((mu-t)/sqrt(v)) ]
    mu = bh * tau^2/(tau^2+se^2),   v = tau^2 se^2/(tau^2+se^2)

## 两个承重命题

命题 1（最优召集）：固定 |R̂| = K 时，E|R ∩ R̂| = sum_{g in R̂} p_g 对成员指示是
    线性的，故取 p_g 最大的 K 个。条件在扰动强度上（给定 |R| = n）时精确最优。

命题 2（停止规则）：Jaccard J = m/(n+K-m)，M(K) = sum_{i<=K} p_i（p 降序），则
    dE[J]/dK = 0  <=>  p_{K*} = M(K*)/(n+K*) = J/(1+J) = h/2
    即「只要下一个基因的检出概率高于当前已达 h 的一半，就继续加」。
    官方锚点核对：J=0.399 -> h=0.570 -> p* = 0.285 = h/2 ✓

    **闭式解在 p 曲线平滑时成立（误差 <5%），在阶跃处失效（68%）**，因为推导假设
    p 对 K 可微、而阶跃时最优点落在悬崖上而非导数零点。阶跃对应「已知 R_p」，
    平滑对应「预测不确定」。我们的 p_g 是 probit 出来的连续量，必然平滑，所以
    K* > E|R_p|：不确定性越大该召集越多（Jaccard 分母只按加进去的假阳性增长，
    而每个 p > J/(1+J) 的基因仍净赚）。

    **实现里不用闭式**，直接数值最大化 E[J]，自动同时覆盖两种形状。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm

__all__ = [
    "fit_prior",
    "p_exceed",
    "expected_jaccard_curve",
    "best_k",
    "stopping_threshold",
    "identifiability",
    "call_set",
]

_EPS = 1e-300


def _mix_densities(bh: np.ndarray, v0: np.ndarray, tau2: float
                   ) -> tuple[np.ndarray, np.ndarray]:
    """两个成分的（未归一化）密度。常数因子在后验里约掉，故省去 1/sqrt(2pi)。"""
    f0 = np.exp(-0.5 * bh * bh / v0) / np.sqrt(v0)
    f1 = np.exp(-0.5 * bh * bh / (v0 + tau2)) / np.sqrt(v0 + tau2)
    return f0, f1


def _profile_pi0(f0: np.ndarray, f1: np.ndarray, n_iter: int = 200) -> float:
    """给定 tau 时 pi0 的剖面解。EM 不动点迭代，凹问题，收敛很快。"""
    pi0 = 0.9
    for _ in range(n_iter):
        w = pi0 * f0 / np.maximum(pi0 * f0 + (1.0 - pi0) * f1, _EPS)
        pi0 = float(np.clip(w.mean(), 1e-4, 1.0 - 1e-6))
    return pi0


def fit_prior(bhat: np.ndarray, se: np.ndarray,
              tau_lo: float = 1e-3, tau_hi: float = 5.0) -> tuple[float, float]:
    """极大边缘似然拟合 (pi0, tau)。

    对 tau 做一维有界搜索，pi0 用剖面解 —— 两个参数、上万个观测，极好定。

    Parameters
    ----------
    bhat, se : 同形状一维数组。se 必须为正（= lfcSE），且**必须是可信的**：
        整个反卷积的正确性建立在「se 已知且正确」上。se 被系统性低估会让
        pi0 偏低（把噪声当信号）；se 极大的基因对似然几无贡献，会被吸进零成分
        —— 后者在低测序深度的源数据上是真实风险，见 docs/02-findings.md F20。

    Returns
    -------
    (pi0, tau)
    """
    bh = np.asarray(bhat, dtype=np.float64)
    s = np.asarray(se, dtype=np.float64)
    if bh.shape != s.shape:
        raise ValueError(f"bhat 与 se 形状不一致: {bh.shape} vs {s.shape}")
    ok = np.isfinite(bh) & np.isfinite(s) & (s > 0)
    if ok.sum() < 10:
        raise ValueError(f"可用观测太少: {int(ok.sum())}")
    bh, s = bh[ok], s[ok]
    v0 = s * s

    def negll(log_tau: float) -> float:
        tau2 = float(np.exp(2.0 * log_tau))
        f0, f1 = _mix_densities(bh, v0, tau2)
        pi0 = _profile_pi0(f0, f1, n_iter=60)
        mix = pi0 * f0 + (1.0 - pi0) * f1
        return -float(np.sum(np.log(np.maximum(mix, _EPS))))

    r = minimize_scalar(negll, bounds=(np.log(tau_lo), np.log(tau_hi)),
                        method="bounded", options={"xatol": 1e-3})
    tau = float(np.exp(r.x))
    f0, f1 = _mix_densities(bh, v0, tau * tau)
    return _profile_pi0(f0, f1), tau


def p_exceed(bhat: np.ndarray, se: np.ndarray, thr: np.ndarray,
             pi0: float, tau: float) -> np.ndarray:
    """P(|beta| > thr | beta_hat)，逐基因。thr 可逐基因不同（目标 context 的 MDE）。"""
    bh = np.asarray(bhat, dtype=np.float64)
    s = np.asarray(se, dtype=np.float64)
    t = np.abs(np.asarray(thr, dtype=np.float64))
    v0, tau2 = s * s, tau * tau
    f0, f1 = _mix_densities(bh, v0, tau2)
    p_alt = (1.0 - pi0) * f1 / np.maximum(pi0 * f0 + (1.0 - pi0) * f1, _EPS)
    shrink = tau2 / (tau2 + v0)
    mu = bh * shrink
    sd = np.sqrt(np.maximum(tau2 * v0 / (tau2 + v0), _EPS))
    return p_alt * (norm.sf((t - mu) / sd) + norm.cdf((-t - mu) / sd))


def expected_jaccard_curve(p: np.ndarray, n_true: float | None = None
                           ) -> tuple[np.ndarray, np.ndarray]:
    """按 p 降序逐个加入时的 E[J] 曲线。

    E[J] ~ M(K) / (n + K - M(K))，其中 M(K) = 累积检出概率。
    `n_true` 省略时用 sum(p) —— 即模型自己对 |R_p| 的期望，自洽。

    返回 (K 数组, E[J] 数组)。
    """
    ps = np.sort(np.asarray(p, dtype=np.float64))[::-1]
    n = float(np.sum(ps)) if n_true is None else float(n_true)
    m = np.cumsum(ps)
    k = np.arange(1, ps.size + 1, dtype=np.float64)
    return k, m / np.maximum(n + k - m, 1e-12)


def best_k(p: np.ndarray, n_true: float | None = None) -> tuple[int, float]:
    """数值最大化 E[J] 定出最优召集集合大小。返回 (K*, E[J] 上限)。

    不用命题 2 的闭式 p_K* = J/(1+J)：闭式假设 p 对 K 可微，阶跃时误差达 68%。
    数值法代价可忽略（一次累加 + argmax），且两种形状都对。
    """
    k, j = expected_jaccard_curve(p, n_true)
    i = int(np.argmax(j))
    return int(k[i]), float(j[i])


def stopping_threshold(j: float) -> float:
    """命题 2 的闭式停止阈值 p* = J/(1+J) = h/2。仅用于对照与诊断。"""
    return j / (1.0 + j)


def identifiability(se: np.ndarray, tau: float) -> dict:
    """反卷积可用性的守门指标 tau / median(se)。**必须随每次拟合一起报出。**

    实测（真值 322 个基因越阈 0.30，tau=0.5，20k 基因）：

        tau/se = 2.78  →  E|R_p| = 315.5   良态，准确
        tau/se = 1.39  →            331.3
        tau/se = 0.93  →            648.3   开始高估
        tau/se = 0.56  →            984.0   严重高估
        tau/se = 0.35  →              0.0   彻底崩溃

    机制：tau/se 变小时两成分方差趋同（N(0,se^2) vs N(0,se^2+tau^2)），
    (pi0, tau) 不可辨识，估计可往任何方向跑。**不是单调低估** ——
    低深度源数据的危险在于**高估**，不是漏检。

    K562 GWPS: tau/se = 0.495/0.199 = 2.49 → 良态。
    """
    med = float(np.median(np.asarray(se, dtype=np.float64)))
    ratio = tau / med if med > 0 else np.inf
    if ratio >= 2.0:
        verdict = "良态"
    elif ratio >= 1.2:
        verdict = "边缘"
    else:
        verdict = "不可辨识：E|R_p| 可能严重高估或崩溃，不要采用"
    return {"tau_over_se": ratio, "median_se": med, "verdict": verdict}


def call_set(bhat: np.ndarray, se: np.ndarray, thr: np.ndarray,
             pi0: float | None = None, tau: float | None = None,
             n_true: float | None = None
             ) -> tuple[np.ndarray, np.ndarray, dict]:
    """端到端：效应量估计 + 逐基因阈值 → 召集集合。

    Returns
    -------
    idx : 召集的基因下标（按 p 降序）
    p : 逐基因检出概率（原始顺序）
    info : {'pi0', 'tau', 'K', 'E_J', 'E_Rp', 'p_stop_closed_form',
            'tau_over_se', 'identifiability'}
    """
    if pi0 is None or tau is None:
        pi0, tau = fit_prior(bhat, se)
    p = p_exceed(bhat, se, thr, pi0, tau)
    k, ej = best_k(p, n_true)
    order = np.argsort(p)[::-1]
    ident = identifiability(se, tau)
    return order[:k], p, {
        "pi0": pi0, "tau": tau, "K": k, "E_J": ej,
        "E_Rp": float(np.sum(p)),
        "p_stop_closed_form": stopping_threshold(ej),
        "tau_over_se": ident["tau_over_se"],
        "identifiability": ident["verdict"],
    }
