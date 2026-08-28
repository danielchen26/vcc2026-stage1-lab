"""AdaptiveEROP 的收缩公式, 用 numpy 重写 (源为只读参考的 Julia 实现).

Stage 1 要在**没有任何目标细胞系扰动答案**的前提下给出 18533 维的响应向量.
每个 (扰动, 细胞系) 只有 400 个细胞, 而基因数 p ~ 1e4 >> n_c, 逐基因的
样本量–维度比极差: 裸经验估计的方差主导误差. 这正是收缩估计的用武之地 ——
把「本组自己的噪声估计」朝「全局骨架」按解析最优强度拉回.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ledoit_wolf_nu0", "james_stein", "tail_aware_scale"]

_MAD_TO_SIGMA = 1.4826            # 1 / Phi^{-1}(0.75)
_P98_Z = 2.0537489                # Phi^{-1}(0.98)


def ledoit_wolf_nu0(
    residuals: np.ndarray,
    sigma_global: np.ndarray,
    n_c: int,
) -> tuple[float, float]:
    """Ledoit-Wolf 最优收缩强度 alpha*, 以及等价的 Inverse-Wishart 自由度 nu0.

    公式 (源: `~/Documents/Decision_science_codes/dssi-decsci-assay-prediction/
    src/HierarchicalBayes/inverse_wishart.jl`, 函数 `ledoit_wolf_shrinkage`
    与 `compute_nu0_from_shrinkage`):

        D        = residuals - r_bar                     (逐列去心, 缺失按成对可得)
        s_ab     = sum_i D_ia D_ib / n_ab                (成对样本协方差)
        var_ab   = (sum_i (D_ia D_ib)^2 / n_ab - s_ab^2) / n_ab
        alpha*   = clip( sum_ab max(var_ab, 0) / sum_ab (s_ab - F_ab)^2, 0.01, 0.99 )
        nu0      = max( (p + 1) + alpha* * n_c / (1 - alpha*),  p + 2 )

    其中 F = sigma_global 是收缩目标. 分母 < 1e-12 时 alpha* 退回 0.5.
    IW 后验 Sigma_c = w*Sigma_global + (1-w)*S_c 与 LW 线性收缩同构,
    w = (nu0-p-1)/(nu0+n_c-p-1); 令 w = alpha* 反解即得上面的 nu0.

    p 由 `residuals` 的列数推出. `residuals` 中的 NaN 视为缺失.

    为何适用于本问题: 一个 (扰动, 细胞系) 组只有 n_c = 400 个细胞, 而基因维 p 上万,
    样本协方差 S_c 严重奇异. alpha* 用 s_ab 的抽样方差与「S_c 离骨架多远」的比值
    自动定标 —— 噪声大就多收缩, 信号真就少收缩; 转成 nu0 后可以直接喂给
    Inverse-Wishart 先验, 让「细胞多的 context 信自己, 细胞少的信全局」自动成立.

    Returns
    -------
    (alpha_star, nu0)
    """
    R = np.asarray(residuals, dtype=np.float64)
    if R.ndim != 2:
        raise ValueError("residuals 必须是 (n, p) 二维数组")
    n, p = R.shape
    F = np.asarray(sigma_global, dtype=np.float64)
    if F.shape != (p, p):
        raise ValueError(f"sigma_global 应为 ({p}, {p}), 收到 {F.shape}")

    mask = np.isfinite(R)
    M = mask.astype(np.float64)
    with np.errstate(invalid="ignore"):
        cnt_col = M.sum(0)
        r_bar = np.where(cnt_col > 0, np.nansum(np.where(mask, R, 0.0), 0) / np.maximum(cnt_col, 1), 0.0)

    D = np.where(mask, R - r_bar, 0.0)
    counts = M.T @ M                      # n_ab: 成对可得样本数
    sum1 = D.T @ D                        # sum_i D_ia D_ib
    sum_sq = (D * D).T @ (D * D)          # sum_i (D_ia D_ib)^2

    valid = counts > 1
    nc = np.where(valid, counts, 1.0)
    s = sum1 / nc
    var = (sum_sq / nc - s * s) / nc

    numerator = float(np.sum(np.where(valid, np.maximum(var, 0.0), 0.0)))
    denominator = float(np.sum(np.where(valid, (s - F) ** 2, 0.0)))

    if denominator < 1e-12:
        alpha_star = 0.5
    else:
        alpha_star = float(np.clip(numerator / denominator, 0.01, 0.99))

    nu0 = max((p + 1) + alpha_star * n_c / (1.0 - alpha_star), float(p + 2))
    return alpha_star, nu0


def james_stein(
    r_bar: np.ndarray,
    sigma2_gene: np.ndarray,
    sigma2_between: float,
    n_c: int,
) -> np.ndarray:
    """对角退化的 Bayes / James-Stein mean shift. O(p), 无矩阵分解.

    公式 (源: `.../src/HierarchicalBayes/mean_shift.jl`, 函数 `compute_mean_shift`):

        V_c   = (Sigma_b^{-1} + n_c Sigma^{-1})^{-1}
        b_hat = V_c (n_c Sigma^{-1} r_bar)

    取 Sigma = diag(sigma2_gene), Sigma_b = sigma2_between * I 后逐维退化为

        b_hat_j = r_bar_j * n_c * sigma2_between
                  / (sigma2_gene_j + n_c * sigma2_between)

    即收缩因子 w_j = 1 / (1 + sigma2_gene_j / (n_c * sigma2_between)):
    sigma2_between -> 0 时 b_hat -> 0 (完全信全局), -> inf 时 b_hat -> r_bar
    (完全信本组). 本实现用 w_j 的倒数形式, 因此 0 与 inf 两个极限都数值稳定.

    为何适用于本问题: 300 个扰动共享同一批基因, 逐扰动的均值残差 r_bar 只由
    400 个细胞估出, 噪声与 sigma2_gene/n_c 同阶. 逐基因收缩把高噪声基因的
    shift 压回 0, 只让真正超出组间方差尺度的基因保留位移 —— 恰好对应打分器
    「每基因只看一个均值」的一阶矩敏感性, 而且完全不需要 p x p 分解.
    """
    r_bar = np.asarray(r_bar, dtype=np.float64)
    sigma2_gene = np.asarray(sigma2_gene, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = sigma2_gene / (n_c * sigma2_between)
        w = 1.0 / (1.0 + ratio)
    w = np.where(np.isfinite(w), w, 0.0)
    return r_bar * w


def tail_aware_scale(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """尾部自适应的尺度估计: 轻尾用 MAD, 重尾用 2%-98% 分位间距.

        tail_index = std / (1.4826 * MAD)
        tail_index <= 3  ->  scale = 1.4826 * MAD
        tail_index >  3  ->  scale = (p98 - p02) / (2 * 2.0537489)

    2.0537489 = Phi^{-1}(0.98), 所以两支在高斯下都是 sigma 的一致估计.

    源: 本仓库新增 (AdaptiveEROP 的 residuals.jl / inverse_wishart.jl 只做去心和
    LW 收缩, 没有鲁棒尺度这一支; 这里补上, 供 `vcclab.probit` 的 sigma_tilde 用).

    为何适用于本问题: 单细胞计数是重尾的 (负二项 + 零膨胀), 少数高表达细胞会把
    std 抬到 MAD 的数倍. 裸 MAD 在这种情形下**低估**尺度 -> 显著性被高估;
    裸 std 又被离群细胞主导. tail_index 这个开关按分布形状选择, 两端都不失手.
    """
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - med), axis=axis, keepdims=True)
    mad_scale = _MAD_TO_SIGMA * mad
    sd = np.std(x, axis=axis, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        tail_index = sd / mad_scale
    p02, p98 = np.percentile(x, [2.0, 98.0], axis=axis, keepdims=True)
    span_scale = (p98 - p02) / (2.0 * _P98_Z)
    heavy = ~(tail_index <= 3.0)          # MAD == 0 -> tail_index = inf/nan -> 走重尾支
    scale = np.where(heavy, span_scale, mad_scale)
    return np.squeeze(scale, axis=axis)
