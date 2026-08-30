"""E08c — 用经验贝叶斯反卷积，算本届靶基因在真实 context 阈值下的 E|R_p|。

## 为什么需要这一版（run_erp.py 的 bug，保留作教训）

run_erp.py 直接拿 K562 的 beta_hat 与 context 的 MDE 比大小，得出 |R_p| = 981。
**那是噪声底，不是生物学。** beta_hat 的噪声 SD = lfcSE ~ 0.1987，一个真值
beta = 0 的基因也有 P(|beta_hat| > 0.30) = 2*Phi(-0.30/0.1987) = 13% 的概率越阈，
乘 8,248 个基因 ≈ 1,080 个假阳性 —— 与测出的 981 吻合。

把带噪估计当真值去过阈值，是「不做反卷积就数显著」这一类错误。必须建模测量误差。

## 正确的算法

每个扰动上拟合两成分先验（8,248 个观测，只 2 个参数，极好定）：

    beta_g ~ pi0 * delta_0 + (1 - pi0) * N(0, tau^2)
    beta_hat_g | beta_g ~ N(beta_g, se_g^2)        se_g = lfcSE，已知

边缘似然（se 逐基因不同，所以不能只看 z 的方差）：

    L = prod_g [ pi0 * N(beta_hat_g; 0, se_g^2)
                 + (1-pi0) * N(beta_hat_g; 0, se_g^2 + tau^2) ]

后验：

    P(非零 | beta_hat)  = (1-pi0)*N(bh; 0, se^2+tau^2) / [上式括号内]
    beta | 非零, bh     ~ N( bh * tau^2/(tau^2+se^2),  tau^2 se^2/(tau^2+se^2) )

于是对任意阈值 t（零成分贡献 0，因为 beta 恰为 0 而 t > 0）：

    P(|beta_g| > t | bh_g) = P(非零|bh) * [ Phi((-t-mu)/sqrt(v)) + Phi((mu-t)/sqrt(v)) ]

    E|R_p| = sum_g P(|beta_g| > MDE_g)

这个量**就是 TAP 框架的 P(检出)**（差一步跨系迁移），所以本实验同时标定了框架的核心件。

## 顺带把命题 2 用在真实曲线上

有了逐基因的 p_g，直接数值最大化 E[J] 定出每个扰动的最优召集集合大小 K_p，
不用闭式（闭式在 p 曲线呈阶跃时失效，见 docs/08-framework.md 第 5 节）。

跑法：  ~/vcc2026/.venv/bin/python experiments/E08-target-vs-essential/run_erp_eb.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vcclab.detectability import mde  # noqa: E402
from vcclab.scorer import ControlRef, bh_adjust  # noqa: E402

DATA = ROOT / "data" / "nadig2025"
VCC = Path.home() / "vcc2026"
Z_BH = 3.184
ALPHA_BH = float(2 * norm.sf(Z_BH))
N_CELLS = 400
SEED = 0
EPS = 1e-300


def read_gw(kind: str, cols: list[str]) -> pd.DataFrame:
    f = {"p": "K562GW_p", "lfc": "K562GW_lfc", "se": "K562GW_se"}[kind]
    df = pd.read_csv(DATA / f"{f}.csv.gz", index_col=0,
                     usecols=["Unnamed: 0"] + cols,
                     dtype={c: np.float32 for c in cols}, engine="c")
    df.index = df.index.astype(str)
    return df[cols]


def fit_prior(bh: np.ndarray, se: np.ndarray) -> tuple[float, float]:
    """拟合 (pi0, tau)。先对 tau 做一维搜索，pi0 在给定 tau 下有解析剖面解。"""
    v0 = se ** 2

    def negll(log_tau: float) -> float:
        tau2 = float(np.exp(2 * log_tau))
        f0 = np.exp(-0.5 * bh ** 2 / v0) / np.sqrt(v0)
        f1 = np.exp(-0.5 * bh ** 2 / (v0 + tau2)) / np.sqrt(v0 + tau2)
        # 给定 tau，用 EM 的几步迭代求 pi0（凹问题，收敛很快）
        pi0 = 0.9
        for _ in range(60):
            w = pi0 * f0 / np.maximum(pi0 * f0 + (1 - pi0) * f1, EPS)
            pi0 = float(np.clip(w.mean(), 1e-4, 1 - 1e-6))
        return -float(np.sum(np.log(np.maximum(pi0 * f0 + (1 - pi0) * f1, EPS))))

    r = minimize_scalar(negll, bounds=(np.log(1e-3), np.log(5.0)), method="bounded",
                        options={"xatol": 1e-3})
    tau = float(np.exp(r.x))
    tau2 = tau ** 2
    f0 = np.exp(-0.5 * bh ** 2 / v0) / np.sqrt(v0)
    f1 = np.exp(-0.5 * bh ** 2 / (v0 + tau2)) / np.sqrt(v0 + tau2)
    pi0 = 0.9
    for _ in range(200):
        w = pi0 * f0 / np.maximum(pi0 * f0 + (1 - pi0) * f1, EPS)
        pi0 = float(np.clip(w.mean(), 1e-4, 1 - 1e-6))
    return pi0, tau


def p_exceed(bh: np.ndarray, se: np.ndarray, thr: np.ndarray,
             pi0: float, tau: float) -> np.ndarray:
    """P(|beta| > thr | beta_hat)，两侧越阈，零成分贡献 0。"""
    v0, tau2 = se ** 2, tau ** 2
    f0 = np.exp(-0.5 * bh ** 2 / v0) / np.sqrt(v0)
    f1 = np.exp(-0.5 * bh ** 2 / (v0 + tau2)) / np.sqrt(v0 + tau2)
    p_alt = (1 - pi0) * f1 / np.maximum(pi0 * f0 + (1 - pi0) * f1, EPS)
    shrink = tau2 / (tau2 + v0)
    mu = bh * shrink
    sd = np.sqrt(np.maximum(tau2 * v0 / (tau2 + v0), EPS))
    up = norm.sf((thr - mu) / sd)
    dn = norm.cdf((-thr - mu) / sd)
    return p_alt * (up + dn)


def best_k(p: np.ndarray, n_true: float) -> tuple[int, float]:
    """按命题 2 数值最大化 E[J]；p 已降序。E[J] ~ M(K)/(n + K - M(K))。"""
    ps = np.sort(p)[::-1]
    m = np.cumsum(ps)
    k = np.arange(1, len(ps) + 1)
    j = m / np.maximum(n_true + k - m, 1e-9)
    i = int(np.argmax(j))
    return i + 1, float(j[i])


def main() -> None:
    t0 = time.time()
    print("=== E08c 经验贝叶斯：本届靶基因在真实 context 阈值下的 E|R_p| ===")
    print(f"BH-有效 z = {Z_BH}  alpha_eff = {ALPHA_BH:.6f}\n")

    genes = pd.read_csv(VCC / "gene_names.csv")["gene_name"].tolist()
    mdes = {}
    for ctx in ("A", "B", "C"):
        ref = ControlRef.load(VCC / f"context_{ctx}.h5ad", genes)
        m = mde(ref, n_cells=N_CELLS, alpha=ALPHA_BH, seed=SEED, tie_correct=True)
        mdes[ctx] = np.sort(m[np.isfinite(m)])
        print(f"context_{ctx}  MDE 中位 = {np.median(mdes[ctx]):.4f}  n={len(mdes[ctx]):,}")

    gw_cols = [str(c) for c in pd.read_csv(DATA / "K562GW_p.csv.gz",
                                          index_col=0, nrows=1).columns]
    vcc_t = {str(t) for t in pd.read_csv(VCC / "pert_counts.csv")["target_gene"]
             } - {"non-targeting"}
    ess = {str(c) for c in pd.read_csv(DATA / "K562Essential_p.csv.gz",
                                       index_col=0, nrows=1).columns}
    g_vcc = sorted(set(gw_cols) & vcc_t)
    g_ess = sorted(set(gw_cols) & ess)
    cols = sorted(set(g_vcc) | set(g_ess))

    lfc = read_gw("lfc", cols)
    se_df = read_gw("se", cols).reindex(index=lfc.index)
    p_df = read_gw("p", cols).reindex(index=lfc.index)
    B = lfc.to_numpy().astype(np.float64)
    S = se_df.to_numpy().astype(np.float64)
    P = p_df.to_numpy().astype(np.float64)
    pos = {c: i for i, c in enumerate(cols)}
    print(f"\nK562GW: {B.shape[0]:,} 基因 × {B.shape[1]:,} 扰动 "
          f"(本届靶 {len(g_vcc)} · 必需 {len(g_ess)})")

    det = np.nanmedian(S, axis=1)
    keep = np.isfinite(det) & np.isfinite(B).all(1) & np.isfinite(S).all(1)
    order = np.argsort(det[keep])
    gi = np.flatnonzero(keep)[order]
    n_g = len(gi)
    print(f"完整可用基因 {n_g:,}   K562 隐含 MDE 中位 = "
          f"{Z_BH*np.median(det[gi]):.4f}\n")

    rows = []
    for tag, gl in (("V 本届靶", g_vcc), ("E 必需", g_ess)):
        ii = [pos[c] for c in gl]
        # K562 自身功效下的 BH 计数（= E08 的口径，作锚）
        bhn = []
        for j in ii:
            col = P[gi, j]
            ok = np.isfinite(col)
            bhn.append(int((bh_adjust(col[ok]) < 0.05).sum()))
        print(f"{tag}: K562 自身 BH |R_p| 中位 = {np.median(bhn):.0f}  "
              f"(E08 测得 {'15' if tag.startswith('V') else '123'})")

        for ctx in ("A", "B", "C"):
            thr = np.interp(np.linspace(0, 1, n_g),
                            np.linspace(0, 1, len(mdes[ctx])), mdes[ctx])
            erp, ks, js, pi0s, taus = [], [], [], [], []
            for j in ii:
                bh_, se_ = B[gi, j], S[gi, j]
                pi0, tau = fit_prior(bh_, se_)
                pg = p_exceed(bh_, se_, thr, pi0, tau)
                n_hat = float(pg.sum())
                k, jj = best_k(pg, n_hat)
                erp.append(n_hat); ks.append(k); js.append(jj)
                pi0s.append(pi0); taus.append(tau)
            erp = np.array(erp); ks = np.array(ks); js = np.array(js)
            q1, med, q3 = np.percentile(erp, [25, 50, 75])
            rows.append(dict(group=tag, context=ctx, E_Rp=med, q25=q1, q75=q3,
                             K_opt=float(np.median(ks)), E_J=float(np.median(js)),
                             pi0=float(np.median(pi0s)), tau=float(np.median(taus))))
            print(f"   context_{ctx}  E|R_p| 中位 = {med:7.1f}  "
                  f"IQR [{q1:.0f}, {q3:.0f}]   最优 K_p 中位 = {np.median(ks):5.0f}   "
                  f"上限 E[J] = {np.median(js):.3f}   "
                  f"pi0 = {np.median(pi0s):.4f}  tau = {np.median(taus):.3f}")
        print()

    df = pd.DataFrame(rows)
    v = df[df.group == "V 本届靶"]
    print("--- 结论 ---")
    print(f"本届 272 个真实靶基因在真实 context 阈值下 E|R_p| = "
          f"{v.E_Rp.min():.0f} ~ {v.E_Rp.max():.0f} (三个 context)")
    print(f"对应最优召集集合大小 K_p = {v.K_opt.min():.0f} ~ {v.K_opt.max():.0f}")
    print(f"我之前从官方锚点反推的 E|R_p| = 288 (范围 207–365)")
    lo, hi = v.E_Rp.min(), v.E_Rp.max()
    print(f"\n>>> {'相容' if lo <= 365 and hi >= 207 else '不相容，288 需重新审查'} <<<")
    print(f"\n注：run_erp.py 那版直接拿 beta_hat 过阈值，得 981 —— 那是噪声底"
          f"（{Z_BH}·lfcSE 下真值为 0 的基因就有 13% 越阈 × 8,248 ≈ 1,080）。")
    print(f"\n耗时 {time.time()-t0:.0f}s")
    df.to_csv(Path(__file__).parent / "result_erp_eb.csv", index=False)


if __name__ == "__main__":
    main()
