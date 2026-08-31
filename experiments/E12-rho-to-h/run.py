"""E12 — 把「跨系 LFC 相关」正确转换成「集合重叠 h」，判定能否追平榜首。

## 必须纠正的一个跳跃

F22 测得（外部、MASH 功效校正、本届真实靶基因）：
    跨系 LFC Pearson r = 0.339
    donor 内同系  r    = 0.479
    比值 = 0.708

我先前直接写 h = 0.708 × h_replicate = 0.708 × 0.550 = 0.389。**这是错的。**
h = |R_p ∩ R̂_p| / |R_p| 是**阈值化之后的集合重叠**，LFC 相关是连续量。
从相关系数到集合重叠的映射高度非线性（阈值附近的基因决定集合边界，
而那正是相关性最不起作用的区域）。转换必须显式做。

## 本实验的做法

用实测参数搭一个前向模型，直接算 h：

  1. 目标 context 的逐基因阈值 MDE：用 E08b 实测的 context_A 分布
     （p5 0.0907 · p25 0.1822 · p50 0.3029 · p75 0.5405 · p95 1.2971，gate 9,929）
  2. 真实效应量 beta_T ~ pi0*delta_0 + (1-pi0)*N(0, tau^2)，
     标定 (pi0, tau) 使 |R_p| = 253（E10 实测中位）
  3. 源侧预测 beta_S 与 beta_T 的相关为 rho（要扫的自变量）
  4. 按 |beta_S| 降序取 top-K，K 由 vcclab.callset.best_k 定
  5. h = |{|beta_T| > MDE} ∩ callset| / |{|beta_T| > MDE}|

然后回答两个问题：
  - rho = 0.339 时 h 是多少？
  - 追平榜首（h = 0.127，E11 实测）需要多大的 rho？

跑法：  ~/vcc2026/.venv/bin/python experiments/E12-rho-to-h/run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vcclab.callset import best_k  # noqa: E402

GATE = 9_929
MDE_Q = {5: 0.0907, 25: 0.1822, 50: 0.3029, 75: 0.5405, 95: 1.2971}
RP_TARGET = 253          # E10 实测中位
H_REPLICATE = 0.550      # E11 实测
H_TIE = 0.23 * H_REPLICATE
RHO_MEASURED = 0.339     # F22 外部实测，本届真实靶基因
RHO_DONOR = 0.479        # F22 同系 donor 内
N_REP = 40
SEED = 0


def mde_sample(rng: np.random.Generator, n: int) -> np.ndarray:
    """从实测分位数插值出 MDE 样本（对数尺度插值，尾部才对）。"""
    qs = np.array(sorted(MDE_Q))
    vs = np.log(np.array([MDE_Q[q] for q in qs]))
    u = rng.uniform(qs.min(), qs.max(), n)
    return np.exp(np.interp(u, qs, vs))


def calibrate(rng: np.random.Generator, tau: float) -> float:
    """给定 tau，解出使 E|R_p| = RP_TARGET 的 pi0。"""
    mde = mde_sample(rng, GATE)

    def n_sig(pi0: float) -> float:
        frac = 2 * norm.sf(mde / tau)      # 非零基因越阈的概率
        return float((1 - pi0) * frac.sum())

    lo, hi = 1e-6, 1 - 1e-6
    if n_sig(lo) < RP_TARGET:
        return lo
    return brentq(lambda p: n_sig(p) - RP_TARGET, lo, hi, xtol=1e-9)


def simulate(rng: np.random.Generator, rho: float, pi0: float, tau: float,
             se_src: float) -> tuple[float, int, int]:
    """一次前向模拟，返回 (h, |R_p|, K)。"""
    mde = mde_sample(rng, GATE)
    nz = rng.random(GATE) >= pi0
    b_t = np.where(nz, rng.standard_normal(GATE) * tau, 0.0)
    # 源侧真值与目标真值相关 rho（只在非零基因上有意义）
    z = rng.standard_normal(GATE)
    b_s_true = rho * b_t + np.sqrt(max(1 - rho * rho, 0.0)) * tau * z * nz
    b_s = b_s_true + rng.standard_normal(GATE) * se_src        # 源侧测量噪声

    real = np.abs(b_t) > mde
    n_real = int(real.sum())
    if n_real == 0:
        return np.nan, 0, 0
    # 召集：按源侧证据排序，K 由 best_k 定（用源侧能算出的检出概率代理）
    score = np.abs(b_s) / np.maximum(mde, 1e-9)
    order = np.argsort(score)[::-1]
    p_proxy = np.clip(2 * norm.sf(mde / np.maximum(np.abs(b_s), 1e-9)), 0, 1)[order]
    k, _ = best_k(p_proxy, float(n_real))
    call = order[:k]
    return float(real[call].sum()) / n_real, n_real, k


def sweep(rng: np.random.Generator, pi0: float, tau: float, se_src: float,
          rhos: np.ndarray) -> dict[float, tuple[float, float, float]]:
    out = {}
    for rho in rhos:
        hs, ns, ks = [], [], []
        for _ in range(N_REP):
            h, n, k = simulate(rng, rho, pi0, tau, se_src)
            if np.isfinite(h):
                hs.append(h); ns.append(n); ks.append(k)
        out[float(rho)] = (float(np.mean(hs)), float(np.mean(ns)), float(np.mean(ks)))
    return out


def main() -> None:
    print("=== E12 从跨系 LFC 相关到集合重叠 h ===")
    print(f"实测输入：|R_p| = {RP_TARGET}（E10）· h_replicate = {H_REPLICATE}（E11）")
    print(f"          追平线 h = {H_TIE:.3f} · 跨系 rho = {RHO_MEASURED}（F22 外部）\n")

    rng = np.random.default_rng(SEED)
    tau = 0.45
    pi0 = calibrate(rng, tau)
    print(f"标定：tau = {tau}  →  pi0 = {pi0:.5f}  "
          f"（非零基因 {int((1-pi0)*GATE)} 个，使 E|R_p| = {RP_TARGET}）")

    print(f"\n--- 先做自洽性检查：rho = 1 且源侧无噪声，h 应接近 1 ---")
    h1, n1, k1 = simulate(np.random.default_rng(1), 1.0, pi0, tau, 1e-6)
    print(f"  h = {h1:.3f}   |R_p| = {n1}   K = {k1}")

    print(f"\n--- 用同系 donor 相关 {RHO_DONOR} 校准源侧噪声 ---")
    print("  目标：让 rho = 0.479 时 h 复现 E11 实测的 0.550")
    print(f"{'se_src':>8} {'h(rho=0.479)':>14}")
    best_se, best_gap = None, 1e9
    for se_src in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45):
        r = sweep(np.random.default_rng(2), pi0, tau, se_src,
                  np.array([RHO_DONOR]))
        h = r[RHO_DONOR][0]
        print(f"{se_src:8.2f} {h:14.3f}")
        if abs(h - H_REPLICATE) < best_gap:
            best_gap, best_se = abs(h - H_REPLICATE), se_src
    print(f"  → 取 se_src = {best_se}（使 h(0.479) 最接近 {H_REPLICATE}）")

    rhos = np.array([0.0, 0.1, 0.2, 0.30, RHO_MEASURED, 0.4, 0.479, 0.6, 0.8, 1.0])
    res = sweep(np.random.default_rng(3), pi0, tau, best_se, rhos)
    print(f"\n--- h 随 rho 的曲线（se_src = {best_se}）---")
    print(f"{'rho':>7} {'h':>7} {'|R_p|':>7} {'K':>7} {'能否追平':>10}")
    for rho in rhos:
        h, n, k = res[float(rho)]
        mark = "✓" if h >= H_TIE else "✗"
        star = "  ← F22 实测" if abs(rho - RHO_MEASURED) < 1e-9 else (
            "  ← 同系 donor" if abs(rho - RHO_DONOR) < 1e-9 else "")
        print(f"{rho:7.3f} {h:7.3f} {n:7.0f} {k:7.0f} {mark:>10}{star}")

    h_at = res[RHO_MEASURED][0]
    print(f"\n{'='*62}\n结论\n{'='*62}")
    print(f"rho = {RHO_MEASURED}（F22 在本届真实靶基因上实测）→ h = {h_at:.3f}")
    print(f"追平榜首需 h = {H_TIE:.3f}（E11 实测）")
    print(f"余量 = {h_at/H_TIE:.2f}×")
    print(f"\n对比我先前那个错误折算：0.708 × {H_REPLICATE} = "
          f"{0.708*H_REPLICATE:.3f}  （高估 {0.708*H_REPLICATE/h_at:.2f} 倍）")

    # 追平所需的 rho
    lo = [r for r in rhos if res[float(r)][0] < H_TIE]
    hi = [r for r in rhos if res[float(r)][0] >= H_TIE]
    if lo and hi:
        print(f"\n追平所需 rho 落在 ({max(lo):.3f}, {min(hi):.3f}) 之间")
    elif not lo:
        print(f"\n即使 rho = 0 也能追平 —— 说明追平线极低，检查设定")
    else:
        print(f"\n即使 rho = 1 也追不平 —— 集合重叠的结构性上限低于追平线")

    verdict = ("可以追平并超过" if h_at >= H_TIE else "追不平，需要额外信号源")
    print(f"\n>>> {verdict} <<<")


if __name__ == "__main__":
    main()
