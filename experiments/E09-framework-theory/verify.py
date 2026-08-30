"""验证框架的两个承重命题（不验证就不许动手实现）。

命题 1（最优召集集合）
    给定每个基因的检出概率 p_g = P(g ∈ R_p)，在 |R̂| = K 固定下，
    取 p_g 最大的 K 个使 E|R_p ∩ R̂| 最大。
    证明：E|R∩R̂| = Σ_{g∈R̂} p_g，对 R̂ 的成员指示是线性的，故取最大的 K 个。

命题 2（最优 K 的闭式停止规则）
    Jaccard J = m/(n+K-m)，令 M(K) = Σ_{i≤K} p_i（p 降序），则
        dE[J]/dK = 0  ⟺  p_{K*} = M(K*)/(n+K*) = J/(1+J) = h/2
    即「只要下一个基因的检出概率高于当前已达 h 的一半，就继续加」。

两个命题都用暴力搜索独立核对，不信推导。
"""

from __future__ import annotations

import numpy as np


def expected_jaccard(p: np.ndarray, k: int, n_mc: int, rng: np.random.Generator) -> float:
    """蒙特卡洛算 E[J]：按 p 独立伯努利生成真集 R，取前 k 个作 R̂。"""
    g = len(p)
    draws = rng.random((n_mc, g)) < p
    sel = np.zeros(g, bool)
    sel[:k] = True
    inter = (draws & sel).sum(1)
    union = (draws | sel).sum(1)
    return float(np.mean(np.divide(inter, union, out=np.zeros(n_mc), where=union > 0)))


def main() -> None:
    rng = np.random.default_rng(0)
    g = 3000

    # ---------------- 命题 1 ----------------
    print("=== 命题 1：固定 K 下，按 p 取前 K 个是最优召集 ===")
    p = np.sort(rng.beta(0.35, 3.0, g))[::-1]
    k = 288
    n_mc = 4000
    best_by_p = float(np.mean(rng.random((n_mc, g))[:, :k] < p[:k]) * k)
    exact_top = p[:k].sum()
    print(f"  取前 {k} 个: E|R∩R̂| = Σp = {exact_top:.2f}")

    alts = {
        "随机 K 个": rng.permutation(g)[:k],
        "按 p 取最后 K 个": np.arange(g - k, g),
        "隔一个取（前 2K 中）": np.arange(0, 2 * k, 2),
        "前 K/2 + 随机 K/2": np.concatenate(
            [np.arange(k // 2), rng.permutation(np.arange(k // 2, g))[: k - k // 2]]),
    }
    for name, idx in alts.items():
        print(f"  {name:22s} Σp = {p[idx].sum():8.2f}   劣于最优 "
              f"{exact_top - p[idx].sum():7.2f}")
    assert all(p[idx].sum() <= exact_top + 1e-9 for idx in alts.values())
    # 随机搜索也打不过
    worst_margin = min(exact_top - p[rng.permutation(g)[:k]].sum() for _ in range(2000))
    print(f"  2000 次随机集合中最好的仍差 {worst_margin:.2f}  → 命题 1 成立")

    # ---------------- 命题 2 ----------------
    print("\n=== 命题 2：最优 K 满足 p_K* = J/(1+J) = h/2 ===")
    print(f"{'情形':>14} {'暴力最优K':>10} {'E[J]':>8} {'p_K*':>8} "
          f"{'J/(1+J)':>9} {'相对误差':>9}")
    scenarios = {
        "陡降 beta": np.sort(rng.beta(0.30, 3.0, g))[::-1],
        "平缓 beta": np.sort(rng.beta(0.8, 2.0, g))[::-1],
        "阶跃 288": np.concatenate([np.full(288, 0.62), np.full(g - 288, 0.004)]),
        "阶跃 288 弱": np.concatenate([np.full(288, 0.33), np.full(g - 288, 0.010)]),
        "指数衰减": 0.85 * np.exp(-np.arange(g) / 260.0),
    }
    n_mc = 20000
    for name, pv in scenarios.items():
        ks = np.unique(np.clip(
            np.round(np.geomspace(20, 1600, 46)).astype(int), 1, g))
        js = np.array([expected_jaccard(pv, int(kk), n_mc, rng) for kk in ks])
        i = int(np.argmax(js))
        kstar, jstar = int(ks[i]), float(js[i])
        pk = float(pv[kstar - 1])
        pred = jstar / (1 + jstar)
        rel = abs(pk - pred) / pred if pred > 0 else np.nan
        print(f"{name:>14} {kstar:10d} {jstar:8.4f} {pk:8.4f} {pred:9.4f} {rel:8.1%}")

    # ---------------- 用官方锚点核对 ----------------
    print("\n=== 用官方锚点核对停止阈值 ===")
    j_off = 0.399
    h_off = 2 * j_off / (1 + j_off)
    print(f"  官方 replicate 锚点 J = {j_off:.3f}  →  h = 2J/(1+J) = {h_off:.3f}")
    print(f"  停止阈值 p* = J/(1+J) = {j_off/(1+j_off):.3f} = h/2 = {h_off/2:.3f}")
    print("  含义：只要下一个基因的检出概率 > 已达 h 的一半，就继续加入召集集合。")

    # 与已确认的 K* = 287 / E|R_p| = 288 对齐性检查
    print("\n=== 与已确认事实的一致性 ===")
    q = 0.62
    pv = np.concatenate([np.full(288, q), np.full(g - 288, 0.004)])
    ks = np.arange(150, 460, 5)
    js = np.array([expected_jaccard(pv, int(kk), 20000, rng) for kk in ks])
    kstar = int(ks[int(np.argmax(js))])
    print(f"  阶跃型 p（288 个 q={q}）暴力最优 K = {kstar}")
    print(f"  已确认事实 K* = 287 · 解析 E|R_p| = 288  →  "
          f"{'一致' if abs(kstar - 288) <= 12 else '不一致，需查'}")


if __name__ == "__main__":
    main()
