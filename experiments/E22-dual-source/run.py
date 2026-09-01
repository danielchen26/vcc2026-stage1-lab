"""E22 — 双源预测 |R_p| → K_opt，最后一块。

## E21 定位的唯一瓶颈

    K_opt  <--rho=0.911--  |R_p|  <--rho=0.722--  n_src(K562)

K_opt 几乎由 |R_p| 完全决定（0.911），比值按分区稳定（<50: 0.50 / 50-500: 1.05 / >500: 2.94）。
而 oracle K_opt 能 47/47 全胜 baseline、达追平线 1.44×。
**唯一瓶颈是从源侧预测 |R_p| 只有 rho=0.722**（二分类准确率仅 70-74%）。

## 本实验：加第二个独立源

Zhu/Marson CD4+T（Cell 2026）的 suppl table 已下载，Rest 臂（无刺激）给出每扰动的
`n_total_de_genes` —— **第二个独立的响应量级估计**，且覆盖 H1 的 42/50 个扰动。

用两个源联合预测 K_opt，5 折扰动级交叉验证：

    F1  只用 n_src(K562)                E21 的做法，jac 均值 0.0747
    F2  只用 n_de(CD4T-Rest)
    F3  两者联合
    F4  两者 + CD4T 的 ontarget_effect_size
    对照 R2 全报（baseline）0.1198 · R4 oracle K_opt 0.2434

## 判读

    追平线 raw jac = 0.12 + 0.1899 × (0.379 - 0.12) = 0.1692
    **看均值**（分数按扰动平均）

跑法：  ~/vcc2026/.venv/bin/python experiments/E22-dual-source/run.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
E21 = ROOT / "experiments" / "E21-regime-switch" / "result.csv"
CD4 = ROOT / "data" / "external" / "cd4_de.csv"
N_FOLD = 5
SEED = 0
JAC_BASELINE, JAC_REPLICATE, LEADER = 0.12, 0.379, 0.1899
JAC_TIE = JAC_BASELINE + LEADER * (JAC_REPLICATE - JAC_BASELINE)


def main() -> None:
    t0 = time.time()
    print("=== E22 双源预测 K_opt ===")
    print(f"追平线 raw jac = {JAC_TIE:.4f}（看均值）\n")

    d = pd.read_csv(E21)
    print(f"E21 的 47 个扰动已载入（含 k_opt、n_real、n_src、以及各规则的 jac）")

    cd = pd.read_csv(CD4)
    rest = cd[cd.culture_condition == "Rest"].copy()
    cols = [c for c in rest.columns]
    keep = ["target_contrast_gene_name", "n_total_de_genes", "n_cells_target"]
    extra = [c for c in ("ontarget_effect_size", "ontarget_significant") if c in cols]
    rest = rest[keep + extra].rename(
        columns={"target_contrast_gene_name": "target_gene",
                 "n_total_de_genes": "n_cd4", "n_cells_target": "cells_cd4"})
    rest = rest.drop_duplicates("target_gene")
    m = d.merge(rest, on="target_gene", how="left")
    have = m.n_cd4.notna()
    print(f"CD4+T Rest 覆盖 {have.sum()}/{len(m)} 个扰动"
          f"   n_cd4 中位 {m.loc[have,'n_cd4'].median():.0f}")

    print(f"\n=== 各源与靶侧 |R_p| / K_opt 的相关（只在两源都有的 {have.sum()} 个上）===")
    s = m[have]
    for x in ("n_src", "n_cd4"):
        print(f"  {x:8s} vs |R_p|  Spearman {spearmanr(s[x], s.n_real).statistic:6.3f}"
              f"   vs K_opt {spearmanr(s[x], s.k_opt).statistic:6.3f}")
    comb = np.log10(s.n_src + 1) + np.log10(s.n_cd4 + 1)
    print(f"  两源之和(log) vs |R_p|  Spearman {spearmanr(comb, s.n_real).statistic:6.3f}"
          f"   vs K_opt {spearmanr(comb, s.k_opt).statistic:6.3f}")
    print(f"  两源之间 Spearman {spearmanr(s.n_src, s.n_cd4).statistic:.3f}"
          f"  （越低越互补）")

    # 需要 jac(K) 曲线来评估任意 K —— E21 只存了几个固定规则的值。
    # 这里改用「K_opt 的预测误差」代理，再用 E21 已存的 oracle/baseline 做上下界锚。
    # 为了给出真实 jac，必须重算曲线 → 复用 E21 的 result 里没有曲线，故此处
    # 以「预测 K 落在 jac(K) 曲线上的值」为目标重算：直接调用 E21 的管线太重，
    # 因此本实验只回答「双源能否把 K_opt 预测得更准」，并按 E21 实测的
    # 「K 预测精度 → jac」关系给出量化推断。
    feats = {
        "F1_k562": lambda t: np.column_stack([np.ones(len(t)), np.log10(t.n_src + 1)]),
        "F2_cd4": lambda t: np.column_stack([np.ones(len(t)), np.log10(t.n_cd4 + 1)]),
        "F3_both": lambda t: np.column_stack(
            [np.ones(len(t)), np.log10(t.n_src + 1), np.log10(t.n_cd4 + 1)]),
    }
    if "ontarget_effect_size" in m.columns:
        feats["F4_both_eff"] = lambda t: np.column_stack(
            [np.ones(len(t)), np.log10(t.n_src + 1), np.log10(t.n_cd4 + 1),
             np.abs(t.ontarget_effect_size.fillna(0.0))])

    sub = m[have].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    folds = rng.permutation(len(sub)) % N_FOLD
    y = np.log10(sub.k_opt.clip(lower=1))
    print(f"\n=== 预测 log10(K_opt) 的留出精度（5 折，n={len(sub)}）===")
    print(f"{'特征':>16} {'留出 RMSE':>11} {'留出 Spearman':>15} {'中位倍数误差':>13}")
    res = {}
    for name, fn in feats.items():
        X = fn(sub)
        pred = np.zeros(len(sub))
        for f_ in range(N_FOLD):
            tr, te = folds != f_, folds == f_
            c = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
            pred[te] = X[te] @ c
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        rho = spearmanr(10 ** pred, sub.k_opt).statistic
        fold_err = float(np.median(10 ** np.abs(pred - y)))
        res[name] = dict(pred=10 ** pred, rmse=rmse, rho=rho, fold=fold_err)
        print(f"{name:>16} {rmse:11.4f} {rho:15.3f} {fold_err:12.2f}×")

    best = min(res, key=lambda k: res[k]["rmse"])
    print(f"\n最好: {best}   RMSE {res[best]['rmse']:.4f}"
          f"   vs 只用 K562 的 {res['F1_k562']['rmse']:.4f}"
          f"   改善 {1-res[best]['rmse']/res['F1_k562']['rmse']:.0%}")

    print(f"\n=== 已知的 jac 锚点（E21 实测，同一批扰动的子集）===")
    for col, lab in (("jac_R2_all", "全报 baseline"), ("jac_R1_pred", "E21 单源预测 K"),
                     ("jac_R5_oracleN", "oracle |R_p|"), ("jac_R4_oracleK", "oracle K_opt")):
        print(f"  {lab:>18} jac 均值 {sub[col].mean():.4f}  中位 {sub[col].median():.4f}")

    print(f"\n>>> 本实验只验证「双源能否把 K_opt 预测得更准」<<<")
    if res[best]["rmse"] < res["F1_k562"]["rmse"] * 0.95:
        print(f"    是：RMSE 降低 {1-res[best]['rmse']/res['F1_k562']['rmse']:.0%}，"
              f"Spearman {res['F1_k562']['rho']:.3f} → {res[best]['rho']:.3f}")
        print(f"    下一步必须重跑 E21 的完整管线用这个 K，才能给出真实 jac。")
    else:
        print(f"    否：双源没有改善 K_opt 的预测（RMSE {res[best]['rmse']:.4f} "
              f"vs {res['F1_k562']['rmse']:.4f}）")
    sub.assign(**{f"pred_{k}": v["pred"] for k, v in res.items()}).to_csv(
        OUT / "result.csv", index=False)
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
