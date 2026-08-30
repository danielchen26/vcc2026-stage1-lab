"""E08b — 解决 E|R_p| 的冲突：本届靶基因在真实 context 阈值下到底有多少响应基因。

冲突：
    我之前从官方锚点反推 E|R_p| ≈ 288（范围 207–365），并用 K*=287 自证。
    但两条都是自己的推导，没有外部校验。E08 在真实细胞系里量到本届 272 个靶基因
    的 |R_p| 中位数只有 **15**（必需基因 123，同一细胞系同一库同一管线）。
    差 19 倍。必须解决，否则框架里每个扰动的召集集合大小 K_p 全错。

解法（不需要基因 ID 映射）：
    用**真实靶基因在 K562 测到的真实效应量**，套上**真实 context 的 400 细胞
    MDE 曲线**，直接数有多少越阈。

    两边的基因按**可检测性分位**匹配：
      - context 侧：MDE_A[g]，由官方对照细胞算出（Wilcoxon 自举，BH-有效 alpha）
      - K562 侧：按该基因的中位 lfcSE 排序（越小越可检测）
    分位匹配正是这里该用的匹配方式 —— 问题是「有多少基因越阈」，
    决定它的是「效应量与阈值的联合分布」，而分位匹配恰好保留这个联合结构。

明确的假设（不是事实）：
    1. 本届 context 对这 272 个靶基因的响应量级与 K562 相当；
    2. 基因的可检测性排序在两个数据集间大体保序。
    两条都无法在没有 ground truth 的情况下验证，所以结论要按区间报。

跑法：  ~/vcc2026/.venv/bin/python experiments/E08-target-vs-essential/run_erp.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vcclab.detectability import mde  # noqa: E402
from vcclab.scorer import ControlRef  # noqa: E402

DATA = ROOT / "data" / "nadig2025"
VCC = Path.home() / "vcc2026"
Z_BH = 3.184                      # 已确认的 BH-有效 z（不是 1.96）
ALPHA_BH = float(2 * norm.sf(Z_BH))
N_CELLS = 400
SEED = 0


def read_gw(kind: str, cols: list[str]) -> pd.DataFrame:
    f = {"p": "K562GW_p", "lfc": "K562GW_lfc", "se": "K562GW_se"}[kind]
    df = pd.read_csv(DATA / f"{f}.csv.gz", index_col=0,
                     usecols=["Unnamed: 0"] + cols,
                     dtype={c: np.float32 for c in cols}, engine="c")
    df.index = df.index.astype(str)
    return df[cols]


def main() -> None:
    t0 = time.time()
    print("=== E08b 本届靶基因在真实 context 阈值下的 |R_p| ===")
    print(f"BH-有效 z = {Z_BH}  →  alpha_eff = {ALPHA_BH:.6f}\n")

    # ---- 1. 真实 context 的 MDE 曲线 ----
    genes = pd.read_csv(VCC / "gene_names.csv")["gene_name"].tolist()
    mdes = {}
    for ctx in ("A", "B", "C"):
        t = time.time()
        ref = ControlRef.load(VCC / f"context_{ctx}.h5ad", genes)
        m = mde(ref, n_cells=N_CELLS, alpha=ALPHA_BH, seed=SEED, tie_correct=True)
        m = m[np.isfinite(m)]
        mdes[ctx] = np.sort(m)
        q = np.percentile(mdes[ctx], [5, 25, 50, 75, 95])
        print(f"context_{ctx}  gate={ref.G:,}  可算 MDE={len(m):,}  "
              f"p5/p25/p50/p75/p95 = " + " · ".join(f"{x:.4f}" for x in q)
              + f"   ({time.time()-t:.0f}s)")

    # ---- 2. K562GW 侧：本届靶基因组的效应量与可检测性 ----
    gw_cols = [str(c) for c in pd.read_csv(DATA / "K562GW_p.csv.gz",
                                           index_col=0, nrows=1).columns]
    vcc_t = {str(t) for t in pd.read_csv(VCC / "pert_counts.csv")["target_gene"]
             } - {"non-targeting"}
    ess = {str(c) for c in pd.read_csv(DATA / "K562Essential_p.csv.gz",
                                       index_col=0, nrows=1).columns}
    g_vcc = sorted(set(gw_cols) & vcc_t)
    g_ess = sorted(set(gw_cols) & ess)
    print(f"\nK562GW: 本届靶 {len(g_vcc)} 个 · 必需 {len(g_ess)} 个")

    cols = sorted(set(g_vcc) | set(g_ess))
    lfc = read_gw("lfc", cols)
    se = read_gw("se", cols).reindex(index=lfc.index)
    b = np.abs(lfc.to_numpy().astype(np.float64))
    s = se.to_numpy().astype(np.float64)
    pos = {c: i for i, c in enumerate(cols)}
    iv = np.array([pos[c] for c in g_vcc])
    ie = np.array([pos[c] for c in g_ess])

    # 每基因可检测性代理 = 该基因在本届靶扰动上的中位 lfcSE（越小越可检测）
    det = np.nanmedian(s[:, iv], axis=1)
    ok = np.isfinite(det)
    print(f"K562GW 被测基因 {b.shape[0]:,}  可用 {ok.sum():,}")
    print(f"  lfcSE 中位 = {np.nanmedian(det[ok]):.4f}  "
          f"→ 隐含 MDE = z·lfcSE 中位 = {Z_BH*np.nanmedian(det[ok]):.4f}")

    # ---- 3. 分位匹配后逐扰动计数 ----
    order = np.argsort(det[ok])                 # 最可检测的排前面
    idx_ok = np.flatnonzero(ok)[order]
    n_g = len(idx_ok)
    print(f"\n{'context':>9} {'组':>8} {'|R_p| 中位':>10} {'IQR':>16} {'p90':>7}")
    rows = []
    for ctx in ("A", "B", "C"):
        mA = mdes[ctx]
        # 把 context 的 MDE 曲线重采样到 K562GW 的基因数（分位匹配）
        thr = np.interp(np.linspace(0, 1, n_g),
                        np.linspace(0, 1, len(mA)), mA)
        for tag, ii in (("V 本届靶", iv), ("E 必需", ie)):
            hits = (b[np.ix_(idx_ok, ii)] > thr[:, None]).sum(0)
            q1, med, q3, p90 = np.percentile(hits, [25, 50, 75, 90])
            rows.append(dict(context=ctx, group=tag, median_Rp=med,
                             q25=q1, q75=q3, p90=p90))
            print(f"{ctx:>9} {tag:>8} {med:10.0f} "
                  f"{f'[{q1:.0f}, {q3:.0f}]':>16} {p90:7.0f}")

    df = pd.DataFrame(rows)
    v_med = float(df[df.group == "V 本届靶"].median_Rp.median())
    e_med = float(df[df.group == "E 必需"].median_Rp.median())

    print(f"\n--- 结论 ---")
    print(f"本届 272 个真实靶基因在真实 context 阈值下 E|R_p| 中位 = {v_med:.0f}")
    print(f"必需基因同条件下                              = {e_med:.0f}")
    print(f"我之前从官方锚点反推的                          = 288 (范围 207–365)")

    if v_med < 120:
        print(f"\n>>> 冲突确认：反推的 288 与真实靶基因的效应量不相容 <<<")
        print(f"    K_p 必须按扰动定在 ~{v_med:.0f} 量级，不是 288。")
        n, K = v_med, 288
        print(f"    代价量化：n={n:.0f} 时，K=288 的 Jaccard 上限 = "
              f"{n/(n+K-n):.3f}（即便全部命中）")
        print(f"              K={n:.0f} 且全部命中则 Jaccard = 1.000")
        print(f"    → 召集集合大小是目前找到的最大杠杆")
    else:
        print(f"\n>>> 无冲突，288 量级成立 <<<")

    print(f"\n耗时 {time.time()-t0:.0f}s")
    df.to_csv(Path(__file__).parent / "result_erp.csv", index=False)


if __name__ == "__main__":
    main()
