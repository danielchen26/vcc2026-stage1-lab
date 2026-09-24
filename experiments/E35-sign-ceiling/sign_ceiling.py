"""E35 · 符号一致率的天花板是「跨细胞系」还是「源表本身」？

为什么这是现在唯一值得做的检验：
  [F36](../../docs/02-findings.md#f36) 把解码器旋钮判出局；剩下 `reach` + `nmae` 合计
  +0.0720 锁在同一个前提上 —— **跨系可迁移的 lfc 符号**（F33 + F35）。
  路线图据此把新 P0 定为「去找符号可迁移的源」（Replogle 2022 / Jiang 2025）。
  但那两个源**不在本地**，Replogle 原始数据 61.3 GB 而磁盘只剩 10 GB。

  在花下载和算力之前，F33 的表里有一行需要重读：

      K562GW vs K562Essential   57.5%   <- 同一个细胞系，两次不同筛
      K562GW vs JurkatEssential 56.8%   <- 跨系
      K562GW vs RPE1Essential   52.9%   <- 跨系
      K562GW vs HepG2Essential  51.1%   <- 跨系

  **同系 57.5% 与跨系 51–57% 几乎没有差别。** 若这条在大样本上成立，瓶颈就不是
  「跨细胞系不可迁移」，而是**这些 DE 表自己的符号就是噪声** —— 那么换细胞系、
  下载 Replogle 都不解决问题，新 P0 的前提就是错的。

  F33 的 57.5% 只在 **MAT2A 一个扰动**上测得（8 面板里唯一多源共有的）。
  全表共有的扰动有上千个，本实验在全部共有扰动上测，给出分布而非单点。

判读（执行前写下）：
  - 若同系 >> 跨系（差 ≥ 10 个百分点）：跨系确实是瓶颈，新 P0 成立，值得找新源。
  - 若同系 ≈ 跨系（差 < 5 个百分点）：瓶颈是源表的符号精度，**新 P0 的前提错了**，
    换源无用；要么提高每基因估计的功效，要么承认这条路需要别的模型。

⚠️ 先自证：若 K562GW 与 K562Essential 在共有扰动上数值**逐位相同**，说明它们是同一批
测量的子集，比较无意义。脚本会先查这一条并在相同处报警。

跑法：/Users/chetianc/vcc2026/.venv/bin/python experiments/E35-sign-ceiling/sign_ceiling.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "data" / "nadig2025"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05
REF = "K562GW"
PAIRS = [("K562GW", "K562Essential", "同系（K562，两次不同筛）"),
         ("K562GW", "JurkatEssential", "跨系 K562→Jurkat"),
         ("K562GW", "RPE1Essential", "跨系 K562→RPE1"),
         ("K562GW", "HepG2Essential", "跨系 K562→HepG2")]


def cols_of(name: str) -> list[str]:
    """表头 = 扰动名（第一列是基因索引，列名为空串）。"""
    head = pl.read_csv(DATA / f"{name}_lfc.csv.gz", n_rows=0)
    return [c for c in head.columns if c != ""]


def load(name: str, keep: list[str], what: str) -> tuple[np.ndarray, np.ndarray]:
    """读 lfc 或 p 的指定列 + 基因索引。返回 (基因 id, 矩阵 基因×扰动)。

    源表用 R 的 `NA` 表示缺失（实测：K562Essential_lfc 第 8559 列 C12orf60）。
    必须显式声明为 null，否则 polars 把整列推断失败而报 ComputeError；
    绝不能用 `ignore_errors=True` —— 那会把解析失败的值静默变成 null，
    分不清「源里确实缺」与「我读错了」。
    """
    df = pl.read_csv(DATA / f"{name}_{what}.csv.gz", columns=[""] + keep,
                     null_values=["NA", "NaN", "NULL", "Inf", "-Inf"],
                     schema_overrides={c: pl.Float64 for c in keep})
    genes = df[""].to_numpy().astype(str)
    mat = df.select(keep).to_numpy().astype(np.float64)
    return genes, mat


def bh(p: np.ndarray) -> np.ndarray:
    """BH 校正，逐列。NaN 原样返回 NaN。"""
    out = np.full_like(p, np.nan)
    for j in range(p.shape[1]):
        col = p[:, j]
        ok = np.isfinite(col)
        v = col[ok]
        if v.size == 0:
            continue
        order = np.argsort(v)
        ranked = v[order]
        n = v.size
        adj = ranked * n / np.arange(1, n + 1)
        adj = np.minimum.accumulate(adj[::-1])[::-1]
        res = np.empty(n)
        res[order] = np.clip(adj, 0, 1)
        out[ok, j] = res
    return out


def main() -> None:
    t0 = time.time()
    ref_cols = set(cols_of(REF))
    print(f"{REF}: {len(ref_cols):,} 个扰动")

    report = {}
    for a, b, label in PAIRS:
        b_cols = cols_of(b)
        shared = sorted(ref_cols.intersection(b_cols))
        print(f"\n=== {label}  {a} ∩ {b} = {len(shared):,} 个共有扰动 "
              f"（{b} 自身 {len(b_cols):,}）===")
        if not shared:
            print("  无共有扰动，跳过")
            continue

        ga, la = load(a, shared, "lfc")
        gb, lb = load(b, shared, "lfc")
        _, pb = load(b, shared, "p")
        # 两个源表的基因索引不同（实测：K562GW 与 K562Essential 不等长/不同集合），
        # 所以按基因名取交集后重排两侧，绝不假设同序。各自表内先去重（保留首次出现），
        # 否则 get_indexer 会在重复名上任意选行。
        def _dedup(g):
            _, first = np.unique(g, return_index=True)
            return np.sort(first)
        ia, ib = _dedup(ga), _dedup(gb)
        ga, la = ga[ia], la[ia]
        gb, lb, pb = gb[ib], lb[ib], pb[ib]
        common_g, ja, jb = np.intersect1d(ga, gb, return_indices=True)
        print(f"  基因：{a} {ga.size:,} / {b} {gb.size:,} → 交集 {common_g.size:,}")
        assert common_g.size > 1000, f"基因交集只有 {common_g.size}，两表 ID 空间可能不同"
        ga, la, lb, pb = common_g, la[ja], lb[jb], pb[jb]

        # 自证：两个源若逐位相同，则是同一批测量，比较无意义
        both = np.isfinite(la) & np.isfinite(lb)
        ident = np.isclose(la[both], lb[both], rtol=0, atol=0).mean() if both.any() else 0.0
        if ident > 0.99:
            print(f"  ⚠️ 逐位相同比例 {ident:.4f} > 0.99 —— 同一批测量，比较无意义，跳过")
            continue
        print(f"  逐位相同比例 {ident:.6f}（远小于 1 ⇒ 独立测量，可比）")

        # 门：以 b 为「真」侧，取 b 自己的 BH 显著集（镜像 nmae/reach 的 real 侧门）
        qb = bh(pb)
        per_pert, gate_sizes = [], []
        for j, pert in enumerate(shared):
            gate = both[:, j] & np.isfinite(qb[:, j]) & (qb[:, j] < ALPHA)
            gate &= (ga != pert)                      # 剔除被扰动基因自身（issue #172）
            n = int(gate.sum())
            if n < 10:                                # 镜像 min_gate_size=10
                continue
            agree = float((np.sign(la[gate, j]) == np.sign(lb[gate, j])).mean())
            per_pert.append(agree)
            gate_sizes.append(n)
        if not per_pert:
            print("  所有共有扰动的门都 < 10 个基因，跳过")
            continue

        arr = np.asarray(per_pert)
        gs = np.asarray(gate_sizes)
        # 掷硬币的标准误：门内 n 个基因，单扰动 sd = 0.5/sqrt(n)
        se_null = float(np.mean(0.5 / np.sqrt(gs)))
        z = (arr.mean() - 0.5) / (arr.std(ddof=1) / np.sqrt(arr.size))
        report[label] = {"pair": f"{a}->{b}", "n_shared": len(shared),
                         "n_scored": int(arr.size), "mean": float(arr.mean()),
                         "median": float(np.median(arr)), "sd": float(arr.std(ddof=1)),
                         "q10": float(np.quantile(arr, 0.10)),
                         "q90": float(np.quantile(arr, 0.90)),
                         "frac_above_half": float((arr > 0.5).mean()),
                         "median_gate": int(np.median(gs)), "se_null_per_pert": se_null,
                         "t_vs_half": float(z)}
        r = report[label]
        print(f"  计分扰动 {r['n_scored']:,}（门 ≥ 10）  门中位 {r['median_gate']:,}")
        print(f"  一致率 均值 {r['mean']:.4f}  中位 {r['median']:.4f}  "
              f"sd {r['sd']:.4f}  10–90 分位 [{r['q10']:.4f}, {r['q90']:.4f}]")
        print(f"  > 0.5 的扰动占比 {r['frac_above_half']:.3f}   "
              f"t vs 0.5 = {r['t_vs_half']:.1f}")

    print("\n" + "=" * 92)
    print("判读（门在执行前写下）")
    print("=" * 92)
    same = report.get("同系（K562，两次不同筛）")
    cross = [v for k, v in report.items() if k.startswith("跨系")]
    if same and cross:
        best_cross = max(cross, key=lambda v: v["mean"])
        gap = (same["mean"] - best_cross["mean"]) * 100
        print(f"同系 {same['mean']:.4f}（n={same['n_scored']:,}）  "
              f"最好的跨系 {best_cross['pair']} {best_cross['mean']:.4f}"
              f"（n={best_cross['n_scored']:,}）  差 {gap:+.2f} 个百分点")
        if gap >= 10:
            print("⇒ 跨系是瓶颈，新 P0（找符号可迁移的源）成立，值得找新源。")
        elif gap < 5:
            print("⇒ 同系 ≈ 跨系：瓶颈是**源表自己的符号精度**，不是细胞系差异。")
            print("  **新 P0 的前提是错的** —— 换源/下载 Replogle 不解决问题。")
        else:
            print("⇒ 介于 5–10 个百分点之间，不构成判定，需要第三个同系对。")
    json.dump(report, open(OUT / "sign_ceiling.json", "w"), indent=2, ensure_ascii=False)
    print(f"\n已存 {OUT/'sign_ceiling.json'}   耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
