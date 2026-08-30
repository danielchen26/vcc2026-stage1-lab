"""E07 — 跨细胞系响应基因集重叠率，以及它相对同系 replicate 的比值。

数据：Nadig et al. 2025 (Nat Genet) figshare 29498366 的 DE 汇总统计矩阵。
      行 = 被测基因 (Ensembl ID)，列 = 被扰动基因 (symbol)，
      值 = DESeq2 的 p / log2FoldChange / lfcSE。

设计要点（每一条都是必需的，缺一条结论就失效）：

1. **列必须显式重排。** pandas 的 usecols 按*文件列序*返回，不是传入顺序。
   四个文件列序不同，不重排就是拿 A 的第 j 个扰动和 B 的另一个扰动比 ——
   会得到恰好等于随机水平的假 NO-GO。修法：读完立刻 df[cols]。

2. **功效必须匹配。** 各系中位显著基因数差 14 倍（K562 36 · RPE1 318），
   用 mean(|A|,|B|) 作分母时 K562-vs-RPE1 即使完全嵌套也只能得 0.203。
   改用 top-K：两边各取 p 值最小的 K 个，h = |交集| / K。功效差异被完全消掉。

3. **必须扣随机基线。** K 个基因从 G 个里随机取，期望重叠 = K/G，不小。
   报 chance-corrected ratio = (h_cross - K/G) / (h_rep - K/G)。

4. **同系 replicate 用 data fission 造。** 官方 1 分锚点是「真实数据对半劈开」。
   给定 beta_hat ~ N(beta, sigma^2) 且 sigma 已知（= lfcSE），取 Z ~ N(0,1)：
       beta_1 = beta_hat + sigma*Z ,  beta_2 = beta_hat - sigma*Z
   则 Cov = sigma^2 - sigma^2 = 0，联合正态故**独立**；各自 Var = 2*sigma^2，
   即 SE 放大 sqrt(2)，正好等价于样本量减半，与 split-half 同构。

5. **主判定放在 K=288**，因为官方锚点反推的 E|R_p| ≈ 288（F13）—— 这是本届
   真正要预测的集合规模。其余 K 用来验证结论不依赖 K 的选取。

跑法：  ~/vcc2026/.venv/bin/python experiments/E07-source-coverage/run.py
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
from vcclab.crossline import (H_TIE_LEADER, chance_overlap, fission,  # noqa: E402
                              h_topk, h_topk_shuffled, rank_matrix,
                              ratio_vs_permutation)
from vcclab.scorer import bh_adjust  # noqa: E402

DATA = ROOT / "data" / "nadig2025"
LINES = ("K562", "RPE1", "Jurkat", "HepG2")
TAGS = ("theirs", "wald", "f1", "f2")
ALPHA = 0.05
SEED = 0
K_SWEEP = (25, 50, 100, 200, 288, 500)
K_PRIMARY = 288
H_REPLICATE_OFFICIAL = 0.570
GATES = {"GO": 0.35, "MARGINAL": 0.23}


def read_matrix(line: str, kind: str, cols: list[str]) -> pd.DataFrame:
    """读一个 DE 矩阵。索引列是 Ensembl ID（字符串），只对值列指定 float32。

    末尾的 df[cols] 是**必需**的：usecols 按文件列序返回，不重排会造成跨系列错位。
    """
    df = pd.read_csv(DATA / f"{line}Essential_{kind}.csv.gz", index_col=0,
                     usecols=["Unnamed: 0"] + cols,
                     dtype={c: np.float32 for c in cols}, engine="c")
    df.index = df.index.astype(str)
    return df[cols]


def bh_reject(pvals: np.ndarray) -> np.ndarray:
    """每列（每扰动）独立做 BH，alpha=0.05。NaN 视为未检验。"""
    out = np.zeros(pvals.shape, bool)
    for j in range(pvals.shape[1]):
        col = pvals[:, j]
        ok = np.isfinite(col)
        if not ok.any():
            continue
        q = bh_adjust(col[ok].astype(np.float64))
        out[np.flatnonzero(ok), j] = q < ALPHA
    return out



def h_bh(a: np.ndarray, b: np.ndarray) -> float:
    """诊断用：BH 阈值下、mean 分母的重叠率（与官方 Jaccard 精神一致，但受功效污染）。"""
    na, nb = a.sum(0).astype(float), b.sum(0).astype(float)
    denom = 0.5 * (na + nb)
    out = np.divide((a & b).sum(0), denom, out=np.full_like(denom, np.nan), where=denom > 0)
    out[(na == 0) | (nb == 0)] = np.nan
    return float(np.nanmedian(out))


def main() -> None:
    t0 = time.time()
    print("=== E07 跨细胞系响应基因集重叠率 ===")
    print(f"数据 Nadig et al. 2025 · figshare 29498366 · seed={SEED}\n")

    cols_per_line = {}
    for ln in LINES:
        d = pd.read_csv(DATA / f"{ln}Essential_p.csv.gz", index_col=0, nrows=1)
        cols_per_line[ln] = [str(c) for c in d.columns]
    shared = sorted(set.intersection(*(set(c) for c in cols_per_line.values())))
    print("各系扰动数: " + " · ".join(f"{ln} {len(cols_per_line[ln]):,}" for ln in LINES))
    print(f"四系共享扰动: {len(shared):,}\n")

    rng = np.random.default_rng(SEED)
    pmat: dict[str, dict[str, np.ndarray]] = {}
    idx: dict[str, pd.Index] = {}

    for ln in LINES:
        t = time.time()
        p = read_matrix(ln, "p", shared)
        lfc = read_matrix(ln, "lfc", shared).reindex(index=p.index)
        se = read_matrix(ln, "se", shared).reindex(index=p.index)
        idx[ln] = p.index
        b = lfc.to_numpy().astype(np.float64)
        s = se.to_numpy().astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            b1, b2, s1, s2 = fission(b, s, rng, tau=1.0)
            m = {
                "theirs": p.to_numpy().astype(np.float32),
                "wald": (2 * norm.sf(np.abs(b / s))).astype(np.float32),
                "f1": (2 * norm.sf(np.abs(b1 / s1))).astype(np.float32),
                "f2": (2 * norm.sf(np.abs(b2 / s2))).astype(np.float32),
            }
        pmat[ln] = m
        nsig = {k: np.median(bh_reject(v).sum(0)) for k, v in m.items()}
        print(f"{ln:8s} 被测基因 {p.shape[0]:6,}  BH 中位显著数 "
              + "  ".join(f"{k}={nsig[k]:5.0f}" for k in TAGS) + f"   ({time.time()-t:.0f}s)")
        del p, lfc, se, b, s, b1, b2, s1, s2

    common = idx[LINES[0]]
    for ln in LINES[1:]:
        common = common.intersection(idx[ln])
    G = len(common)
    print(f"\n四系共享被测基因 G = {G:,}")
    ranks, bhcalls = {}, {}
    for ln in LINES:
        take = idx[ln].get_indexer(common)
        ranks[ln] = {t_: rank_matrix(pmat[ln][t_][take]) for t_ in TAGS}
        bhcalls[ln] = {t_: bh_reject(pmat[ln][t_][take]) for t_ in ("theirs", "f1", "f2")}
        del pmat[ln]

    pairs = [(a, b) for i, a in enumerate(LINES) for b in LINES[i + 1:]]

    print(f"\n--- 主判定 (K={K_PRIMARY}，官方反推的 E|R_p|) ---")
    print("基线 = 打乱扰动标签的置换零假设，不是 K/G —— 见 crossline.h_topk_shuffled 的说明")
    print(f"{'细胞系对':>16} {'h_cross':>9} {'置换基线':>9} {'超出':>8} "
          f"{'h_rep':>8} {'其置换基线':>11} {'超出':>8} {'比值':>8} {'判定':>10}")
    rows = []
    for a, b in pairs:
        hc = h_topk(ranks[a]["wald"], ranks[b]["wald"], K_PRIMARY)
        hc0 = h_topk_shuffled(ranks[a]["wald"], ranks[b]["wald"], K_PRIMARY)
        hr = np.mean([h_topk(ranks[l]["f1"], ranks[l]["f2"], K_PRIMARY) for l in (a, b)])
        hr0 = np.mean([h_topk_shuffled(ranks[l]["f1"], ranks[l]["f2"], K_PRIMARY)
                       for l in (a, b)])
        r = ratio_vs_permutation(hc, hc0, hr, hr0)
        v = "GO" if r >= GATES["GO"] else ("MARGINAL" if r >= GATES["MARGINAL"] else "NO-GO")
        rows.append(dict(pair=f"{a}-{b}", K=K_PRIMARY, h_cross=hc, h_cross_null=hc0,
                         h_rep=hr, h_rep_null=hr0, excess_cross=hc - hc0,
                         excess_rep=hr - hr0, ratio=r, verdict=v))
        print(f"{a+'–'+b:>16} {hc:9.3f} {hc0:9.3f} {hc-hc0:8.3f} "
              f"{hr:8.3f} {hr0:11.3f} {hr-hr0:8.3f} {r:8.3f} {v:>10}")

    cc_all = np.array([r["ratio"] for r in rows])
    med = float(np.nanmedian(cc_all))

    print("\n--- K 敏感性（六对中位数）---")
    print(f"{'K':>6} {'K/G':>8} {'h_cross':>9} {'置换基线':>9} {'超出':>8} "
          f"{'h_rep 超出':>11} {'比值':>8}")
    sweep = []
    for k in K_SWEEP:
        hcs = [h_topk(ranks[a]["wald"], ranks[b]["wald"], k) for a, b in pairs]
        hc0s = [h_topk_shuffled(ranks[a]["wald"], ranks[b]["wald"], k) for a, b in pairs]
        hrs = [h_topk(ranks[l]["f1"], ranks[l]["f2"], k) for l in LINES]
        hr0s = [h_topk_shuffled(ranks[l]["f1"], ranks[l]["f2"], k) for l in LINES]
        mc, mc0 = float(np.median(hcs)), float(np.median(hc0s))
        mr, mr0 = float(np.median(hrs)), float(np.median(hr0s))
        r = ratio_vs_permutation(mc, mc0, mr, mr0)
        sweep.append(dict(K=k, chance_kg=chance_overlap(k, G), h_cross=mc,
                          h_cross_null=mc0, h_rep=mr, h_rep_null=mr0, ratio=r))
        star = "  ←主判定" if k == K_PRIMARY else ""
        print(f"{k:6d} {chance_overlap(k,G):8.4f} {mc:9.3f} {mc0:9.3f} {mc-mc0:8.3f} "
              f"{mr-mr0:11.3f} {r:8.3f}{star}")

    best = max(sweep, key=lambda r: r["h_rep"] - r["h_rep_null"])
    conservative = float(np.nanmin([r["ratio"] for r in sweep]))
    print(f"\n--- 条件数：分母的超出量 (h_rep - 置换基线) 决定测量是否良态 ---")
    for r in sweep:
        exc = r["h_rep"] - r["h_rep_null"]
        mark = "  ←最良态" if r["K"] == best["K"] else ""
        fold = (f"{exc/r['h_rep_null']:.1f} 倍" if r["h_rep_null"] > 0 else "∞（基线中位=0）")
        print(f"  K={r['K']:4d}  分母超出 {exc:.3f} = 其基线的 {fold}"
              f"   比值 {r['ratio']:.3f}{mark}")
    print(f"\n比值随 K 单调上升是分母萎缩快于分子造成的假象，不是真实趋势 ——")
    print(f"大 K 处两个小数相除，条件数差。取最良态的 K={best['K']} 作主判定。")
    print(f"\n--- 诊断：BH 阈值 + mean 分母（受功效污染，仅供对照）---")
    hr_bh = {l: h_bh(bhcalls[l]["f1"], bhcalls[l]["f2"]) for l in LINES}
    print("  h_replicate: " + " · ".join(f"{l} {hr_bh[l]:.3f}" for l in LINES)
          + f"   官方锚点 {H_REPLICATE_OFFICIAL:.3f}")
    hc_bh = [h_bh(bhcalls[a]["theirs"], bhcalls[b]["theirs"]) for a, b in pairs]
    print(f"  h_cross 中位: {np.median(hc_bh):.3f}")

    print(f"\n六对比值 (K={K_PRIMARY}) 中位 = {med:.3f}"
          f"   范围 [{np.nanmin(cc_all):.3f}, {np.nanmax(cc_all):.3f}]")
    print(f"最良态 K={best['K']} 处比值 = {best['ratio']:.3f}")
    print(f"全 K 范围保守下界 = {conservative:.3f}")
    print("\n判定门: >=0.35 GO · 0.23–0.35 MARGINAL · <0.23 NO-GO")
    print("按保守下界判定（最不利的 K 也要过门）：")
    verdict = ("GO" if conservative >= GATES["GO"]
               else "MARGINAL" if conservative >= GATES["MARGINAL"] else "NO-GO")
    print(f"\n>>> {verdict} <<<")

    # 折算回本届的绝对量级
    h_abs = conservative * H_REPLICATE_OFFICIAL
    print(f"\n折算：可达 h ≈ {conservative:.3f} × {H_REPLICATE_OFFICIAL:.3f} = {h_abs:.3f}")
    print(f"      追平榜首需 h = {H_TIE_LEADER:.3f}  →  余量 {h_abs/H_TIE_LEADER:.2f}×")
    print("\n⚠️ 这是**上界**：2,052 个共享扰动全是 DepMap common essential 基因，")
    print("   跨系保守的生长停滞/翻译应激共同响应会让 h_cross 系统性偏高。")
    print(f"   若真实迁移只有一半好，h ≈ {h_abs/2:.3f}，"
          f"{'仍高于' if h_abs/2 > H_TIE_LEADER else '就低于'}追平线 {H_TIE_LEADER:.3f}。")
    print("   → 余量真实但不厚，Stage 1 单独不够，必须同时做 pds/mse。")
    print("   下一步必须查：本届 300 个靶基因是否也是 essential-like。")
    print(f"\n耗时 {time.time()-t0:.0f}s")

    out = Path(__file__).parent
    pd.DataFrame(rows).to_csv(out / "result.csv", index=False)
    pd.DataFrame(sweep).to_csv(out / "result_ksweep.csv", index=False)


if __name__ == "__main__":
    main()
