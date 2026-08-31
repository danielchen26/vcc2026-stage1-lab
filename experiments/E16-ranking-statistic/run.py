"""E16 — 排序统计量的阶梯：leverage 到底在哪一层。

## 为什么这是关键

三个已测的数并排：
    E11  官方 DE 判定集的对半重叠          h = 0.550   同细胞系
    E15b 按裸 |beta| 排序预测另一半的 R_p   h = 0.200   同细胞系
    E14  跨系 TAP                        h = 0.145   K562 → H1

跨系只损失 27%（0.145 / 0.200）。**瓶颈不是跨系迁移。**
真正的鸿沟是 0.200 vs 0.550 —— 同一个细胞系、同样 400 个细胞，
只因为换了排序统计量就差 2.75 倍。

原因：官方判据是**秩位移**（|psi_bar - 0.5| 超过逐基因阈值），
而阈值随该基因对照分布的形状变动 6.8 倍（MDE 的 p25→p95）。
按裸 |beta| 排序完全忽略这一层。

## 本实验：把阶梯一次量完

每个扰动对半劈开，**只用半 A** 构造排序，靶是**半 B 独立算出的官方 R_p**，
统一用 K = |R_p(B)| 的预言机规模（隔离排序质量）：

    1. |beta_A|                     裸对数倍数变化（E15b 的做法）
    2. |beta_A| / MDE               按逐基因可检出阈值归一 ← 关键对照
    3. -log10(p_adj) from A         **官方统计量本身**，同系上限
    4. 只按 1/MDE                   零生物学地板（E13 已测 0.022）

若 ② 接近 ③，则「用错统计量」是唯一缺口，修法平凡且收益 2.75 倍。
若 ② 仍远低于 ③，则秩位移里有 |beta| 与 MDE 都抓不到的信息，需要保留分布形状。

跑法：  ~/vcc2026/.venv/bin/python experiments/E16-ranking-statistic/run.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import norm, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vcclab.detectability import mde  # noqa: E402
from vcclab.scorer import ControlRef  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "E10-h1-erp"))
from run import VCC_CTRL_CELLS, VCC_PERT_CELLS, VCC_UMI, read_rows, thin, to_cpm  # noqa: E402

H5 = ROOT / "data" / "vcc2025" / "adata_Validation.h5ad"
OUT = Path(__file__).parent
ALPHA = 0.05
Z_BH = 3.184
SEED = 0
EPS = 1.0
METHODS = ("abs_beta", "beta_over_mde", "official_padj", "inv_mde")
LABELS = {"abs_beta": "① |beta|（裸）", "beta_over_mde": "② |beta|/MDE",
          "official_padj": "③ 官方 p_adj（上限）", "inv_mde": "④ 1/MDE（地板）"}


def main() -> None:
    t0 = time.time()
    print("=== E16 排序统计量的阶梯 ===\n")

    with h5py.File(H5, "r") as f:
        genes = [g.decode() if isinstance(g, bytes) else str(g)
                 for g in f["var"]["_index"][:]]
        o = f["obs"]["target_gene"]
        cats = [c.decode() if isinstance(c, bytes) else str(c)
                for c in o["categories"][:]]
        codes = o["codes"][:]
    ntc = cats.index("non-targeting")
    rng = np.random.default_rng(SEED)

    rows = rng.choice(np.flatnonzero(codes == ntc), VCC_CTRL_CELLS, replace=False)
    ctrl_raw = thin(read_rows(rows), VCC_UMI, rng)
    p_tmp = OUT / "_ntc.h5ad"
    ad.AnnData(X=sp.csr_matrix(ctrl_raw),
               var=pd.DataFrame(index=pd.Index(genes))).write_h5ad(p_tmp)
    ref = ControlRef.load(p_tmp, genes)
    gidx = np.asarray(ref.gidx)
    G = len(gidx)
    m = mde(ref, n_cells=VCC_PERT_CELLS, alpha=float(2 * norm.sf(Z_BH)),
            seed=SEED, tie_correct=True)
    m_safe = np.where(np.isfinite(m) & (m > 0), m, np.inf)
    ctrl_cpm = np.asarray(ctrl_raw[:, gidx].multiply(
        1e6 / np.maximum(np.asarray(ctrl_raw.sum(1)), 1.0)).mean(0)).ravel()
    print(f"gate = {G:,}   MDE 中位 {np.nanmedian(m):.4f}   "
          f"p25/p95 = {np.nanpercentile(m,25):.4f}/{np.nanpercentile(m,95):.4f}"
          f"  （动态范围 {np.nanpercentile(m,95)/np.nanpercentile(m,25):.1f}×）")

    need = 2 * VCC_PERT_CELLS
    print(f"\n{'扰动':>10} {'|R_p(B)|':>9} " +
          " ".join(f"{LABELS[k][:12]:>13}" for k in METHODS))
    recs = []
    for ci, name in enumerate(cats):
        if ci == ntc:
            continue
        idx = np.flatnonzero(codes == ci)
        if len(idx) < need:
            continue
        pick = rng.permutation(idx)[:need]
        a_sel, b_sel = pick[:VCC_PERT_CELLS], pick[VCC_PERT_CELLS:]

        Ma = thin(read_rows(a_sel), VCC_UMI, rng)
        cpm_a = to_cpm(Ma)
        padj_a, lfc_a = ref.de_table(cpm_a, tie_correct=True)
        pm_a = cpm_a[:, gidx].mean(0)
        beta_a = np.log2((pm_a + EPS) / (ctrl_cpm + EPS))

        padj_b, _ = ref.de_table(to_cpm(thin(read_rows(b_sel), VCC_UMI, rng)),
                                 tie_correct=True)
        real = padj_b < ALPHA
        n_real = int(real.sum())
        if n_real < 5:
            continue

        score = {
            "abs_beta": np.abs(beta_a),
            "beta_over_mde": np.abs(beta_a) / m_safe,
            "official_padj": -np.log10(np.maximum(padj_a, 1e-300)),
            "inv_mde": 1.0 / m_safe,
        }
        rec = dict(target_gene=name, n_real=n_real)
        for k, s in score.items():
            top = np.argsort(s)[::-1][:n_real]
            rec[f"h_{k}"] = float(real[top].sum()) / n_real
        recs.append(rec)
        print(f"{name:>10} {n_real:9d} " +
              " ".join(f"{rec['h_'+k]:13.3f}" for k in METHODS))

    df = pd.DataFrame(recs)
    df.to_csv(OUT / "result.csv", index=False)
    p_tmp.unlink(missing_ok=True)

    print(f"\n{'='*74}\n结论（n = {len(df)} 个扰动，预言机规模 K = |R_p(B)|）\n{'='*74}")
    print(f"{'排序统计量':>22} {'h 中位':>9} {'IQR':>18} {'相对 ①':>9}")
    base = df.h_abs_beta.median()
    for k in METHODS:
        v = df[f"h_{k}"]
        q = np.percentile(v, [25, 50, 75])
        print(f"{LABELS[k]:>22} {q[1]:9.3f} {f'[{q[0]:.3f}, {q[2]:.3f}]':>18} "
              f"{q[1]/base:8.2f}×")

    print(f"\n--- 配对检验 ---")
    for k in ("beta_over_mde", "official_padj"):
        d = df[f"h_{k}"] - df.h_abs_beta
        p = wilcoxon(df[f"h_{k}"], df.h_abs_beta).pvalue
        print(f"  {LABELS[k]:>22} vs ①: Δh 中位 {d.median():+.4f}  "
              f"胜 {(d>0).sum()}/{len(df)}  p = {p:.5f}")
    d = df.h_official_padj - df.h_beta_over_mde
    p = wilcoxon(df.h_official_padj, df.h_beta_over_mde).pvalue
    print(f"  {'③ vs ②':>22}: Δh 中位 {d.median():+.4f}  "
          f"胜 {(d>0).sum()}/{len(df)}  p = {p:.5f}")

    h2, h3 = df.h_beta_over_mde.median(), df.h_official_padj.median()
    print(f"\n  ② / ③ = {h2/h3:.1%}"
          f"  → {'MDE 归一化已经吃掉大部分缺口' if h2/h3 > 0.8 else '秩位移里还有 |beta| 与 MDE 抓不到的信息'}")
    print(f"\n  E14 的跨系 TAP h = 0.145，相对同系上限 ③ {0.145/h3:.1%}")
    print(f"  → 缺口分解：跨系损失 {1-0.145/h3:.0%}，"
          f"统计量损失（①→③）{1-base/h3:.0%}")
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
