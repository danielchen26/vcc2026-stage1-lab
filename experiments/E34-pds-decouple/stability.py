"""E34 · `k_off` 的稳定性检验 —— `k=500` 的 0.8750 是真优化点还是 panel 噪声？

为什么必须做这一步：
  `decouple_sweep.log` 里 k 扫描出一个**孤立尖峰**
      k=300 → 0.7679   k=500 → 0.8750   k=1000 → 0.8036   k=2000/3000 → 0.7857
  而 `pds` 是**组内排名**（`rank_denominator="n-1"`、`exclusion_scope="panel"`，
  见 run.py:58-68），8 个扰动 ⇒ 每扰动量子 1/7、均值量子 1/56 = 0.0179。
  在**同一批 8 个扰动**上扫 k 再在同一批上评分 = 在测试集上选参数（[T8](../../docs/06-traps.md#t8)
  「pds 依赖 panel 组成」+ 迭代纪律「不要记忆某个集合」）。

  逐扰动读数进一步加深怀疑：k=500 相对 k=1000 的全部增益来自 TCF7L2（0.57→1.00）
  与 GNG12（0.29→0.57）—— 恰好是基线上最难的两个（pds 0.43 / 0.14），
  且源侧显著基因数最少（n_src = 1 与 5）。

两个检验：
  1. **密网格**：真优化点应当有邻域。若 500 两侧立刻掉回 0.78–0.80，它是单点尖峰。
  2. **留一法**：逐个剔除一个扰动，在剩下 7 个构成的 panel 上重算。
     若 k=500 只在少数几个 LOO panel 上胜过 k=1000，它是 panel 噪声。
     ⚠️ LOO 会改变指标本身（panel 变了，量子变成 1/42），所以**只比较同一 LOO panel 内
     的 k 之间**，绝不跨 panel 比绝对值。

跑法：/Users/chetianc/vcc2026/.venv/bin/python experiments/E34-pds-decouple/stability.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "e28run", ROOT / "experiments" / "E28-pds" / "run.py")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

K_CALL = 288
DENSE_K = (300, 350, 400, 450, 500, 550, 600, 700, 800, 1000)
LOO_K = (500, 1000)


def main() -> None:
    perts, genes, real, gi_full, B, S = m._load_beta()
    m_full = m._ctrl_profile(real)
    bts = float(m.vcc_cfg().bulk_target_sum)
    real_bulk = m.pseudobulk_bulk_lognorm(real, m.PERT_COL, bulk_target_sum=bts)
    G = genes.size
    n_p = len(perts)
    print(f"扰动 {n_p}  基因 {G:,}  共同基因 {gi_full.size:,}")
    print(f"扰动顺序 {list(perts)}")
    print(f"real_bulk 类型 {type(real_bulk).__name__}")

    Bc = np.where(np.isfinite(B), B, 0.0)
    Bd = Bc - Bc.mean(1, keepdims=True)

    def _topk_mask(v, k):
        if k is None or k >= v.size:
            return np.ones(v.size, bool)
        keep = np.zeros(v.size, bool)
        keep[np.argsort(np.abs(v))[::-1][:k]] = True
        return keep

    def lf_design(k_off, cols=None):
        """设计 I：召集 288 原值 + 集外 top-k_off 去均值（scale 1.0）。

        `cols` 限定要生成哪些扰动列（LOO 用）。返回 (n_sel, G)。
        """
        idx = range(n_p) if cols is None else cols
        lf = np.zeros((len(list(idx)), G))
        for row, j in enumerate(idx):
            call = _topk_mask(Bc[:, j], K_CALL)
            v = np.where(call, Bc[:, j], 0.0)
            off = Bd[:, j].copy()
            off[call] = 0.0
            if k_off is not None:
                off[~_topk_mask(off, k_off)] = 0.0
            lf[row, gi_full] = v + off
        return lf

    def lf_baseline(cols=None):
        """设计 A：只有召集 288，原值。"""
        idx = range(n_p) if cols is None else cols
        lf = np.zeros((len(list(idx)), G))
        for row, j in enumerate(idx):
            call = _topk_mask(Bc[:, j], K_CALL)
            lf[row, gi_full] = np.where(call, Bc[:, j], 0.0)
        return lf

    # ---- real_bulk 的子集化：兼容 (labels, matrix) 与 AnnData 两种返回 ----
    if isinstance(real_bulk, tuple):
        lab_r, mat_r = real_bulk

        def sub_real(keep_labels):
            sel = [i for i, l in enumerate(lab_r) if l in keep_labels]
            return (np.asarray(lab_r)[sel], np.asarray(mat_r)[sel])
    else:
        def sub_real(keep_labels):
            mask = [l in keep_labels for l in real_bulk.obs[m.PERT_COL].astype(str)]
            return real_bulk[np.asarray(mask)]

    def mean_pds(lf, cols=None):
        idx = list(range(n_p)) if cols is None else list(cols)
        ps = [perts[j] for j in idx]
        pb = m._pred_bulk_closed(m_full, lf, ps, bts)
        rb = real_bulk if cols is None else sub_real(set(ps) | {m.CTRL})
        got = m.pds(pb, rb, genes)
        return float(np.mean(list(got.values()))), got

    # ================= 检验 1：密网格 =================
    print("\n" + "=" * 78)
    print("检验 1 · 密网格（全 8 扰动 panel，量子 1/56 = 0.0179）")
    print("=" * 78)
    base, _ = mean_pds(lf_baseline())
    print(f"{'k_off':>8s} {'pds':>8s} {'Δ vs A':>9s} {'量子数':>7s}")
    print("-" * 40)
    print(f"{'A(无)':>8s} {base:8.4f} {0.0:+9.4f} {0:7d}")
    dense = {}
    for k in DENSE_K:
        mu, _ = mean_pds(lf_design(k))
        dense[k] = mu
        print(f"{k:8d} {mu:8.4f} {mu-base:+9.4f} {round((mu-base)*56):7d}")
    peak = max(dense, key=dense.get)
    nb = [k for k in DENSE_K if k != peak]
    runner = max(nb, key=dense.get)
    print(f"\n峰值 k={peak} pds {dense[peak]:.4f}；次优 k={runner} pds {dense[runner]:.4f}；"
          f"差 {dense[peak]-dense[runner]:+.4f}（{round((dense[peak]-dense[runner])*56)} 个量子）")

    # ================= 检验 2：留一法 =================
    print("\n" + "=" * 78)
    print("检验 2 · 留一法（每次剔除 1 个扰动，7 扰动 panel，量子 1/42 = 0.0238）")
    print("⚠️ 只在同一行内横向比较 k；绝不跨行比绝对值（panel 不同，指标不同）")
    print("=" * 78)
    hdr = f"{'剔除':>10s} {'A':>8s}" + "".join(f"{'k='+str(k):>9s}" for k in LOO_K) + f"{'胜者':>9s}"
    print(hdr)
    print("-" * len(hdr))
    wins = {k: 0 for k in LOO_K}
    for h in range(n_p):
        cols = [j for j in range(n_p) if j != h]
        b_loo, _ = mean_pds(lf_baseline(cols), cols)
        vals = {}
        for k in LOO_K:
            vals[k], _ = mean_pds(lf_design(k, cols), cols)
        win = max(vals, key=vals.get)
        wins[win] += 1
        print(f"{perts[h]:>10s} {b_loo:8.4f}"
              + "".join(f"{vals[k]:9.4f}" for k in LOO_K)
              + f"{'k='+str(win):>9s}")
    print("-" * len(hdr))
    print(f"胜场：" + "  ".join(f"k={k} → {wins[k]}/{n_p}" for k in LOO_K))
    print("\n判读：若 k=500 的胜场 ≤ 4/8，它在 8 个 panel 里不稳定 ⇒ 是 panel 噪声，"
          "\n      应取有邻域支撑的 k（密网格里的平台），而不是全局 argmax。")


if __name__ == "__main__":
    main()
