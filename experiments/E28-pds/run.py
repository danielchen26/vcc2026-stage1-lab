"""E28: pds_cosine 的本地精确评估器 + 谱设计优化。

## 为什么值得单独做

读源码得到的三件事：

1. `pds_cosine` 是**排名**指标：PDS_p = 1 - rank(self)/D，D = n-1。
   它只要求 pred_eff[p] 离 real_eff[p] 比离 real_eff[j≠p] 更近 —— **只需相对可
   区分性，不需要绝对精度**。而我们的跨系信号是匹配 r=0.339 vs 错配 0.006（55 倍）。

2. `exclusion_scope="panel"`（v2 默认）下无信息提交精确得 **0.5000**
   （源码注释实测三个官方 val context）。所以 baseline = 0.5，
   from_baseline = (pds - 0.5) / (1 - 0.5) = 2*(pds - 0.5)。

3. `scoring=BOUNDED_UNFLOORED` —— **0 处钳位被移除**，pds 差会真实倒扣
   （源码实测过 -1.17/-1.26/-1.22）。

E27 实测我们 pds = 0.5178571 = 1 - 3.375/7，即 8 个扰动上秩和 27 vs 随机 28：
**只比随机好 1 个位次**。信号没进谱里（lfc 只给了召集集合的 29-288 个基因）。

4. 它**只吃 pseudobulk 均值**，不需要 DE。所以本地可精确重算，
   秒级而非 compute_metrics 的 2,600 s —— 这是本脚本存在的理由。

## 阶段

    parity  用官方函数 + 预算 bulk 复现 E27 的 0.5178571（证明 harness 精确）
    design  在同一 harness 上比较若干谱设计，直接读 pds
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import anndata as ad  # noqa: E402
from cell_eval2 import EvalConfig  # noqa: E402
from cell_eval2.metrics.discrimination import discrimination_score  # noqa: E402
from cell_eval2.prep import pseudobulk, pseudobulk_bulk_lognorm  # noqa: E402

E27 = ROOT / "experiments" / "E27-six-metrics" / "out"
OFFICIAL_PDS = 0.5178571428571429      # E27 agg_pred.parquet, 官方 compute_metrics
PERT_COL, CTRL = "target_gene", "non-targeting"


def vcc_cfg():
    cfg = EvalConfig.from_preset("vcc2026")
    return replace(cfg, pert_col=PERT_COL, device="cpu")


def pds(pred_bulk, real_bulk, genes) -> dict[str, float]:
    """官方 discrimination_score，vcc2026 的精确参数。"""
    return discrimination_score(
        pred_bulk=pred_bulk, real_bulk=real_bulk,
        pert_col=PERT_COL, control=CTRL,
        distance="cosine",          # vcc2026 default
        rank_denominator="n-1",     # vcc2026 default
        tie_policy="midrank",       # vcc2026 default
        exclude_target_gene=True,
        exclusion_scope="panel",    # v2 default; 无信息提交在此下精确得 0.5
        control_source="pred",
        genes=genes,
    )


def stage_parity():
    """两种 pseudobulk 空间都试，看哪个复现官方值。"""
    pred = ad.read_h5ad(E27 / "pred.h5ad")
    real = ad.read_h5ad(E27 / "real.h5ad")
    genes = np.asarray(real.var.index.values, dtype=str)
    cfg = vcc_cfg()
    bts = float(cfg.bulk_target_sum)
    print(f"bulk_target_sum = {bts:g}   基因 {genes.size}   "
          f"pred {pred.n_obs} 细胞 / real {real.n_obs} 细胞")

    for name, fn in (
        ("bulk_lognorm", lambda a: pseudobulk_bulk_lognorm(a, PERT_COL, bulk_target_sum=bts)),
        ("plain",        lambda a: pseudobulk(a, PERT_COL)),
    ):
        try:
            got = pds(fn(pred), fn(real), genes)
        except Exception as e:                       # noqa: BLE001
            print(f"  {name:13s} 报错 {type(e).__name__}: {e}")
            continue
        m = float(np.mean(list(got.values())))
        d = abs(m - OFFICIAL_PDS)
        print(f"  {name:13s} pds 均值 {m:.10f}   与官方差 {d:.2e}"
              f"   {'✅ 精确复现' if d < 1e-9 else ''}")
        if d < 1e-9:
            print(f"    逐扰动: " + "  ".join(f"{k}={v:.3f}" for k, v in sorted(got.items())))
            return name
    return None



def _load_beta():
    """复用 E27 的源侧加载：K562GW lfc/se → 全基因下标上的 B/S 矩阵。"""
    import pandas as pd
    DATA = ROOT / "data"
    perts = [l.strip() for l in (E27 / "perts.csv").read_text().split() if l.strip()]
    real = ad.read_h5ad(E27 / "real.h5ad")
    genes = np.asarray(real.var.index.values, dtype=str)
    e2s = pd.read_csv(DATA / "external" / "ens2sym.csv").dropna()

    def rd(n):
        d = pd.read_csv(DATA / "nadig2025" / f"{n}.csv.gz", index_col=0,
                        usecols=["Unnamed: 0"] + perts,
                        dtype={c: np.float32 for c in perts}, engine="c")
        d.index = d.index.astype(str)
        return d[perts]

    lfc_s, se_s = rd("K562GW_lfc"), rd("K562GW_se")
    se_s = se_s.reindex(index=lfc_s.index)
    sym = pd.Index(lfc_s.index.map(dict(zip(e2s.ensembl, e2s.symbol))))
    keep = pd.notna(sym)
    lfc_s, se_s, sym = lfc_s[keep], se_s[keep], sym[keep]
    dd = ~sym.duplicated()
    lfc_s, se_s, sym = lfc_s[dd], se_s[dd], sym[dd]
    common = pd.Index(genes).intersection(sym)
    gi_full = pd.Index(genes).get_indexer(common)
    gi_s = sym.get_indexer(common)
    B = lfc_s.to_numpy().astype(np.float64)[gi_s]
    S = se_s.to_numpy().astype(np.float64)[gi_s]
    return perts, genes, real, gi_full, B, S


def _ctrl_profile(real):
    """real.h5ad 的 non-targeting 细胞 CPM 均值 = m_full。"""
    import scipy.sparse as sp
    tg = np.asarray(real.obs["target_gene"].values, dtype=str)
    X = real.X[tg == CTRL]
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    s = X.sum(1, keepdims=True)
    cpm = np.divide(X, s, out=np.zeros(X.shape, dtype=np.float64), where=s > 0) * 1e6
    return cpm.mean(0)


def _pred_bulk_closed(m_full, lf_mat, perts, bts):
    """闭式 pred_bulk_lognorm。我们的细胞每个恰好和为 1e6，故 池计数/组总 = tgt/1e6，
    无需模拟细胞即可得到官方 bulk_lognorm（parity 阶段已验证空间正确）。"""
    rows, labels = [], []
    for j, p in enumerate(perts):
        tgt = m_full * 2.0 ** lf_mat[j]
        tgt *= 1e6 / tgt.sum()
        rows.append(np.log1p(bts * tgt / 1e6))
        labels.append(p)
    tgt = m_full * (1e6 / m_full.sum())
    rows.append(np.log1p(bts * tgt / 1e6))
    labels.append(CTRL)
    return np.asarray(labels, dtype=str), np.asarray(rows, dtype=np.float64)


def stage_design():
    perts, genes, real, gi_full, B, S = _load_beta()
    m_full = _ctrl_profile(real)
    bts = float(vcc_cfg().bulk_target_sum)
    real_bulk = pseudobulk_bulk_lognorm(real, PERT_COL, bulk_target_sum=bts)
    G = genes.size
    print(f"扰动 {len(perts)}  基因 {G:,}  共同基因 {gi_full.size:,}  bts {bts:g}")

    Z = np.where(np.isfinite(B) & np.isfinite(S) & (S > 0), B / np.maximum(S, 1e-9), 0.0)
    Bc = np.where(np.isfinite(B), B, 0.0)

    def lf_from(vals, topk=None, scale=1.0):
        lf = np.zeros((len(perts), G))
        for j in range(len(perts)):
            v = vals[:, j].copy()
            if topk is not None and topk < v.size:
                v[np.argsort(np.abs(v))[::-1][topk:]] = 0.0
            lf[j, gi_full] = scale * v
        return lf

    designs = [
        ("A 当前：仅召集 288 个", lf_from(Bc, topk=288)),
        ("B 全基因 beta", lf_from(Bc)),
        ("B' 全基因 beta x0.3", lf_from(Bc, scale=0.3)),
        ("C 全基因 beta 去面板均值", lf_from(Bc - Bc.mean(1, keepdims=True))),
        ("D 全基因 z x0.1", lf_from(Z, scale=0.1)),
        ("E z 去面板均值 x0.1", lf_from(Z - Z.mean(1, keepdims=True), scale=0.1)),
        ("F top1000 beta", lf_from(Bc, topk=1000)),
        ("G top1000 去面板均值", lf_from(Bc - Bc.mean(1, keepdims=True), topk=1000)),
    ]
    print(f"\n{'设计':30s} {'pds':>8s} {'from_base':>10s}  逐扰动")
    print("-" * 92)
    best = None
    for name, lf in designs:
        got = pds(_pred_bulk_closed(m_full, lf, perts, bts), real_bulk, genes)
        m = float(np.mean(list(got.values())))
        fb = 2.0 * (m - 0.5)
        print(f"{name:30s} {m:8.4f} {fb:10.4f}  " + " ".join(f"{got[p]:.2f}" for p in perts))
        if best is None or m > best[1]:
            best = (name, m)
    print(f"\n最优 {best[0]}  pds {best[1]:.4f}  from_baseline {2*(best[1]-0.5):.4f}")
    print(f"（当前 A 的官方 from_baseline = 0.0357，六指标总分 0.1110，领先者 0.1899）")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "parity"
    if stage == "design":
        stage_design()
    elif stage == "parity":
        space = stage_parity()
        print(f"\n匹配空间 = {space}")
