"""变体 V2：多项式抽样 + top1000 去面板均值 + 1e6 深度。

与 E27 的 variant A 的差别，全部有实测依据：
  1. hamilton 确定性最大余额法 → 多项式抽样
     实测：同一设计 pds from_baseline 0.0357 → 0.4286（+0.393，纯实现修复）
     根因：hamilton 让 CPM<~0.5 的基因在每个细胞都分到 0 计数，2264 个基因
           恒为零，在每一行产生相同伪影，把所有余弦拉到一起。
  2. 放弃两点 psi_bar 构造
     实测交换比：它买到 sig_jaccard 的 +0.001，代价 0.214 原始 pds = 0.43 from_baseline。
  3. lfc 用 top1000 |beta| 并去面板均值（去掉跨扰动共享成分，纯预测侧操作，无泄漏）
     实测：pds 0.8214，from_baseline +0.6429（vs 全基因 +0.50、top3000 +0.536）
深度维持 1e6：E27 的 variant A 同样是 1e6 且 expr_mse_capped_norm 拿到 +0.522，
故 1e6 已被证明不会毁掉该指标；且 pds 在 1e6 比 20k 高 0.214 from_baseline。
"""
from __future__ import annotations
import importlib.util, sys, time
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
E27 = ROOT / "experiments" / "E27-six-metrics" / "out"
OUT = Path(__file__).resolve().parent / "out"
TOPK, LAM, DEPTH, SEED = 1000, 1.0, 1_000_000, 0

spec = importlib.util.spec_from_file_location("e28", Path(__file__).parent / "run.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    perts, genes, real, gi_full, B, S = m._load_beta()
    m_full = m._ctrl_profile(real)
    tgr = np.asarray(real.obs["target_gene"].values, dtype=str)
    n_ntc = int((tgr == "non-targeting").sum())
    print(f"扰动 {len(perts)}  基因 {genes.size:,}  共同 {gi_full.size:,}  NTC {n_ntc:,}")

    Bc = np.where(np.isfinite(B), B, 0.0)
    V = Bc - Bc.mean(1, keepdims=True)          # 去面板均值
    rng = np.random.default_rng(SEED)
    X, obs = [], []
    for j, p in enumerate(perts):
        v = V[:, j].copy()
        if TOPK < v.size:
            v[np.argsort(np.abs(v))[::-1][TOPK:]] = 0.0
        lf = np.zeros(genes.size); lf[gi_full] = LAM * v
        tgt = m_full * 2.0 ** lf; tgt *= 1e6 / tgt.sum()
        X.append(sp.csr_matrix(rng.multinomial(DEPTH, tgt / 1e6, size=400).astype(np.float32)))
        obs += [p] * 400
        print(f"  {p:10s} 非零 lfc {int((lf!=0).sum()):5d}  |lfc| 中位 "
              f"{np.median(np.abs(lf[lf!=0])):.3f}")
    tgt = m_full * (1e6 / m_full.sum())
    X.append(sp.csr_matrix(rng.multinomial(DEPTH, tgt / 1e6, size=n_ntc).astype(np.float32)))
    obs += ["non-targeting"] * n_ntc
    Xs = sp.vstack(X, format="csr")
    a = ad.AnnData(X=Xs, obs=pd.DataFrame({"target_gene": obs},
                   index=[str(i) for i in range(len(obs))]),
                   var=pd.DataFrame(index=pd.Index(genes)))
    a.write_h5ad(OUT / "pred_v2.h5ad")
    print(f"pred_v2.h5ad {Xs.shape}  {(OUT/'pred_v2.h5ad').stat().st_size/1e6:.0f} MB"
          f"  {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
