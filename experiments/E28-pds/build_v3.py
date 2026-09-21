"""变体 V3：单旋钮实验 —— 只把 hamilton+两点构造换成多项式抽样，设计完全照 E27 variant A。

V2 一次改了三件事（量化器、top1000、去面板均值），结果无法归因且净损 -0.019：
  pds +0.5357 但 direction_reach -0.1278、expr_mse -0.5219。
读 de_direction_reach 的定义后知道原因：它测的是我们自己 |lfc| 排序的方向纯净前缀深度
 k*/N_conf，去面板均值让排序不再跟源侧置信度对齐，纯前缀立刻崩掉。
而 expr_mse 要绝对精度，用 8 个扰动估共享成分等于加噪声（原始 3.42 比退化基线 2.60 还差）。

本地 harness：不去均值、只换抽样，pds 仍有 0.7143（from_baseline +0.4286）
= V2 全部 pds 收益的 80%，且零去均值损害。故本变体只动量化器。

旧标题：多项式抽样 + top1000 去面板均值 + 1e6 深度。

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
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

ROOT = Path(__file__).resolve().parents[2]
E27 = ROOT / "experiments" / "E27-six-metrics" / "out"
OUT = Path(__file__).resolve().parent / "out"
TOPK, LAM, DEPTH, SEED = None, 1.0, 20_000, 0   # 设计完全照 A

spec = importlib.util.spec_from_file_location("e28", Path(__file__).parent / "run.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

# E27 variant A 的 build 实测打印出的 K（n_src 在 gate∩common 的 7016 个基因上做 BH）。
# 在此硬编码而非重算：我第一次重算时用了全部 K562GW 行，得到不同的 n_src
# （MAT2A 154 vs 82 → K 7481 vs 288），那会让本变体同时改动量化器和设计，无法归因。
K_A = {"TCF7L2": 29, "GNG12": 288, "VCL": 29, "COX4I1": 288,
       "MAT2A": 288, "PAXIP1": 288, "SLIRP": 288, "ZNF581": 29}


def _kmap(perts, n_common):
    missing = set(perts) - set(K_A)
    if missing:
        raise ValueError(f"K_A 缺少扰动 {sorted(missing)}；请从 E27 的 build 输出补齐")
    for p in perts:
        print(f"  {p:10s} K={K_A[p]:5d}  (照 E27 variant A)")
    return {p: K_A[p] for p in perts}


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    perts, genes, real, gi_full, B, S = m._load_beta()
    m_full = m._ctrl_profile(real)
    tgr = np.asarray(real.obs["target_gene"].values, dtype=str)
    n_ntc = int((tgr == "non-targeting").sum())
    print(f"扰动 {len(perts)}  基因 {genes.size:,}  共同 {gi_full.size:,}  NTC {n_ntc:,}")

    Bc = np.where(np.isfinite(B), B, 0.0)
    V = Bc                                       # 不去面板均值：照 A
    KMAP = _kmap(perts, gi_full.size)
    rng = np.random.default_rng(SEED)
    X, obs = [], []
    for j, p in enumerate(perts):
        # E27 variant A 的规则：n_src 分区定 K，排序统计量 |beta|/SE
        b, s_ = B[:, j], S[:, j]
        good = np.isfinite(b) & np.isfinite(s_) & (s_ > 0)
        rank = np.where(good, np.abs(b) / np.maximum(s_, 1e-9), -np.inf)
        K = KMAP[p]
        sel = np.argsort(rank)[::-1][:min(K, int(good.sum()))]
        lf = np.zeros(genes.size); lf[gi_full[sel]] = LAM * Bc[sel, j]
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
    a.write_h5ad(OUT / "pred_v3.h5ad")
    print(f"pred_v3.h5ad {Xs.shape}  {(OUT/'pred_v3.h5ad').stat().st_size/1e6:.0f} MB"
          f"  {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
