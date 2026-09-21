"""变体 V7：真正的单旋钮 —— K 规则 29/288/G → 扁平 288，其余逐字照 V6。

由 E27 run.py 逐字复制，仅三处：输出路径、排序统计量、只写 pred_v7。
gate、K 规则（29/288/G）、bootstrap、两点构造、hamilton、1e6 深度、种子全不变。

依据（design_probe.py，管线照 A，已验证的 pds harness）：
  |beta|/SE  29/288/G   pds 0.5179  from_baseline +0.0357  ← A
  |beta|/SE  288 全部    pds 0.6250  from_baseline +0.2500
  |beta|     29/288/G   pds 0.7143  from_baseline +0.4286  ← 本变体
  |beta|     288 全部    pds 0.7500  from_baseline +0.5000
机制：余弦由大坐标主导，|beta| 把信号放在余弦看得见的地方。

风险：|beta|/SE 是 E18 为 sig_jaccard 选出的最优排序，且 de_direction_reach 读的是
我们自己 |lfc| 排序的符号纯净前缀深度 —— 两者都可能退化。故必须六指标一起测。
E27 — 用官方打分器测全部 6 个计分指标，得出边际收益图。

## 为什么这是当前唯一该做的实验

源码给出的事实：
    aggregate_metrics: "one NaN-skipping value per metric over perturbations -- the mean"
    score_metrics(...) -> {metric, from_baseline} + **avg_score**

    总分 = (1/6) * sum_i (ours_i - baseline_i) / (anchor_i - baseline_i)

vcc2026 里 scored=True 的只有 6 个：
    de_wilcoxon_sig_jaccard                    higher, anchor=1.0   ← 我全部精力在这
    de_wilcoxon_lfc_nmae                       lower,  anchor=0.0   ← 未测
    de_wilcoxon_direction_fidelity_yield_raw   higher, anchor=1.0   ← 未测
    de_wilcoxon_direction_reach_raw            higher, anchor=1.0   ← 未测
    pds_cosine                                 higher, anchor=1.0   ← 未测
    expr_mse_unbiased_capped_norm              lower,  anchor=0.0   ← 未测

**sig_jaccard 只占 1/6。** E14-E26 关于「打不过榜首」的结论只覆盖这 1/6。
剩下 5/6 从未测量 —— 本实验就是去测。

## 为什么能在本机测

打分需要真答案，而 A/B/C 没有。但 **H1 2025 有细胞级真数据**，
且与本届同协议。所以：
    real = H1 的真实扰动细胞（降采样到本届条件）+ NTC
    pred = 我们用 K562 迁移 + 解码器构造的细胞          + 同一批 NTC
    base = 「原样输出对照」的退化提交                    + 同一批 NTC
score_metrics(pred, base) 给出 6 个 from_baseline。

## 设计选择

- 排序：|beta_K562| / lfcSE_K562，MDE 有限的基因才入选（E18 选出的最好排序）
- K：先验分区（不泄漏；E24 的 P3）—— n_src < 5 → 29；< 100 → 288；否则全报
- 迁移的 lfc：beta_K562 乘收缩系数 lam。**lam 是 lfc_nmae 的旋钮**，本轮固定 1.0，
  测出基线位置后再扫。

## 分阶段 + 检查点

官方 compute_metrics 约 100 s/扰动，shell 长任务会被打断，所以：
    stage=build  构造三份 h5ad 存盘
    stage=score  跑 compute_metrics + score_metrics，逐份存中间结果
可分别调用，已完成的阶段自动跳过。

跑法：
    ~/vcc2026/.venv/bin/python experiments/E27-six-metrics/run.py build
    ~/vcc2026/.venv/bin/python experiments/E27-six-metrics/run.py score
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vcclab.decoder import design_cells  # noqa: E402
from vcclab.detectability import mde  # noqa: E402
from vcclab.scorer import ControlRef, bh_adjust  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "E10-h1-erp"))
from run import VCC_CTRL_CELLS, VCC_PERT_CELLS, VCC_UMI, read_rows, thin, to_cpm  # noqa: E402

DATA = ROOT / "data"
H5 = DATA / "vcc2025" / "adata_Validation.h5ad"
MAPCSV = DATA / "external" / "ens2sym.csv"
OUT = Path(__file__).resolve().parents[1] / "E28-pds" / "out"
N_PERT = 8                 # 官方 compute_metrics ~100 s/扰动
ALPHA, Z_BH, SEED = 0.05, 3.184, 0
LAMBDA = 0.5               # V6 唯一改动：lfc 收缩系数 1.0 -> 0.5
# 依据（lambda_probe.py，管线照 V5，已验证的 pds harness + bulk_lognorm 上的 MSE 代理）：
#   lam 1.00  pds 0.7143 (+0.4286)  MSE 代理 0.003565  1.000
#   lam 0.70  pds 0.7143 (+0.4286)  MSE 代理 0.002810  0.788
#   lam 0.50  pds 0.7143 (+0.4286)  MSE 代理 0.002468  0.692  <- 甜点
#   lam 0.30  pds 0.6964 (+0.3929)  MSE 代理 0.002259  0.634
# 余弦对尺度不变（log1p 让它只是近似不变），故收缩到 0.5 时 pds 分毫不动而误差降 31%。
# 目标：救回 V5 丢掉的 expr_mse（0.5219 -> 0.3930），并让 lfc_nmae 从收缩中获益。
SCORED = ("de_wilcoxon_sig_jaccard", "de_wilcoxon_lfc_nmae",
          "de_wilcoxon_direction_fidelity_yield_raw",
          "de_wilcoxon_direction_reach_raw", "pds_cosine",
          "expr_mse_unbiased_capped_norm")


def as_str(g):
    if isinstance(g, h5py.Group) and "categories" in g:
        c = [x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]
        return np.asarray(c)[g["codes"][:]]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g[:]])


def build() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== stage=build ===")
    with h5py.File(H5, "r") as f:
        genes = as_str(f["var"]["_index"])
        tg_all = as_str(f["obs"]["target_gene"])
    cats = sorted(set(tg_all) - {"non-targeting"})
    rng = np.random.default_rng(SEED)

    ntc_rows = rng.choice(np.flatnonzero(tg_all == "non-targeting"),
                          VCC_CTRL_CELLS, replace=False)
    ctrl = thin(read_rows(ntc_rows), VCC_UMI, rng)
    ntc_path = OUT / "_ctrl.h5ad"
    ad.AnnData(X=sp.csr_matrix(ctrl),
               var=pd.DataFrame(index=pd.Index(genes))).write_h5ad(ntc_path)
    ref = ControlRef.load(ntc_path, list(genes))
    gidx = np.asarray(ref.gidx)
    m = mde(ref, n_cells=VCC_PERT_CELLS, alpha=float(2 * norm.sf(Z_BH)),
            seed=SEED, tie_correct=True)
    print(f"gate {ref.G:,}  对照 {ref.n_ctrl:,}  MDE 中位 {np.nanmedian(m):.4f}")

    # 源侧
    e2s = pd.read_csv(MAPCSV).dropna()
    gw_cols = [str(c) for c in pd.read_csv(DATA / "nadig2025" / "K562GW_p.csv.gz",
                                          index_col=0, nrows=1).columns]
    usable = sorted(set(gw_cols) & set(cats))
    # 按 H1 真实 |R_p| 分层挑 N_PERT 个，跨全谱
    real_n = {}
    for p in usable:
        idx = np.flatnonzero(tg_all == p)
        real_n[p] = len(idx)
    cand = [p for p in usable if real_n[p] >= VCC_PERT_CELLS]
    cand.sort(key=lambda p: real_n[p])
    pick = [cand[int(i)] for i in np.linspace(0, len(cand) - 1, N_PERT).round()]
    print(f"选中 {len(pick)} 个扰动: {pick}")

    def rd(n):
        d = pd.read_csv(DATA / "nadig2025" / f"{n}.csv.gz", index_col=0,
                        usecols=["Unnamed: 0"] + pick,
                        dtype={c: np.float32 for c in pick}, engine="c")
        d.index = d.index.astype(str)
        return d[pick]

    lfc_s, se_s, p_s = rd("K562GW_lfc"), rd("K562GW_se"), rd("K562GW_p")
    se_s, p_s = se_s.reindex(index=lfc_s.index), p_s.reindex(index=lfc_s.index)
    sym = pd.Index(lfc_s.index.map(dict(zip(e2s.ensembl, e2s.symbol))))
    keep = pd.notna(sym)
    lfc_s, se_s, p_s, sym = lfc_s[keep], se_s[keep], p_s[keep], sym[keep]
    dd = ~sym.duplicated()
    lfc_s, se_s, p_s, sym = lfc_s[dd], se_s[dd], p_s[dd], sym[dd]
    gate_sym = genes[gidx]
    common = pd.Index(gate_sym).intersection(sym)
    gi_g = pd.Index(gate_sym).get_indexer(common)      # gate 内下标
    gi_s = sym.get_indexer(common)
    B = lfc_s.to_numpy().astype(np.float64)[gi_s]
    S = se_s.to_numpy().astype(np.float64)[gi_s]
    P = p_s.to_numpy().astype(np.float64)[gi_s]
    print(f"共同基因 {len(common):,}")

    X_pred, X_real, X_base, obs_p, obs_r = [], [], [], [], []
    for j, p in enumerate(pick):
        b, s = B[:, j], S[:, j]
        good = np.isfinite(b) & np.isfinite(s) & (s > 0)
        score = np.where(good, np.abs(b), -np.inf)   # V5 唯一改动：|beta| 取代 |beta|/SE
        pc = P[:, j]
        fin = np.isfinite(pc)
        n_src = int((bh_adjust(pc[fin]) < ALPHA).sum()) if fin.any() else 0
        K = 288   # V7 唯一改动：扁平 K=288，取代 E24 的 29/288/G 先验分区
        # 依据（design_probe.py，管线照 A，已验证的 pds harness）：
        #   |beta| + 29/288/G  pds 0.7143  from_baseline +0.4286  <- V5/V6
        #   |beta| + 288 全部   pds 0.7500  from_baseline +0.5000  <- 本变体
        # 只影响 n_src<5 的三个扰动（TCF7L2/VCL/ZNF581，原 K=29）：29 个基因的信号
        # 在余弦里几乎不可见，也给不出足够长的 direction_reach 前缀。
        sel = np.argsort(score)[::-1][:min(K, int(good.sum()))]
        r_set = gi_g[sel]
        lfc_t = LAMBDA * b[sel]
        X_pred.append(sp.csr_matrix(design_cells(ref, r_set, lfc_t,
                                                 n_cells=VCC_PERT_CELLS, seed=SEED)))
        # 退化基线：对照均值复制 400 遍
        base_row = np.zeros(len(genes))
        base_row[gidx] = ref.m_gate
        from vcclab.decoder import hamilton
        X_base.append(sp.csr_matrix(np.tile(hamilton(base_row), (VCC_PERT_CELLS, 1))))
        idx = np.flatnonzero(tg_all == p)
        idx = rng.choice(idx, VCC_PERT_CELLS, replace=False)
        X_real.append(sp.csr_matrix(thin(read_rows(idx), VCC_UMI, rng)))
        obs_p += [p] * VCC_PERT_CELLS
        obs_r += [p] * VCC_PERT_CELLS
        print(f"  {p:10s} n_src={n_src:5d} K={K:5d} |R̂|={len(r_set):5d}"
              f"  真实细胞 {real_n[p]:5d}")

    ctrl_csr = sp.csr_matrix(ctrl)
    n_ntc = ctrl_csr.shape[0]
    for tag, blocks in (("pred_v7", X_pred),):
        X = sp.vstack(blocks + [ctrl_csr], format="csr")
        obs = pd.DataFrame({"target_gene": obs_p + ["non-targeting"] * n_ntc})
        obs.index = obs.index.astype(str)
        a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=pd.Index(genes)))
        a.write_h5ad(OUT / f"{tag}.h5ad")
        print(f"{tag}.h5ad  {X.shape}  {(OUT/f'{tag}.h5ad').stat().st_size/1e6:.0f} MB")
    ntc_path.unlink(missing_ok=True)
    pd.Series(pick).to_csv(OUT / "perts.csv", index=False, header=False)
    print(f"\nbuild 完成 {time.time()-t0:.0f}s")


def score() -> None:
    t0 = time.time()
    from cell_eval2 import EvalConfig, compute_metrics, aggregate_metrics
    from cell_eval2.score import score_metrics
    cfg = EvalConfig.from_preset("vcc2026")
    cfg = replace(cfg, pert_col="target_gene", device="cpu")
    cfg = replace(cfg, de=replace(cfg.de, backend="scanpy"))
    print("=== stage=score ===")
    for tag in ("pred", "base"):
        f = OUT / f"agg_{tag}.parquet"
        if f.exists():
            print(f"{tag}: 已有 {f.name}，跳过")
            continue
        t = time.time()
        df = compute_metrics(str(OUT / f"{tag}.h5ad"), str(OUT / "real.h5ad"),
                             config=cfg)
        agg = aggregate_metrics(df)
        agg.write_parquet(f)
        print(f"{tag}: compute_metrics + aggregate 完成 ({time.time()-t:.0f}s)")
    res = score_metrics(str(OUT / "agg_pred.parquet"), str(OUT / "agg_base.parquet"),
                        comparison_statistic="mean")
    print(f"\n{'='*70}\n6 个计分指标的 from_baseline\n{'='*70}")
    print(res)
    try:
        d = res.to_pandas()
        d.to_csv(OUT / "score.csv", index=False)
        print(f"\n已存 {OUT/'score.csv'}")
    except Exception:
        pass
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "build"
    {"build": build, "score": score}[stage]()
