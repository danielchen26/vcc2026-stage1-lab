"""E26 — 攻排序，不攻 K：用 Nadig 四细胞系学「哪些下游基因跨系可靠」。

## 为什么换靶

天花板算清了（同一批 47 扰动，中位数口径）：

    当前预测 K              0.21×  榜首
    oracle |R_p| 当 K       1.12×  ← **即使 |R_p| 预测完美的上限**
    oracle K_opt            1.81×  ← 用到答案本身（jac(K) 的 argmax），不可从源侧得知

前五轮（E19/E21/E22/E23/E25）全在动 K，全撞在 1.12× 这面墙上。
而排序侧的账完全不同：

    跨系实测 h = 0.175 [E14]
    同系均值型统计量上限 h = 0.500 [E17]  →  只恢复了 35%

排序翻倍 → h≈0.35 → 约 0.99× 榜首。**这条能打过。**

## 未被使用的资产

Nadig 四细胞系必需面板与本届 300 靶基因**零重叠**（F18），所以不能当扰动源。
**但它能学「哪些下游基因跨细胞系可靠」** —— 那个权重按**下游基因**索引，
与被扰动的基因无关。2,052 扰动 × 4 细胞系 ≈ 8,000 个观测，
比 E19 用 47 个扰动估的 sigma_cross 强一个量级（那版 p=0.025 显著更差）。

## 形式化

对每个下游基因 g，用四系的 beta 估跨系一致性。三个候选权重：

    w_icc(g)   = 组内相关：Var_between_pert / (Var_between_pert + Var_between_line)
                 高 = 该基因的响应由扰动决定，不由细胞系决定 → 可迁移
    w_corr(g)  = 六个细胞系对上 beta 的平均 Pearson（跨 2,052 个扰动）
    w_sign(g)  = 四系符号一致率（对幅度不敏感，更稳健）

排序改为  score_g = (|beta_K562(g)| / lfcSE_K562(g)) * phi(w(g))
其中 phi 取三种形式：恒等、幂 w^a、硬门限 w > 分位数。

## 关键假设（要测的就是它）

权重在**必需基因扰动**上估出，用到**非必需靶基因**上。
即假设「基因级可迁移性是基因的属性，不随扰动类别变化」。若否，本方案作废。

## 判据

靶 = H1 官方真集（本届确切条件），预言机规模 K=|R_p| 隔离排序质量。
对照 = 无权重的 |beta|/lfcSE（E14/E18 的最好排序，h=0.175）。
配对 Wilcoxon。**h 显著上升才算成立。**

跑法：  ~/vcc2026/.venv/bin/python experiments/E26-gene-reliability/run.py
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

DATA = ROOT / "data"
ND = DATA / "nadig2025"
H5 = DATA / "vcc2025" / "adata_Validation.h5ad"
MAPCSV = DATA / "external" / "ens2sym.csv"
OUT = Path(__file__).parent
LINES = ("K562", "RPE1", "Jurkat", "HepG2")
ALPHA, Z_BH, SEED = 0.05, 3.184, 0


def main() -> None:
    t0 = time.time()
    print("=== E26 用 Nadig 四细胞系学基因级跨系可靠性，改善排序 ===\n")

    # ---------- 1. 四系必需面板 → 逐基因权重 ----------
    cols = {}
    for ln in LINES:
        d = pd.read_csv(ND / f"{ln}Essential_p.csv.gz", index_col=0, nrows=1)
        cols[ln] = [str(c) for c in d.columns]
    shared = sorted(set.intersection(*(set(c) for c in cols.values())))
    print(f"四系共享扰动 {len(shared):,}（与本届靶基因零重叠 —— 权重的估计集与应用集不交）")

    t = time.time()
    Bl, idx0 = {}, None
    for ln in LINES:
        d = pd.read_csv(ND / f"{ln}Essential_lfc.csv.gz", index_col=0,
                        usecols=["Unnamed: 0"] + shared,
                        dtype={c: np.float32 for c in shared}, engine="c")
        d.index = d.index.astype(str)
        d = d[shared]                                  # 必须重排（E07 的坑）
        idx0 = d.index if idx0 is None else idx0.intersection(d.index)
        Bl[ln] = d
    for ln in LINES:
        Bl[ln] = Bl[ln].reindex(index=idx0)
    A = np.stack([Bl[ln].to_numpy().astype(np.float32) for ln in LINES])   # 系 × 基因 × 扰动
    print(f"四系 lfc 张量 {A.shape}  ({time.time()-t:.0f}s)")

    finite = np.isfinite(A).all(0)                       # 基因 × 扰动
    okg = finite.sum(1) >= 200
    print(f"可用下游基因 {okg.sum():,} / {A.shape[1]:,}（要求 ≥200 个扰动四系齐全）")

    Am = np.where(np.isfinite(A), A, np.nan)
    with np.errstate(invalid="ignore"):
        # ICC：扰动间方差 / (扰动间 + 细胞系间)
        mean_over_line = np.nanmean(Am, axis=0)                    # 基因 × 扰动
        v_pert = np.nanvar(mean_over_line, axis=1)                 # 基因
        v_line = np.nanmean(np.nanvar(Am, axis=0), axis=1)         # 基因
        w_icc = v_pert / np.maximum(v_pert + v_line, 1e-12)
        # 符号一致率
        sg = np.sign(Am)
        agree = np.nanmean(np.abs(np.nansum(sg, axis=0)) / 4.0, axis=1)
        w_sign = agree
        # 六对平均 Pearson
        pr = []
        for i in range(4):
            for j in range(i + 1, 4):
                x, y = Am[i], Am[j]
                mx, my = np.nanmean(x, 1, keepdims=True), np.nanmean(y, 1, keepdims=True)
                xc, yc = x - mx, y - my
                num = np.nansum(xc * yc, 1)
                den = np.sqrt(np.nansum(xc ** 2, 1) * np.nansum(yc ** 2, 1))
                pr.append(num / np.maximum(den, 1e-12))
        w_corr = np.nanmean(np.stack(pr), axis=0)
    gene_ens = np.array(idx0)
    W = pd.DataFrame({"ensembl": gene_ens, "w_icc": w_icc, "w_sign": w_sign,
                      "w_corr": w_corr})[okg]
    print(f"权重: ICC 中位 {W.w_icc.median():.3f} · 符号一致 {W.w_sign.median():.3f}"
          f" · 六对相关 {W.w_corr.median():.3f}")

    # ---------- 2. 靶侧 H1 ----------
    with h5py.File(H5, "r") as f:
        h1g = [g.decode() if isinstance(g, bytes) else str(g)
               for g in f["var"]["_index"][:]]
        o = f["obs"]["target_gene"]
        cats = [c.decode() if isinstance(c, bytes) else str(c)
                for c in o["categories"][:]]
        codes = o["codes"][:]
    ntc = cats.index("non-targeting")
    rng = np.random.default_rng(SEED)
    rows = rng.choice(np.flatnonzero(codes == ntc), VCC_CTRL_CELLS, replace=False)
    p_tmp = OUT / "_ntc.h5ad"
    ad.AnnData(X=sp.csr_matrix(thin(read_rows(rows), VCC_UMI, rng)),
               var=pd.DataFrame(index=pd.Index(h1g))).write_h5ad(p_tmp)
    ref = ControlRef.load(p_tmp, h1g)
    gidx = np.asarray(ref.gidx)
    m = mde(ref, n_cells=VCC_PERT_CELLS, alpha=float(2 * norm.sf(Z_BH)),
            seed=SEED, tie_correct=True)
    gate_sym = np.array(h1g)[gidx]

    # ---------- 3. 源侧 K562GW + 基因映射 ----------
    e2s = pd.read_csv(MAPCSV).dropna()
    ens2sym = dict(zip(e2s.ensembl, e2s.symbol))
    gw_cols = [str(c) for c in pd.read_csv(ND / "K562GW_p.csv.gz",
                                          index_col=0, nrows=1).columns]
    perts = sorted(set(gw_cols) & (set(cats) - {"non-targeting"}))

    def rd(n):
        d = pd.read_csv(ND / f"{n}.csv.gz", index_col=0,
                        usecols=["Unnamed: 0"] + perts,
                        dtype={c: np.float32 for c in perts}, engine="c")
        d.index = d.index.astype(str)
        return d[perts]

    lfc = rd("K562GW_lfc")
    se = rd("K562GW_se").reindex(index=lfc.index)
    sym = pd.Index(lfc.index.map(ens2sym))
    keep = pd.notna(sym)
    lfc, se, sym, ens = lfc[keep], se[keep], sym[keep], lfc.index[keep]
    dd = ~sym.duplicated()
    lfc, se, sym, ens = lfc[dd], se[dd], sym[dd], ens[dd]

    wmap = {k: dict(zip(W.ensembl, W[k])) for k in ("w_icc", "w_sign", "w_corr")}
    common = pd.Index(gate_sym).intersection(sym)
    gi_h1, gi_k5 = pd.Index(gate_sym).get_indexer(common), sym.get_indexer(common)
    G = len(common)
    B = lfc.to_numpy().astype(np.float64)[gi_k5]
    S = se.to_numpy().astype(np.float64)[gi_k5]
    ens_c = np.array(ens)[gi_k5]
    thr = m[gi_h1]
    ok = np.isfinite(thr) & (thr > 0)
    wv = {k: np.array([wmap[k].get(e, np.nan) for e in ens_c])
          for k in ("w_icc", "w_sign", "w_corr")}
    have_w = np.isfinite(wv["w_icc"])
    print(f"\n共同基因 G = {G:,}  其中有 Nadig 权重的 {have_w.sum():,}"
          f" ({have_w.mean():.0%})   扰动 {len(perts)}")

    # ---------- 4. 逐扰动比较排序 ----------
    variants = ["base"]
    for k in ("w_icc", "w_sign", "w_corr"):
        variants += [f"{k}_mul", f"{k}_gate50", f"{k}_gate75"]
    print(f"\n{'扰动':>10} {'|R_p|':>6} " + " ".join(f"{v[:9]:>10}" for v in variants[:6]))
    recs = []
    for j, name in enumerate(perts):
        idx = np.flatnonzero(codes == cats.index(name))
        if len(idx) > VCC_PERT_CELLS:
            idx = rng.choice(idx, VCC_PERT_CELLS, replace=False)
        padj, _ = ref.de_table(to_cpm(thin(read_rows(idx), VCC_UMI, rng)),
                               tie_correct=True)
        real = (padj < ALPHA)[gi_h1]
        n_real = int(real.sum())
        if n_real == 0:
            continue
        b, s = B[:, j], S[:, j]
        good = np.isfinite(b) & np.isfinite(s) & (s > 0) & ok
        base = np.where(good, np.abs(b) / np.maximum(s, 1e-9), -np.inf)
        rec = dict(target_gene=name, n_real=n_real)
        for v in variants:
            if v == "base":
                sc_ = base
            else:
                key, mode = v.rsplit("_", 1)
                w = np.nan_to_num(wv[key], nan=np.nanmedian(wv[key]))
                if mode == "mul":
                    sc_ = base * np.clip(w, 0, None)
                else:
                    q = 50 if mode == "gate50" else 75
                    thrw = np.nanpercentile(wv[key][have_w], q)
                    sc_ = np.where(w >= thrw, base, -np.inf)
            top = np.argsort(sc_)[::-1][:n_real]
            rec[f"h_{v}"] = float(real[top].sum()) / n_real
        recs.append(rec)
        print(f"{name:>10} {n_real:6d} " +
              " ".join(f"{rec['h_'+v]:10.3f}" for v in variants[:6]))

    df = pd.DataFrame(recs)
    df.to_csv(OUT / "result.csv", index=False)
    p_tmp.unlink(missing_ok=True)

    print(f"\n{'='*72}\n结论（n = {len(df)}，预言机规模 K = |R_p|）\n{'='*72}")
    print(f"{'排序变体':>18} {'h 中位':>8} {'h 均值':>8} {'vs base':>9} "
          f"{'胜/负':>9} {'p':>9}")
    hb = df.h_base.median()
    for v in variants:
        h = df[f"h_{v}"]
        if v == "base":
            print(f"{'base（E14/E18）':>18} {h.median():8.3f} {h.mean():8.3f} "
                  f"{'—':>9} {'—':>9} {'—':>9}")
            continue
        d = h - df.h_base
        p = wilcoxon(h, df.h_base).pvalue if d.abs().sum() else 1.0
        print(f"{v:>18} {h.median():8.3f} {h.mean():8.3f} {h.median()/hb:8.2f}× "
              f"{(d>0).sum():4d}/{(d<0).sum():<4d} {p:9.5f}")

    best = max([v for v in variants if v != "base"], key=lambda v: df[f"h_{v}"].median())
    hbest = df[f"h_{best}"].median()
    p = wilcoxon(df[f"h_{best}"], df.h_base).pvalue
    print(f"\n最好: {best}  h {hbest:.3f} vs base {hb:.3f} = {hbest/hb:.2f}×  p={p:.5f}")
    print(f"同系均值型上限 h=0.500 → 恢复率 {hb/0.500:.0%} → {hbest/0.500:.0%}")
    good = hbest > hb and p < 0.05
    print(f"\n>>> {'基因级可靠性权重有效' if good else '无效：权重不改善排序'} <<<")
    if not good:
        print("    含义：基因级跨系可迁移性**不是**基因的稳定属性，")
        print("    或必需基因扰动上估的权重不适用于非必需靶基因。")
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
