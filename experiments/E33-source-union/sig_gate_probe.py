"""Part 1c（修正 F28 的分母）：在 `de_lfc_nmae` 自己的 gate 上量覆盖率。

cell_eval2 的 `de_lfc_nmae`（metrics/de.py:587-723，逐字读过）不是在全部 18,080 个
基因、也不是在 CPM gate 上算的：

    real_sig = real_df.filter(p_adj < 0.05).filter(log2_fold_change.is_finite())
               然后 _exclude_own_gene(...)
    nmae     = mean|lfc_pred - lfc_real| / mean|lfc_real|   over real_sig

即**逐扰动的、real 侧显著基因集**。所以 F28 的「7,481 / 18,080 = 41% → 下限 0.59」
用错了分母两次：18,080 是 var 长度（本 harness 的 CPM gate 只有 10,779），而真正的
分母是每个扰动自己的显著集。

并且下限是**幅值加权**的，不是计数加权：
    floor = sum_{未覆盖} |lfc_real| / sum_{real_sig} |lfc_real|
（未覆盖处只能预测 0，分子项恰等于分母项。）

real 侧 DE 用 vcclab.scorer 的精确复刻（对官方逐位验证过，见该模块文档），
对照组取 real.h5ad 自己的 non-targeting 细胞 —— 与 control_source="real" 一致。
为省磁盘（全机只剩 ~11 GB，五个 agent 并发）不落盘临时 h5ad，直接在内存里
复刻 ControlRef.__init__。

跑法：~/vcc2026/.venv/bin/python experiments/E33-source-union/sig_gate_probe.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vcclab.scorer import EPS, GATE_CPM, TS_BULK, TS_CELL, ControlRef  # noqa: E402

DATA = ROOT / "data"
REAL = ROOT / "experiments" / "E27-six-metrics" / "out" / "real.h5ad"   # 只读，绝不删改
MAPCSV = DATA / "external" / "ens2sym.csv"
OUT = Path(__file__).resolve().parent / "out"
ALPHA, LAMBDA = 0.05, 0.7
SOURCES = ("K562GW", "K562Essential", "RPE1Essential",
           "JurkatEssential", "HepG2Essential")
FILL_ORDER = ("HepG2Essential", "JurkatEssential", "RPE1Essential", "K562Essential")


def as_str(g):
    if isinstance(g, h5py.Group) and "categories" in g:
        c = [x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]
        return np.asarray(c)[g["codes"][:]]
    if isinstance(g, h5py.Group) and "values" in g:      # nullable-string-array 编码
        g = g["values"]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g[:]])


def ref_from_matrix(X: sp.csr_matrix, gene_names) -> ControlRef:
    """ControlRef.__init__ 的内存版逐字复刻（scorer.py:84-105），跳过 h5ad 落盘。"""
    r = object.__new__(ControlRef)
    r.n_ctrl, r.n_genes = X.shape
    lib = np.asarray(X.sum(1)).ravel()
    cpm = X.multiply((TS_CELL / lib)[:, None]).tocsc()
    r.m_full = np.asarray(cpm.mean(0)).ravel()
    pb = np.asarray(X.sum(0)).ravel()
    r.b_ctrl = np.log1p(TS_BULK * pb / pb.sum())
    r.gidx = np.flatnonzero(r.m_full > GATE_CPM)
    r.m_gate = r.m_full[r.gidx]
    r.G = len(r.gidx)
    sub = cpm[:, r.gidx].tocsc()
    r._sorted = [np.sort(sub.data[sub.indptr[j]:sub.indptr[j + 1]]).astype(np.float32)
                 for j in range(r.G)]
    r._nzero = np.array([r.n_ctrl - a.size for a in r._sorted], dtype=np.int64)
    r._cpm_csr = cpm.tocsr()
    return r


def real_de():
    """{pert: (p_adj, lfc)} 在 CPM gate 内 + gate_sym。结果缓存。"""
    cache = OUT / "real_de.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        perts = [str(p) for p in z["perts"]]
        return z["gate_sym"], {p: (z[f"p_{p}"], z[f"l_{p}"]) for p in perts}
    with h5py.File(REAL, "r") as f:
        genes = as_str(f["var"]["_index"])
        tg = as_str(f["obs"]["target_gene"])
        X = sp.csr_matrix((f["X/data"][:], f["X/indices"][:], f["X/indptr"][:]),
                          shape=tuple(f["X"].attrs["shape"]))
    ntc = tg == "non-targeting"
    print(f"real.h5ad {X.shape}  对照 {int(ntc.sum()):,}  扰动 "
          f"{len(set(tg.tolist())) - 1}")
    ref = ref_from_matrix(X[ntc].tocsr(), genes)
    print(f"real 侧 CPM gate G={ref.G:,}")
    out = {}
    for p in sorted(set(tg.tolist()) - {"non-targeting"}):
        t = time.time()
        Xi = X[tg == p].tocsr()
        lib = np.asarray(Xi.sum(1)).ravel()
        cnt = np.asarray(Xi.multiply((TS_CELL / lib)[:, None]).todense())
        pa, lf = ref.de_table(cnt, tie_correct=True)
        out[p] = (pa, lf)
        print(f"  {p:10s} n={Xi.shape[0]:4d}  显著 {(pa < ALPHA).sum():5,d}"
              f"  |lfc| 中位 {np.median(np.abs(lf)):.4f}  ({time.time()-t:.0f}s)")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, gate_sym=genes[ref.gidx], perts=np.array(list(out)),
                        **{f"p_{k}": v[0] for k, v in out.items()},
                        **{f"l_{k}": v[1] for k, v in out.items()})
    return genes[ref.gidx], out


def load_source(name: str, picks: list[str], e2s: dict):
    cols = [str(c) for c in pd.read_csv(DATA / "nadig2025" / f"{name}_lfc.csv.gz",
                                        index_col=0, nrows=1).columns]
    have = [p for p in picks if p in cols]
    if not have:
        return pd.DataFrame()
    d = pd.read_csv(DATA / "nadig2025" / f"{name}_lfc.csv.gz", index_col=0,
                    usecols=["Unnamed: 0"] + have,
                    dtype={c: np.float32 for c in have}, engine="c")
    d.index = d.index.astype(str)
    d = d[have]
    sym = pd.Index(d.index.map(e2s))
    keep = pd.notna(sym)
    d, sym = d[keep], sym[keep]
    dd = ~sym.duplicated()
    d, sym = d[dd], sym[dd]
    d.index = sym
    return d


def main() -> None:
    t0 = time.time()
    picks = [l.strip() for l in
             (ROOT / "experiments" / "E28-pds" / "out" / "perts.csv").read_text().split()
             if l.strip()]
    gate_sym, de = real_de()
    gpos = pd.Index(gate_sym)
    e2s_df = pd.read_csv(MAPCSV).dropna()
    e2s = dict(zip(e2s_df.ensembl, e2s_df.symbol))
    loaded = {s: load_source(s, picks, e2s) for s in SOURCES}
    loaded = {s: d for s, d in loaded.items() if not d.empty}
    gw = loaded["K562GW"]
    gw_sym = set(gw.index) & set(gate_sym.tolist())

    # ---- real_sig：metrics/de.py 的 gate，逐扰动 ----
    print(f"\n=== de_lfc_nmae 自己的 gate（p_adj<{ALPHA}，有限 lfc，剔除自身基因）===")
    print(f"{'扰动':10s} {'|real_sig|':>10s} {'GW覆盖':>8s} {'计数%':>7s} {'幅值%':>7s} "
          f"{'并集':>7s} {'并集幅值%':>9s} {'floor_GW':>9s} {'floor_∪':>8s}")
    print("-" * 88)
    sig = {}
    rows = []
    for p in picks:
        pa, lf = de[p]
        m = (pa < ALPHA) & np.isfinite(lf) & (gate_sym != p)
        idx = np.flatnonzero(m)
        s_sym = gate_sym[idx]
        s_lfc = lf[idx]
        w = np.abs(s_lfc)
        in_gw = np.isin(s_sym, list(gw_sym))
        uni_syms = set(gw_sym)
        for s_name in FILL_ORDER:
            d = loaded.get(s_name)
            if d is not None and p in d.columns:
                uni_syms |= set(d.index) & set(gate_sym.tolist())
        in_uni = np.isin(s_sym, list(uni_syms))
        cw = float(w[in_gw].sum() / max(w.sum(), 1e-12))
        uw = float(w[in_uni].sum() / max(w.sum(), 1e-12))
        sig[p] = (s_sym, s_lfc, idx)
        rows.append((p, len(idx), int(in_gw.sum()), in_gw.mean(), cw,
                     int(in_uni.sum()), uw, 1 - cw, 1 - uw))
        print(f"{p:10s} {len(idx):10,d} {int(in_gw.sum()):8,d} {in_gw.mean():7.1%} "
              f"{cw:7.1%} {int(in_uni.sum()):7,d} {uw:9.1%} {1-cw:9.4f} {1-uw:8.4f}")
    print("-" * 88)
    mc = float(np.mean([r[4] for r in rows])); mu = float(np.mean([r[6] for r in rows]))
    print(f"{'均值':10s} {np.mean([r[1] for r in rows]):10,.0f} "
          f"{np.mean([r[2] for r in rows]):8,.0f} {np.mean([r[3] for r in rows]):7.1%} "
          f"{mc:7.1%} {np.mean([r[5] for r in rows]):7,.0f} {mu:9.1%} "
          f"{1-mc:9.4f} {1-mu:8.4f}")
    print(f"→ K562GW 幅值覆盖 {mc:.2%} ⇒ nmae 完美预测下限 {1-mc:.4f}"
          f"；并集 {mu:.2%} ⇒ {1-mu:.4f}；差 {(1-mc)-(1-mu):+.4f}")

    # ---- B/A：真实 nmae 行为，不是天花板 ----
    print(f"\n=== 实测 nmae（λ={LAMBDA}，在 real_sig 上，非天花板）===")
    print(f"{'扰动':10s} {'预测0':>8s} {'GW only':>9s} {'GW+填充':>9s} {'填充Δ':>8s} "
          f"{'GW符号':>8s} {'填充符号':>9s}")
    print("-" * 70)

    def build_pred(p, use_fill: bool):
        out = np.zeros(len(gate_sym))
        pos = gpos.get_indexer(pd.Index(sorted(gw_sym)))
        si = gw.index.get_indexer(pd.Index(sorted(gw_sym)))
        v = gw[p].to_numpy(dtype=np.float64)[si]
        out[pos] = LAMBDA * np.where(np.isfinite(v), v, 0.0)
        filled = np.zeros(len(gate_sym), bool)
        if use_fill:
            for s_name in FILL_ORDER:            # 优先级：自身 gate 覆盖率降序，固定不调
                d = loaded.get(s_name)
                if d is None or p not in d.columns:
                    continue
                cand = pd.Index(sorted((set(d.index) & set(gate_sym.tolist())) - gw_sym))
                if len(cand) == 0:
                    continue
                pos2 = gpos.get_indexer(cand)
                si2 = d.index.get_indexer(cand)
                v2 = d[p].to_numpy(dtype=np.float64)[si2]
                free = (out[pos2] == 0.0) & (~filled[pos2]) & np.isfinite(v2)
                out[pos2[free]] = LAMBDA * v2[free]
                filled[pos2[free]] = True
        return out, filled

    n0 = nA = nB = 0.0
    for p in picks:
        s_sym, s_lfc, idx = sig[p]
        den = float(np.abs(s_lfc).mean())
        pa_, fa = build_pred(p, False)
        pb_, fb = build_pred(p, True)
        a = float(np.abs(pa_[idx] - s_lfc).mean()) / den
        b = float(np.abs(pb_[idx] - s_lfc).mean()) / den
        z = float(np.abs(0 - s_lfc).mean()) / den
        gwm = np.isin(s_sym, list(gw_sym))
        sg = pa_[idx][gwm]
        ag_gw = float(np.mean(np.sign(sg) == np.sign(s_lfc[gwm]))) if gwm.any() else np.nan
        fm = fb[idx]
        ag_f = float(np.mean(np.sign(pb_[idx][fm]) == np.sign(s_lfc[fm]))) if fm.any() else np.nan
        n0 += z; nA += a; nB += b
        print(f"{p:10s} {z:8.4f} {a:9.4f} {b:9.4f} {b-a:+8.4f} {ag_gw:8.1%} "
              f"{ag_f:9.1%}" if fm.any() else
              f"{p:10s} {z:8.4f} {a:9.4f} {b:9.4f} {b-a:+8.4f} {ag_gw:8.1%} "
              f"{'—':>9s}")
    k = len(picks)
    print("-" * 70)
    print(f"{'均值':10s} {n0/k:8.4f} {nA/k:9.4f} {nB/k:9.4f} {(nB-nA)/k:+8.4f}")
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
