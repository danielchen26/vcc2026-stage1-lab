"""Part 1b（决定性检验，仍不跑 scorer）：填充基因的 lfc 到底能不能打败「预测 0」。

Part 1 给的是覆盖率天花板（= 完美预测时的 nmae 下限）。但 nmae 的真实行为比
覆盖率更残酷：符号错的时候 |pred - ref| = |pred| + |ref| > |ref|，比预测 0 **更差**。
所以「覆盖了」不等于「有收益」。

本脚本用真正的参考（E27 的 real.h5ad，只读）直接量：
  对每个扰动、每个基因集合，比较
      A = sum |0        - real_lfc|     （V8 现状：lfc_all 未传，非召集基因预测 0）
      B = sum |LAMBDA*s - real_lfc|     （填充后：用源 lfc 收缩 0.7）
  B/A < 1 → 该源在该基因集上有收益；>= 1 → 填充反而加大误差。

real_lfc 的定义逐字照 ControlRef.de_table 第 159 行：
      lfc = log2((mean_cpm(pert cells) + EPS) / (mean_cpm(ctrl cells) + EPS))
在 gate 内。官方 DE 是文件内 pert vs non-targeting，故对照均值取 real.h5ad 自己的
non-targeting 细胞。

跑法：~/vcc2026/.venv/bin/python experiments/E33-source-union/fill_value_probe.py
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
from vcclab.scorer import EPS, GATE_CPM, TS_CELL  # noqa: E402

DATA = ROOT / "data"
REAL = ROOT / "experiments" / "E27-six-metrics" / "out" / "real.h5ad"   # 只读
MAPCSV = DATA / "external" / "ens2sym.csv"
OUT = Path(__file__).resolve().parent / "out"
LAMBDA = 0.7
SOURCES = ("K562GW", "K562Essential", "RPE1Essential",
           "JurkatEssential", "HepG2Essential")


def as_str(g):
    if isinstance(g, h5py.Group) and "categories" in g:
        c = [x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]
        return np.asarray(c)[g["codes"][:]]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g[:]])


def real_lfc() -> tuple[np.ndarray, dict, np.ndarray]:
    """返回 (gate_sym, {pert: real_lfc 在 gate 内}, gate 内对照均值 CPM)。"""
    cache = OUT / "real_lfc.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return (z["gate_sym"], {str(k): z[f"lfc_{k}"] for k in z["perts"]}, z["m_gate"])
    with h5py.File(REAL, "r") as f:
        genes = as_str(f["var"]["_index"])
        tg = as_str(f["obs"]["target_gene"])
        X = sp.csr_matrix((f["X/data"][:], f["X/indices"][:], f["X/indptr"][:]),
                          shape=tuple(f["X"].attrs["shape"]))
    lib = np.asarray(X.sum(1)).ravel()
    cpm = X.multiply((TS_CELL / lib)[:, None]).tocsr()
    ntc = tg == "non-targeting"
    m_full = np.asarray(cpm[ntc].mean(0)).ravel()
    gidx = np.flatnonzero(m_full > GATE_CPM)
    gate_sym, m_gate = genes[gidx], m_full[gidx]
    print(f"real.h5ad  {X.shape}  对照 {int(ntc.sum()):,}  gate {len(gidx):,}")
    out = {}
    for p in sorted(set(tg.tolist()) - {"non-targeting"}):
        mp = np.asarray(cpm[tg == p].mean(0)).ravel()[gidx]
        out[p] = np.log2((mp + EPS) / (m_gate + EPS))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, gate_sym=gate_sym, m_gate=m_gate,
                        perts=np.array(list(out)), **{f"lfc_{k}": v for k, v in out.items()})
    return gate_sym, out, m_gate


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
    gate_sym, rl, m_gate = real_lfc()
    gpos = pd.Index(gate_sym)
    e2s_df = pd.read_csv(MAPCSV).dropna()
    e2s = dict(zip(e2s_df.ensembl, e2s_df.symbol))
    loaded = {s: load_source(s, picks, e2s) for s in SOURCES}
    loaded = {s: d for s, d in loaded.items() if not d.empty}

    gw = loaded["K562GW"]
    gw_cov = gpos.intersection(gw.index)

    print(f"\n真实 lfc 规模：gate {len(gate_sym):,}  "
          f"|real_lfc| 中位 {np.median([np.median(np.abs(rl[p])) for p in picks]):.4f}")

    # ---- 检验 1：K562GW 自身在它覆盖的基因上，lfc 能否打败预测 0 ----
    print(f"\n检验 1  K562GW 覆盖基因上 B/A（<1 才有收益，LAMBDA={LAMBDA}）")
    print(f"{'扰动':10s} {'n':>7s} {'A=sum|0-r|':>11s} {'B=sum|λs-r|':>12s} {'B/A':>7s}"
          f" {'符号一致':>8s}")
    print("-" * 62)
    tot_a = tot_b = 0.0
    for p in picks:
        gi = gpos.get_indexer(gw_cov)
        si = gw.index.get_indexer(gw_cov)
        r = rl[p][gi]
        s = gw[p].to_numpy(dtype=np.float64)[si]
        ok = np.isfinite(r) & np.isfinite(s)
        r, s = r[ok], s[ok]
        A = float(np.abs(r).sum())
        B = float(np.abs(LAMBDA * s - r).sum())
        nz = (s != 0) & (r != 0)
        ag = float(np.mean(np.sign(s[nz]) == np.sign(r[nz]))) if nz.any() else float("nan")
        tot_a += A; tot_b += B
        print(f"{p:10s} {ok.sum():7,d} {A:11.1f} {B:12.1f} {B/A:7.3f} {ag:8.1%}")
    print("-" * 62)
    print(f"{'合计':10s} {'':7s} {tot_a:11.1f} {tot_b:12.1f} {tot_b/tot_a:7.3f}")

    # ---- 检验 2：MAT2A 的填充基因上，每个 Essential 源能否打败预测 0 ----
    print(f"\n检验 2  填充基因（K562GW 不覆盖）上 B/A")
    print(f"{'扰动':10s} {'源':16s} {'n':>7s} {'A':>10s} {'B':>10s} {'B/A':>7s} {'符号一致':>8s}")
    print("-" * 74)
    for p in picks:
        for s_name, d in loaded.items():
            if s_name == "K562GW" or p not in d.columns:
                continue
            fill = pd.Index(sorted(set(gate_sym) - set(gw_cov))).intersection(d.index)
            if len(fill) == 0:
                continue
            gi = gpos.get_indexer(fill)
            si = d.index.get_indexer(fill)
            r = rl[p][gi]
            v = d[p].to_numpy(dtype=np.float64)[si]
            ok = np.isfinite(r) & np.isfinite(v)
            r, v = r[ok], v[ok]
            A = float(np.abs(r).sum())
            B = float(np.abs(LAMBDA * v - r).sum())
            nz = (v != 0) & (r != 0)
            ag = float(np.mean(np.sign(v[nz]) == np.sign(r[nz]))) if nz.any() else float("nan")
            print(f"{p:10s} {s_name:16s} {ok.sum():7,d} {A:10.1f} {B:10.1f} "
                  f"{B/A:7.3f} {ag:8.1%}")

    # ---- 检验 3：把检验 2 的收益折算到整个 gate 上的 nmae 变化（8 扰动均值）----
    print(f"\n检验 3  折算到 8 扰动均值的 nmae（分母 = 每扰动 sum|real_lfc| over gate）")
    print(f"{'方案':44s} {'nmae':>8s}")
    print("-" * 54)

    def nmae(pred_fn) -> float:
        acc = []
        for p in picks:
            r = rl[p]
            pr = pred_fn(p)
            acc.append(np.abs(pr - r).sum() / max(np.abs(r).sum(), 1e-12))
        return float(np.mean(acc))

    zeros = np.zeros(len(gate_sym))

    def pred_zero(p):
        return zeros

    def make_pred(use_fill: bool):
        def f(p):
            out = np.zeros(len(gate_sym))
            gi = gpos.get_indexer(gw_cov)
            si = gw.index.get_indexer(gw_cov)
            v = gw[p].to_numpy(dtype=np.float64)[si]
            out[gi] = LAMBDA * np.where(np.isfinite(v), v, 0.0)
            if use_fill:
                # 优先级：按该源自身 gate 覆盖率降序（HepG2 > RPE1 > Jurkat > K562Ess）
                for s_name in ("HepG2Essential", "JurkatEssential",
                               "RPE1Essential", "K562Essential"):
                    d = loaded.get(s_name)
                    if d is None or p not in d.columns:
                        continue
                    cand = pd.Index(sorted(set(gate_sym) - set(gw_cov))).intersection(d.index)
                    gi2 = gpos.get_indexer(cand)
                    si2 = d.index.get_indexer(cand)
                    v2 = d[p].to_numpy(dtype=np.float64)[si2]
                    m = (out[gi2] == 0.0) & np.isfinite(v2)
                    out[gi2[m]] = LAMBDA * v2[m]
            return out
        return f

    print(f"{'V8 现状（lfc_all 未传 → 全 gate 预测 0*）':44s} {nmae(pred_zero):8.4f}")
    print(f"{'V11 假想：lfc_all = K562GW only':44s} {nmae(make_pred(False)):8.4f}")
    print(f"{'V12 假想：lfc_all = K562GW + Essential 填充':44s} {nmae(make_pred(True)):8.4f}")
    print("* V8 实际在 288 个召集基因上非 0，故真实 V8 略优于此行；此行是 lfc_all 通道的基线。")
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
