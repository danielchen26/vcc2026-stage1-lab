"""Part 1：在付钱做 build 之前先量天花板 —— 五个 nadig2025 细胞系 + CD4 源
对「被打分基因集」(gate) 的逐源覆盖率、并集覆盖率、以及与 K562GW 的符号一致率。

不跑 decoder，不跑 scorer。只复刻 build_v8.py 第 122-178 行的源侧块。

gate 的取法与 build_v8 逐字等价：
  ControlRef.__init__ 里 gidx = flatnonzero(m_full > GATE_CPM)，
  m_full = mean_0( X.multiply((TS_CELL/lib)[:,None]) )。
这里直接复刻这三行（跳过 per-gene 排序，那只服务于 Wilcoxon，与 gate 无关），
省掉约 10 分钟，数值完全相同。

跑法：~/vcc2026/.venv/bin/python experiments/E33-source-union/coverage_probe.py
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vcclab.scorer import GATE_CPM, TS_CELL  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "E10-h1-erp"))
from run import VCC_CTRL_CELLS, VCC_UMI, read_rows, thin  # noqa: E402

DATA = ROOT / "data"
H5 = DATA / "vcc2025" / "adata_Validation.h5ad"
MAPCSV = DATA / "external" / "ens2sym.csv"
OUT = Path(__file__).resolve().parent / "out"
SEED = 0
SOURCES = ("K562GW", "K562Essential", "RPE1Essential",
           "JurkatEssential", "HepG2Essential")


def as_str(g):
    if isinstance(g, h5py.Group) and "categories" in g:
        c = [x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]
        return np.asarray(c)[g["codes"][:]]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g[:]])


def gate_symbols() -> np.ndarray:
    """build_v8 的 gate_sym = genes[ref.gidx]，缓存到 npy。"""
    cache = OUT / "gate_sym.npy"
    if cache.exists():
        return np.load(cache, allow_pickle=True)
    with h5py.File(H5, "r") as f:
        genes = as_str(f["var"]["_index"])
        tg_all = as_str(f["obs"]["target_gene"])
    rng = np.random.default_rng(SEED)
    ntc_rows = rng.choice(np.flatnonzero(tg_all == "non-targeting"),
                          VCC_CTRL_CELLS, replace=False)
    ctrl = thin(read_rows(ntc_rows), VCC_UMI, rng)
    X = sp.csr_matrix(ctrl)
    lib = np.asarray(X.sum(1)).ravel()
    cpm = X.multiply((TS_CELL / lib)[:, None]).tocsc()
    m_full = np.asarray(cpm.mean(0)).ravel()
    gidx = np.flatnonzero(m_full > GATE_CPM)
    gate_sym = genes[gidx]
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(cache, gate_sym)
    print(f"gate {len(gate_sym):,} / 全基因 {len(genes):,}（已缓存 {cache.name}）")
    return gate_sym


def load_source(name: str, picks: list[str], e2s: dict) -> tuple[pd.DataFrame, list[str]]:
    """逐字照 build_v8 的 rd() + ensembl->symbol + 去重，但只要该源真有的扰动列。"""
    cols = [str(c) for c in pd.read_csv(DATA / "nadig2025" / f"{name}_lfc.csv.gz",
                                        index_col=0, nrows=1).columns]
    have = [p for p in picks if p in cols]
    if not have:
        return pd.DataFrame(), []
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
    return d, have


def floor_nmae(cov: float) -> float:
    """F28 的算术：完美预测覆盖到的比例 cov，未覆盖处只能预测 0。
    nmae 的分母是参考自身的平均 |lfc|，故未覆盖基因贡献全额误差 → 下限 = 1 - cov。"""
    return 1.0 - cov


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    picks = [l.strip() for l in
             (ROOT / "experiments" / "E28-pds" / "out" / "perts.csv").read_text().split()
             if l.strip()]
    print(f"8 个扰动: {picks}\n")

    gate_sym = gate_symbols()
    G = len(gate_sym)
    gset = set(gate_sym.tolist())
    e2s_df = pd.read_csv(MAPCSV).dropna()
    e2s = dict(zip(e2s_df.ensembl, e2s_df.symbol))

    # ---- CD4 源先查，便宜 ----
    z = np.load(DATA / "external" / "cd4t_rest_de.npz", allow_pickle=True)
    cd4_perts = [str(x) for x in z["perts"]]
    cd4_genes = [str(x) for x in z["genes"]]
    print(f"[CD4+T] npz rows={z['log_fc'].shape[0]}  唯一扰动={len(set(cd4_perts))}"
          f"  基因名 {len(cd4_genes):,}  ∩gate={len(gset & set(cd4_genes)):,}"
          f"  本次 8 扰动命中={[p for p in picks if p in set(cd4_perts)]}")

    # ---- 五个 nadig2025 源 ----
    tab, loaded = [], {}
    for s in SOURCES:
        t = time.time()
        d, have = load_source(s, picks, e2s)
        if d.empty:
            tab.append((s, 0, 0, 0.0, []))
            print(f"[{s}] 无本次扰动，跳过")
            continue
        inter = pd.Index(gate_sym).intersection(d.index)
        loaded[s] = d
        tab.append((s, len(have), len(inter), len(inter) / G, have))
        print(f"[{s}] 行 {len(d):,} 唯一symbol  ∩gate {len(inter):,} "
              f"({len(inter)/G:.1%})  携带扰动 {len(have)}/{len(picks)} {have}"
              f"  ({time.time()-t:.0f}s)")

    print(f"\n{'源':16s} {'携带扰动':>8s} {'∩gate':>8s} {'覆盖率':>8s} {'nmae 下限':>10s}")
    print("-" * 58)
    for s, nh, ni, cov, _ in tab:
        print(f"{s:16s} {nh:8d} {ni:8,d} {cov:8.1%} {floor_nmae(cov):10.4f}")

    # ---- 逐扰动并集覆盖 ----
    print(f"\n{'扰动':10s} {'源数':>4s} {'K562GW':>8s} {'并集':>8s} {'新增':>7s} "
          f"{'并集覆盖':>9s} {'nmae 下限':>10s}")
    print("-" * 66)
    per_pert = {}
    for p in picks:
        srcs = [s for s in SOURCES if s in loaded and p in loaded[s].columns]
        gw = set(loaded["K562GW"].index[np.isfinite(loaded["K562GW"][p])]) & gset \
            if "K562GW" in srcs else set()
        uni = set()
        for s in srcs:
            d = loaded[s]
            uni |= set(d.index[np.isfinite(d[p])]) & gset
        per_pert[p] = (srcs, gw, uni)
        print(f"{p:10s} {len(srcs):4d} {len(gw):8,d} {len(uni):8,d} "
              f"{len(uni)-len(gw):7,d} {len(uni)/G:9.1%} {floor_nmae(len(uni)/G):10.4f}")

    mean_gw = float(np.mean([len(v[1]) for v in per_pert.values()])) / G
    mean_uni = float(np.mean([len(v[2]) for v in per_pert.values()])) / G
    print("-" * 66)
    print(f"{'平均':10s} {'':4s} {mean_gw*G:8,.0f} {mean_uni*G:8,.0f} "
          f"{(mean_uni-mean_gw)*G:7,.0f} {mean_uni:9.1%} {floor_nmae(mean_uni):10.4f}")
    print(f"K562GW 平均覆盖 {mean_gw:.2%} → nmae 下限 {floor_nmae(mean_gw):.4f}；"
          f"并集 {mean_uni:.2%} → 下限 {floor_nmae(mean_uni):.4f}；"
          f"改善 {floor_nmae(mean_gw)-floor_nmae(mean_uni):+.4f}")

    # ---- 重叠结构：与 K562GW 的符号一致率 ----
    print(f"\n{'源对':34s} {'扰动':10s} {'重叠基因':>9s} {'符号一致':>9s}")
    print("-" * 66)
    rows = []
    for s in SOURCES[1:]:
        if s not in loaded:
            continue
        for p in loaded[s].columns:
            a = loaded["K562GW"][p]
            b = loaded[s][p]
            common = a.index.intersection(b.index)
            common = common[np.isin(common, gate_sym)]
            av, bv = a.loc[common].to_numpy(), b.loc[common].to_numpy()
            ok = np.isfinite(av) & np.isfinite(bv) & (av != 0) & (bv != 0)
            agree = float(np.mean(np.sign(av[ok]) == np.sign(bv[ok]))) if ok.any() else float("nan")
            rows.append((s, p, int(ok.sum()), agree))
            print(f"{'K562GW vs '+s:34s} {p:10s} {int(ok.sum()):9,d} {agree:9.1%}")
    # 同一对上仅在 |lfc| 较大（双侧 |z| 不可得，用 |lfc| 前 10%）的子集再看一遍
    print(f"\n{'源对（|lfc| 双侧前 10%）':34s} {'扰动':10s} {'重叠基因':>9s} {'符号一致':>9s}")
    print("-" * 66)
    for s in SOURCES[1:]:
        if s not in loaded:
            continue
        for p in loaded[s].columns:
            a, b = loaded["K562GW"][p], loaded[s][p]
            common = a.index.intersection(b.index)
            common = common[np.isin(common, gate_sym)]
            av, bv = a.loc[common].to_numpy(), b.loc[common].to_numpy()
            ok = np.isfinite(av) & np.isfinite(bv) & (av != 0) & (bv != 0)
            av, bv = av[ok], bv[ok]
            if av.size == 0:
                continue
            q = np.quantile(np.abs(av), 0.9)
            r = np.quantile(np.abs(bv), 0.9)
            m = (np.abs(av) >= q) & (np.abs(bv) >= r)
            agree = float(np.mean(np.sign(av[m]) == np.sign(bv[m]))) if m.any() else float("nan")
            print(f"{'K562GW vs '+s:34s} {p:10s} {int(m.sum()):9,d} {agree:9.1%}")

    # ---- Essential 源彼此之间（与 K562GW 无关的交叉校验）----
    print(f"\nEssential 源彼此（对照：非 K562GW 的跨系一致率）")
    print("-" * 66)
    ess = [s for s in SOURCES[1:] if s in loaded]
    for s1, s2 in itertools.combinations(ess, 2):
        for p in set(loaded[s1].columns) & set(loaded[s2].columns):
            a, b = loaded[s1][p], loaded[s2][p]
            common = a.index.intersection(b.index)
            common = common[np.isin(common, gate_sym)]
            av, bv = a.loc[common].to_numpy(), b.loc[common].to_numpy()
            ok = np.isfinite(av) & np.isfinite(bv) & (av != 0) & (bv != 0)
            agree = float(np.mean(np.sign(av[ok]) == np.sign(bv[ok]))) if ok.any() else float("nan")
            print(f"{s1+' vs '+s2:34s} {p:10s} {int(ok.sum()):9,d} {agree:9.1%}")

    # ---- 填充基因（K562GW 未覆盖）的规模与 |lfc| 尺度 ----
    print(f"\n填充基因（K562GW 不覆盖、其它源覆盖）：")
    print("-" * 66)
    for p in picks:
        srcs, gw, uni = per_pert[p]
        fill = uni - gw
        if not fill:
            continue
        for s in srcs:
            if s == "K562GW":
                continue
            d = loaded[s]
            f = sorted(fill & set(d.index))
            if not f:
                continue
            v = d.loc[f, p].to_numpy()
            gwv = loaded["K562GW"][p].to_numpy()
            gwv = gwv[np.isfinite(gwv)]
            print(f"  {p:10s} {s:16s} n={len(f):6,d}  |lfc| 中位 {np.nanmedian(np.abs(v)):.4f}"
                  f"  vs K562GW 中位 {np.nanmedian(np.abs(gwv)):.4f}")

    np.save(OUT / "per_pert_cov.npy",
            np.array([[len(per_pert[p][1]), len(per_pert[p][2])] for p in picks]))
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
