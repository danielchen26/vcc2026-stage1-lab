"""E33 步骤 2-5：在 **V8 的刻度** 上做 2×3 解耦矩阵，并用已验证的 reach 估计器
同时给出 pds 与 reach 两侧的价格。不跑 cell_eval2，不写任何 .h5ad。

## 为什么必须重做 E34 的矩阵

`reanchor_probe.py` 实测（闭式）：

    A  (E34 臂A: 7,481 宇宙 / 无掩码 / scale 1.0)   pds 0.7321
    A' (V8 真实设计: 7,016 / good 掩码 / λ=0.7)      pds 0.7500
    V8 实测 (cell_eval2 对真实提交)                   pds 0.7500      ← 与 A' 逐位相同

⇒ **闭式→构建偏移 = +0.0000**（闭式路径对 pds 是忠实的，与 E28 parity 的 1.1e-16 一致）；
  A 与 V8 的全部差距 +0.0179（正好 1 个粒度单位）是**设计差异**，不是刻度差异。
  且 A''（V8 设计但 scale=1.0）也是 0.7500 → pds 对幅度不敏感，λ 不参与这笔账。

⇒ 于是 E34 的「C 比 A 好 +0.0179」是拿错基准量出来的：C=0.7500 与 V8 的 A'=0.7500
  **完全相同**，去面板均值单独一分不值。本脚本把 2×3 矩阵全部重挂到 A' 上。

## 矩阵

召集集一律 = V8 的那 288 个（λ=0.7，逐位不动），只变 `lfc_all` 侧通道：

            off = 无        off = 全基因      off = top1000
    原值      A'            (raw-all)         (raw-1000)
    去均值    A'            (dm-all)          (dm-1000)

reach 用 `reach_probe` 已验证的估计器（对 cell_eval2 实测 0.1482 差 −0.0000），
逐扰动给出 k*/N_conf，并额外给出**剔除 MAT2A** 的版本 —— MAT2A 占 V8 reach 的 84.3%
且是质量守恒伪影（`N_conf`=5，池内 0 个召集基因）。

跑法：~/vcc2026/.venv/bin/python experiments/E33-source-union/decouple_v8scale.py
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import polars as pl
from scipy import sparse as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vcclab.decoder import design_cells  # noqa: E402
from vcclab.scorer import TS_CELL  # noqa: E402
from sig_gate_probe import ref_from_matrix  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "E10-h1-erp"))
from run import VCC_CTRL_CELLS, VCC_PERT_CELLS, VCC_UMI, read_rows, thin  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "e28run", ROOT / "experiments" / "E28-pds" / "run.py")
m28 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m28)

sys.path.insert(0, str(ROOT / "experiments" / "E29-official-baseline"))
import anchor_local as A29  # noqa: E402

DATA = ROOT / "data"
H5 = DATA / "vcc2025" / "adata_Validation.h5ad"
OUT = Path(__file__).resolve().parent / "out"
ALPHA, SEED, LAMBDA, K_CALL = 0.05, 0, 0.7, 288
PURITY_FLOOR = 0.9
TOPK = 1000


def as_str(g):
    if isinstance(g, h5py.Group) and "categories" in g:
        c = [x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]
        return np.asarray(c)[g["codes"][:]]
    if isinstance(g, h5py.Group) and "values" in g:
        g = g["values"]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g[:]])


def topk_mask(v, k):
    if k is None or k >= v.size:
        return np.ones(v.size, bool)
    keep = np.zeros(v.size, bool)
    keep[np.argsort(np.abs(v))[::-1][:k]] = True
    return keep


def main() -> None:
    t0 = time.time()
    perts, genes, real, gi_full, B, S = m28._load_beta()
    m_full_e28 = m28._ctrl_profile(real)
    bts = float(m28.vcc_cfg().bulk_target_sum)
    real_bulk = m28.pseudobulk_bulk_lognorm(real, m28.PERT_COL, bulk_target_sum=bts)
    G, n_p = genes.size, len(perts)
    gran = 1.0 / (n_p * (n_p - 1))

    gate_sym = np.load(OUT / "gate_sym.npy", allow_pickle=True)
    gpos_all = pd.Index(genes)
    e34_syms = genes[gi_full]
    v8_syms = pd.Index(gate_sym).intersection(pd.Index(e34_syms))
    sel = pd.Index(e34_syms).get_indexer(v8_syms)
    gi_v8 = gpos_all.get_indexer(v8_syms)
    Bv, Sv = B[sel], S[sel]
    goodv = np.isfinite(Bv) & np.isfinite(Sv) & (Sv > 0)
    Bvc = np.where(np.isfinite(Bv), Bv, 0.0)
    Bvd = Bvc - Bvc.mean(1, keepdims=True)
    print(f"V8 宇宙 {len(v8_syms):,}  扰动 {n_p}  pds 粒度 {gran:.6f}")

    call_idx = {}
    for j in range(n_p):
        score = np.where(goodv[:, j], np.abs(Bv[:, j]), -np.inf)
        call_idx[j] = np.argsort(score)[::-1][:min(K_CALL, int(goodv[:, j].sum()))]

    def make_lf(off_src, off_topk):
        """返回 (全基因 lf 矩阵, 每扰动的 off 通道在 V8 宇宙内的布尔掩码)。"""
        lf = np.zeros((n_p, G))
        offmask = np.zeros((n_p, len(v8_syms)), bool)
        for j in range(n_p):
            c = call_idx[j]
            if off_src is not None:
                off = off_src[:, j].copy()
                off[c] = 0.0
                off[~topk_mask(off, off_topk)] = 0.0
                lf[j, gi_v8] = off
                offmask[j] = off != 0.0
            lf[j, gi_v8[c]] = LAMBDA * Bv[c, j]      # 召集集逐位 = V8
        return lf, offmask

    arms = [
        ("A' V8 现状（off = 无）",            None, None),
        ("raw-all   off=全基因 原值",         Bvc, None),
        ("raw-1000  off=top1000 原值",        Bvc, TOPK),
        ("dm-all    off=全基因 去面板均值",   Bvd, None),
        ("dm-1000   off=top1000 去面板均值",  Bvd, TOPK),
    ]
    built = {}
    print(f"\n=== pds（闭式，已证明 offset=0，故等于提交刻度）===")
    print("⚠️ 本矩阵只给 pds 与 reach 定价。expr_mse / nmae / direction_fidelity /")
    print("   sig_jaccard 在其中**未被测量**（前者已被 clamp 钉在 0，后三者是噪声）。")
    print(f"{'臂':36s} {'pds':>8s} {'Δ vs A\'':>9s} {'档':>4s}  逐扰动")
    print("-" * 92)
    base = None
    call_fp = None            # 召集集指纹：(全基因下标, lfc 值) —— 必须跨臂逐位相同
    for name, src, tk in arms:
        lf, offmask = make_lf(src, tk)
        # 要求 2（Main）：把「召集集逐位不变」写成断言而不是散文。
        fp = [(gi_v8[call_idx[j]].copy(), lf[j, gi_v8[call_idx[j]]].copy())
              for j in range(n_p)]
        if call_fp is None:
            call_fp = fp
            for j in range(n_p):
                exp = LAMBDA * Bv[call_idx[j], j]
                assert np.array_equal(fp[j][1], exp), f"{name} 扰动{j} 召集值≠λ·b"
        else:
            for j in range(n_p):
                assert np.array_equal(fp[j][0], call_fp[j][0]), f"{name} 召集下标漂移"
                assert np.array_equal(fp[j][1], call_fp[j][1]), f"{name} 召集 lfc 漂移"
        got = m28.pds(m28._pred_bulk_closed(m_full_e28, lf, perts, bts), real_bulk, genes)
        mu = float(np.mean(list(got.values())))
        if base is None:
            base = mu
        built[name] = (lf, offmask, mu)
        print(f"{name:36s} {mu:8.4f} {mu-base:+9.4f} {(mu-base)/gran:+4.0f}  "
              + " ".join(f"{got[p]:.2f}" for p in perts))
    print("-" * 92)
    print(f"断言通过：5 个臂的 288 个召集基因下标与 lfc 值逐位相同（只有 off 通道在变）。")

    # ---- reach：用已验证的估计器，需要 decoder + de_table ----
    print(f"\n=== reach（估计器已对 cell_eval2 验证，差 −0.0000）===")
    with h5py.File(H5, "r") as f:
        genes2 = as_str(f["var"]["_index"])
        tg_all = as_str(f["obs"]["target_gene"])
    assert np.array_equal(genes2, genes), "var 顺序不一致"
    rng = np.random.default_rng(SEED)
    ntc_rows = rng.choice(np.flatnonzero(tg_all == "non-targeting"),
                          VCC_CTRL_CELLS, replace=False)
    ctrl = thin(read_rows(ntc_rows), VCC_UMI, rng)
    ref = ref_from_matrix(sp.csr_matrix(ctrl), genes)
    gate = genes[np.asarray(ref.gidx)]
    assert np.array_equal(gate, gate_sym)
    zr = np.load(OUT / "real_de.npz", allow_pickle=True)
    rde = {str(p): (zr[f"p_{p}"], zr[f"l_{p}"]) for p in zr["perts"]}
    gate_pos = pd.Index(gate)
    gi_gate_v8 = gate_pos.get_indexer(v8_syms)          # V8 宇宙 → gate 内下标

    print(f"{'臂':36s} {'reach':>8s} {'去MAT2A':>8s} {'k* 逐扰动':>40s}")
    print("-" * 96)
    reach_tab = {}
    for name, src, tk in arms:
        lf_mat, offmask, _ = built[name]
        per, ks = {}, []
        for j, p in enumerate(perts):
            c = call_idx[j]
            r_set = gi_gate_v8[c]
            lfc_t = LAMBDA * Bv[c, j]
            lfc_all = None
            if src is not None:
                la = np.zeros(ref.G)
                la[gi_gate_v8] = lf_mat[j, gi_v8]      # 含召集集，design_cells 会被 r_set 覆盖
                lfc_all = la
            V = design_cells(ref, r_set, lfc_t, n_cells=VCC_PERT_CELLS, seed=SEED,
                             lfc_all=lfc_all)
            pa_p, lf_p = ref.de_table(V * (TS_CELL / V.sum(1, keepdims=True)),
                                      tie_correct=True)
            pa_r, lf_r = rde[p]
            pool = (pa_r < ALPHA) & (gate != p)
            idx = np.flatnonzero(pool)
            n_conf = int(len(idx))
            in_den = np.isfinite(lf_r[idx]) & (lf_r[idx] != 0)
            mt = in_den & (np.sign(lf_p[idx]) == np.sign(lf_r[idx]))
            order = np.lexsort((gate[idx], -np.abs(lf_p[idx]), pa_p[idx],
                                -(pa_p[idx] < ALPHA).astype(int)))
            nd = np.cumsum(in_den[order].astype(int))
            nm = np.cumsum(mt[order].astype(int))
            pur = np.where(nd > 0, nm / np.maximum(nd, 1), np.nan)
            hit = (nd > 0) & (pur >= PURITY_FLOOR)
            k_star = int(nd[hit].max()) if hit.any() else 0
            per[p] = k_star / n_conf if n_conf else np.nan
            ks.append(k_star)
        w = float(np.mean(list(per.values())))
        wo = float(np.mean([v for k, v in per.items() if k != "MAT2A"]))
        reach_tab[name] = (w, wo)
        print(f"{name:36s} {w:8.4f} {wo:8.4f}  " + " ".join(f"{k:4d}" for k in ks))
    print("-" * 96)

    # ---- 两侧一起换算成官方 avg_score ----
    v8m = {r["metric"]: r["mean"] for r in pl.read_parquet(
        ROOT / "experiments" / "E28-pds" / "out" / "agg_v8.parquet").iter_rows(named=True)}
    print(f"\n=== 官方两端刻度：pds 与 reach 同时替换（其余四成员保持 V8 实测）===")
    print(f"{'臂':36s} {'pds':>7s} {'reach':>7s} {'avg b_lo/r_lo':>14s} {'avg b_hi/r_hi':>14s}"
          f" {'Δavg':>16s}")
    print("-" * 100)
    b0 = None
    for name, _, _ in arms:
        raw = dict(v8m)
        raw["pds_cosine"] = built[name][2]
        raw["de_wilcoxon_direction_reach_raw"] = reach_tab[name][0]
        res = A29.official_two_ended_from_raw(raw)
        lo = res["b_lo/r_lo"]["avg_score"]["score"]
        hi = res["b_hi/r_hi"]["avg_score"]["score"]
        if b0 is None:
            b0 = (lo, hi)
        print(f"{name:36s} {raw['pds_cosine']:7.4f} "
              f"{raw['de_wilcoxon_direction_reach_raw']:7.4f} {lo:14.4f} {hi:14.4f}"
              f"  {lo-b0[0]:+7.4f}/{hi-b0[1]:+7.4f}")
    print("-" * 100)
    print(f"\n⚠️ 粒度：pds 与 reach 在 8 扰动下都是粗量化的（pds 一档 {gran:.4f}）。"
          f"Δ 小于一档的不可读。")
    print(f"耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
