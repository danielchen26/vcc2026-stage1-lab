#!/usr/bin/env python
"""导出交互式 app 需要的真实数据。

用法：  ~/vcc2026/.venv/bin/python scripts/export_app_data.py
幂等。输出到 app/src/data/。

产出：
  mde.json      最小可检出效应：全 9,929 个门内基因的分布 + 200 个具名基因 + 散点
  psi.json      拨盘演示：SELENOT 的真实对照 ECDF（2048 分位点）+ 12 行精确 dial 表
  callset.json  报数 K 的权衡曲线（纯解析）
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

# 复用已验证的 ControlRef（唯一可信基线）
sys.path.insert(0, str(Path("~/vcc2026").expanduser()))
from vcc_local import ControlRef  # noqa: E402

DATA = Path("~/vcc2026").expanduser()
OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "data"
OUT.mkdir(parents=True, exist_ok=True)

TOOLS = "numpy 2.5.2 · scipy 1.18.1 · cell-eval2 0.16.0"
TODAY = "2026-08-27"
N_PRED = 400


def meta(**kw) -> dict:
    return {
        "generatedBy": "scripts/export_app_data.py",
        "date": TODAY,
        "machine": "Apple M1 Pro / 16 GB / no CUDA",
        "toolVersions": TOOLS,
        **kw,
    }


def compute_mde(ref: ControlRef, n_cells: int = N_PRED, seed: int = 0) -> np.ndarray:
    """每个门内基因的最小可检出效应 |lfc|。

    定义：从对照里自举 n_cells 个细胞的该基因 CPM 值，整体乘 2**lfc 后取整，
    对 lfc 在 [0, 4] 上二分 18 步，找使 |psi_bar/n_ctrl - 0.5| >= 1.96*sigma/(n1*n2)
    的最小 lfc。sigma 用未并列校正版，以便与已报告的参考值可比。
    """
    rg = np.random.default_rng(seed)
    n2 = ref.n_ctrl
    boot = rg.choice(n2, n_cells, replace=False)
    base = np.asarray(ref._cpm_csr[boot][:, ref.gidx].todense())
    sd = math.sqrt(n_cells * n2 * (n_cells + n2 + 1) / 12.0)
    crit = 1.96 * sd / (n_cells * n2)

    out = np.full(ref.G, np.nan)
    for j in range(ref.G):
        col = base[:, j]
        # 先显式评估 lfc = 0：若自举样本本身已越过阈值，MDE = 0。
        # 这不是 bug —— 它就是 alpha=0.05 的名义一类错误率（理论 4.76%，见 meta.floorFrac）。
        if abs(ref.psi(j, np.round(col)).mean() / n2 - 0.5) >= crit:
            out[j] = 0.0
            continue
        lo, hi = 0.0, 4.0
        for _ in range(18):
            mid = 0.5 * (lo + hi)
            d = abs(ref.psi(j, np.round(col * 2.0**mid)).mean() / n2 - 0.5)
            if d < crit:
                lo = mid
            else:
                hi = mid
        out[j] = 0.5 * (lo + hi)
    return out


def export_mde(ref: ControlRef, genes: np.ndarray) -> dict:
    t0 = time.time()
    mde = compute_mde(ref)
    elapsed = time.time() - t0

    ok = np.isfinite(mde) & (mde < 3.99)
    qs = {f"p{q}": round(float(np.percentile(mde[ok], q)), 4) for q in (5, 25, 50, 75, 95)}
    floor_frac = float((mde[ok] == 0.0).mean())
    span = qs["p95"] / qs["p25"]
    print(f"MDE 全量 {ref.G:,} 基因: {elapsed:.1f}s")
    print(f"一类错误噪声底: MDE=0 的基因占 {floor_frac:.2%}（理论 4.76%）")
    print(f"诚实的动态范围 p95/p25 = {span:.1f}×   （p95/p5 会被噪声底虚高，不用）")
    ref_vals = {"p5": 0.032, "p25": 0.097, "p50": 0.177, "p75": 0.351, "p95": 1.044}
    print(f"{'分位':>6} {'本次':>9} {'参考(2k抽样)':>14} {'相对差':>8}")
    for k, v in qs.items():
        r = ref_vals[k]
        print(f"{k:>6} {v:9.4f} {r:14.4f} {abs(v - r) / r:7.1%}")

    edges = np.linspace(0.0, 2.0, 61)
    counts, _ = np.histogram(np.clip(mde[ok], 0, 2.0), bins=edges)

    sym = genes[ref.gidx]
    zero_frac = ref._nzero / ref.n_ctrl

    order = np.argsort(mde[ok])
    idx_ok = np.flatnonzero(ok)[order]
    pick = idx_ok[np.linspace(0, len(idx_ok) - 1, 200).astype(int)]
    named = [
        {
            "sym": str(sym[j]),
            "mde": round(float(mde[j]), 4),
            "ctrlMeanCpm": round(float(ref.m_gate[j]), 2),
            "zeroFrac": round(float(zero_frac[j]), 4),
        }
        for j in pick
    ]

    rg = np.random.default_rng(0)
    sc = rg.choice(idx_ok, min(1200, len(idx_ok)), replace=False)
    scatter = [
        {"x": round(float(ref.m_gate[j]), 3), "y": round(float(mde[j]), 3)} for j in sc
    ]

    return {
        "meta": meta(
            context="A", nGenes=int(ref.G), nResolved=int(ok.sum()),
            elapsedSec=round(elapsed, 1), sigma="uncorrected",
            floorFrac=round(floor_frac, 4),
            spanP95P25=round(span, 1),
            caveat=("MDE=0 的那 ~5% 基因是 alpha=0.05 的名义一类错误率，不是可检出性；"
                    "故动态范围用 p95/p25。另：真实打分器在 ~9,929 个基因上做 BH，"
                    "比单基因 alpha=0.05 严得多，所以此处的 MDE 是操作门槛的下界。"),
        ),
        "quantiles": qs,
        "hist": {"binEdges": [round(float(e), 4) for e in edges],
                 "counts": [int(c) for c in counts]},
        "genes": named,
        "scatter": scatter,
    }


def export_psi(ref: ControlRef, genes: np.ndarray) -> dict:
    """从已验证的 dossier 版本复制，补 mdeThis 与 meta。"""
    src = Path("~/code/vcc2026-dossier/src/data/psi.json").expanduser()
    payload = json.loads(src.read_text())
    sym = genes[ref.gidx]
    j = int(np.flatnonzero(sym == payload["gene"])[0])
    m = compute_mde_single(ref, j)
    payload["mdeThis"] = round(float(m), 4)
    payload["meta"] = meta(context=payload["context"], gene=payload["gene"],
                           note="ecdfQ/dial 数值原样保留，未改动")
    print(f"psi.json: 基因 {payload['gene']}  mdeThis = {m:.4f}")
    return payload


def compute_mde_single(ref: ControlRef, j: int, n_cells: int = N_PRED, seed: int = 0) -> float:
    rg = np.random.default_rng(seed)
    n2 = ref.n_ctrl
    boot = rg.choice(n2, n_cells, replace=False)
    col = np.asarray(ref._cpm_csr[boot][:, ref.gidx[j]].todense()).ravel()
    sd = math.sqrt(n_cells * n2 * (n_cells + n2 + 1) / 12.0)
    crit = 1.96 * sd / (n_cells * n2)
    lo, hi = 0.0, 4.0
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        d = abs(ref.psi(j, np.round(col * 2.0**mid)).mean() / n2 - 0.5)
        if d < crit:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def export_callset() -> dict:
    """报数 K 的权衡，纯解析。nReal 由官方 jac 基线锚点反推：0.029 × 9929 ≈ 288。"""
    n_real = 288
    anchors = {"jac": {"b": 0.029, "r": 0.399}, "fid": {"b": 0.513, "r": 0.813}}
    a_true, a_false = 0.90, 0.55
    ks = np.unique(np.round(np.logspace(math.log10(20), math.log10(3000), 80)).astype(int))

    curves = []
    for h in (0.1, 0.2, 0.3, 0.4, 0.5):
        pts = []
        for k in ks:
            hit = h * min(int(k), n_real)
            jac = hit / (int(k) + n_real - hit)
            denom = max(int(k), n_real)
            fid = (a_true * hit + a_false * (int(k) - hit)) / denom
            pts.append({
                "K": int(k),
                "jacRaw": round(float(jac), 4),
                "jacScaled": round(float((jac - anchors["jac"]["b"]) /
                                         (anchors["jac"]["r"] - anchors["jac"]["b"])), 4),
                "fidRaw": round(float(fid), 4),
                "fidScaled": round(float((fid - anchors["fid"]["b"]) /
                                         (anchors["fid"]["r"] - anchors["fid"]["b"])), 4),
            })
        best = max(pts, key=lambda p: p["jacScaled"])
        if abs(h - 0.3) < 1e-9:
            print(f"callset: h=0.3 时 jacScaled 最大在 K={best['K']} "
                  f"(nReal={n_real}, 相对偏差 {abs(best['K'] - n_real) / n_real:.0%})")
        curves.append({"h": h, "points": pts})

    return {
        "meta": meta(nRealSource="从官方 jac 基线锚点 0.021–0.037 × 门内基因数反推",
                     assumptions=f"aTrue={a_true}, aFalse={a_false}, |R̂|=K"),
        "nReal": n_real,
        "anchors": anchors,
        "curves": curves,
    }


def main() -> None:
    genes = pd.read_csv(DATA / "gene_names.csv")["gene_name"].to_numpy()
    t0 = time.time()
    ref = ControlRef(DATA / "context_A.h5ad", genes)
    print(f"ControlRef 加载: {time.time() - t0:.1f}s  gate={ref.G}  对照={ref.n_ctrl}\n")

    for name, payload in (
        ("mde.json", export_mde(ref, genes)),
        ("psi.json", export_psi(ref, genes)),
        ("callset.json", export_callset()),
    ):
        p = OUT / name
        p.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        print(f"→ {p.relative_to(p.parent.parent.parent.parent)}  {p.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
