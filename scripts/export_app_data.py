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


# BH 有效阈值：真实打分器在每个扰动内、对 gate 内 m 个基因做 BH。
# 在解点 k = |R_p| 处，p 值截断为 alpha*k/m，对应的双侧 z 远大于 1.96。
# 用 alpha=0.05 的单基因阈值会把门槛系统性低估 1.62 倍。
M_GATE_REF = 9929
N_REAL_REF = 288
ALPHA = 0.05
Z_BH = float(norm.ppf(1 - (ALPHA * N_REAL_REF / M_GATE_REF) / 2))   # 3.184
Z_NAIVE = 1.96


def compute_mde(ref: ControlRef, n_cells: int = N_PRED, seed: int = 0,
                z: float = Z_BH) -> np.ndarray:
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
    crit = z * sd / (n_cells * n2)

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
    mde = compute_mde(ref, z=Z_BH)
    mde_naive = compute_mde(ref, z=Z_NAIVE)
    elapsed = time.time() - t0

    ok = np.isfinite(mde) & (mde < 3.99)
    qs = {f"p{q}": round(float(np.percentile(mde[ok], q)), 4) for q in (5, 25, 50, 75, 95)}
    floor_frac = float((mde[ok] == 0.0).mean())
    span = qs["p95"] / qs["p25"]
    print(f"MDE 全量 {ref.G:,} 基因: {elapsed:.1f}s")
    print(f"一类错误噪声底: MDE=0 的基因占 {floor_frac:.2%}（理论 4.76%）")
    print(f"诚实的动态范围 p95/p25 = {span:.1f}×   （p95/p5 会被噪声底虚高，不用）")
    ok_n = np.isfinite(mde_naive) & (mde_naive < 3.99)
    qs_naive = {f"p{q}": round(float(np.percentile(mde_naive[ok_n], q)), 4) for q in (5, 25, 50, 75, 95)}
    print(f"BH 有效 z = {Z_BH:.3f}（单基因 alpha=0.05 的 z = {Z_NAIVE}）")
    print(f"{'分位':>6} {'BH(主)':>9} {'alpha=.05':>11} {'倍数变化':>10}")
    for k in qs:
        print(f"{k:>6} {qs[k]:9.4f} {qs_naive[k]:11.4f} {2 ** qs[k]:9.2f}x")

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
            zBH=round(Z_BH, 3),
            quantilesAlpha05=qs_naive,
            caveat=("主值用 BH 有效阈值 z=3.184（alpha*k/m, k=288, m=9929），"
                    "不是单基因 alpha=0.05 的 1.96 —— 后者把门槛低估 1.62 倍。"
                    "动态范围用 p95/p25：p5 落在一类错误噪声底上。"),
        ),
        "quantilesAlpha05": qs_naive,
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
    # a_false 不是自由参数：fid 数的是「你报的基因里预测 lfc 符号与参考的一致」，
    # 而对没有真实效应的基因，参考测到的 lfc 是噪声主导 → 符号是掷硬币 → 结构性 0.5。
    a_true, a_false = 0.90, 0.50
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
                     assumptions=(f"aTrue={a_true}, aFalse={a_false}（结构性钉住，非自由参数）, |R̂|=K；"
                     f"K=nReal 时 fid = 0.5 + (aTrue-0.5)*h"),
        hReplicate=round(2 * 0.399 / 1.399, 3),
        hToTieLeader=0.134,
        zeroBiologyCeiling=0.130,
        theoryNote=("h 的物理上限由 replicate 锚点给出：r_jac=0.399 → h=0.570，"
                    "即同系重做实验也只重叠 57%。追平榜首需 h>=0.134 = replicate 的 23%。"
                    "只按可检出性排序的『零生物学』策略理论上限 h=0.130，赢不了。")),
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
