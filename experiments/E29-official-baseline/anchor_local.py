"""E29b：在我们自己的 real.h5ad 上算 cell_eval2 官方的 split-half replicate anchor（刻度 1 端），
然后把已落盘的 agg_*.parquet 全家（A / V2..V8 / 退化基线 / 官方 generic 基线）重算到
两端刻度 (u-b)/(r-b) 上。

为什么不直接调 compute_replicate_anchor(n_splits=5)：它是一个不可恢复的 5 次循环，
单次 split 在本机约 10-20 min，5 次撞上 4 个并发兄弟进程会跑到 1.5-2 h。本驱动逐 split
落盘，任何时刻都能用「已完成的 k 个 split」组装出一个诚实的 anchor，并在 RESULT.md 里
标明 k。用的全是 anchor.py 自己的函数（_derive_seeds / _score_one_split / _lfc_nmae_raw /
_compute_de_side / _resolve_target_sum_from_control），没有一行打分算术是我重写的。

from_replicate 列同样不是我算的：走 score.py 自己的 _replicate_entries(score.py:377) +
_reference_column(score.py:199)，即 score_metrics 内部 score.py:898/917-919 的同一条路。

用法：
  python anchor_local.py splits [n]   # 逐 split 计算并落盘（默认 5，可断点续跑）
  python anchor_local.py assemble     # 用已有 split 组装 anchor_agg.parquet
  python anchor_local.py rescore      # 全家重算 from_baseline + from_replicate
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
E27 = ROOT / "experiments" / "E27-six-metrics" / "out"
E28 = ROOT / "experiments" / "E28-pds" / "out"
OUT = Path(__file__).resolve().parent / "out"
REAL = E27 / "real.h5ad"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_official_baseline import SCORED, cfg_v8, wide  # noqa: E402

N_SPLITS = 5          # 官方 anchor 的 split 数（anchor.py:183 的默认值）
BASE_SEED = 0         # anchor.py:182 的默认值


def stage_splits(n: int = N_SPLITS) -> None:
    """逐 seed 算 split-half replicate，每算完一个立刻落盘。"""
    import anndata as ad
    from cell_eval2.anchor import (
        _compute_de_side, _derive_seeds, _lfc_nmae_names, _lfc_nmae_raw,
        _resolve_target_sum_from_control, _score_one_split,
    )
    from cell_eval2.catalog import resolve_metrics
    from cell_eval2.run import metric_output_names

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = cfg_v8()
    assert not cfg.autodetect_input_type and cfg.version != "v1"   # anchor.py:205 的前置条件
    t0 = time.time()
    real_ad = ad.read_h5ad(str(REAL))       # load_anndata(real, backed=False)，同 anchor.py:214
    print(f"real 载入 {time.time()-t0:.0f}s  {real_ad.shape}", flush=True)

    metrics = list(resolve_metrics(cfg.metrics, version=cfg.version)[0])
    expected = metric_output_names(cfg)
    seeds = _derive_seeds(BASE_SEED, n)
    print(f"seeds={seeds}\nexpected={expected}", flush=True)
    nmae_names = _lfc_nmae_names(expected)

    de_full = None
    if nmae_names:
        t = time.time()
        nmae_cfg = _resolve_target_sum_from_control(cfg, real_ad)
        de_full = _compute_de_side(real_ad, cfg=nmae_cfg, fp=None, store=None, side="real")
        print(f"de_full 完成 {time.time()-t:.0f}s  type={type(de_full)}", flush=True)

    for i, seed in enumerate(seeds):
        f = OUT / f"anchor_split_{i}.json"
        if f.exists():
            print(f"split {i} 已有，跳过", flush=True)
            continue
        t = time.time()
        agg, counts = _score_one_split(real_ad, cfg, seed, metrics)
        vals = dict(zip(agg["metric"].to_list(), agg["mean"].to_list()))
        ns = dict(zip(counts["metric"].to_list(), counts["n_perturbations"].to_list()))
        if nmae_names:
            raw, n_ref = _lfc_nmae_raw(real_ad, cfg, seed, de_full)
            for m in nmae_names:
                vals[m], ns[m] = raw, n_ref
        absent = [m for m in expected
                  if vals.get(m) is None or not math.isfinite(float(vals[m]))]
        f.write_text(json.dumps({"split_index": i, "seed": int(seed),
                                 "vals": {m: float(vals[m]) for m in expected
                                          if vals.get(m) is not None},
                                 "n_perturbations": {m: (None if ns.get(m) is None
                                                         else int(ns[m])) for m in expected},
                                 "absent": absent,
                                 "seconds": round(time.time() - t, 1)}, indent=2))
        print(f"split {i} (seed {seed}) 完成 {time.time()-t:.0f}s absent={absent}", flush=True)
    print(f"splits 阶段总耗时 {time.time()-t0:.0f}s", flush=True)


def stage_assemble() -> None:
    """用已落盘的 split 组装 anchor 帧。分组聚合逐字照 anchor.py:282-299。"""
    from cell_eval2.anchor import (
        FULL_GATE_RAW, SPLIT_HALF_RAW, _ANCHOR_SCHEMA, _SPLITS_SCHEMA, _lfc_nmae_names,
    )
    from cell_eval2.run import metric_output_names

    cfg = cfg_v8()
    expected = metric_output_names(cfg)
    nmae_names = _lfc_nmae_names(expected)
    files = sorted(OUT.glob("anchor_split_*.json"))
    if not files:
        raise SystemExit("还没有任何 split")
    rows = []
    for fp in files:
        d = json.loads(fp.read_text())
        assert not d["absent"], (fp.name, d["absent"])
        rows.append(pl.DataFrame({
            "split_index": [d["split_index"]] * len(expected),
            "seed": [d["seed"]] * len(expected),
            "metric": expected,
            "value": [float(d["vals"][m]) for m in expected],
            "n_perturbations": [d["n_perturbations"].get(m) for m in expected],
        }, schema=_SPLITS_SCHEMA))
    splits = pl.concat(rows).sort("split_index", "metric")
    anchor = (
        splits.group_by("metric")
        .agg(replicate=pl.col("value").mean(),
             replicate_sd=pl.col("value").std(ddof=0),
             replicate_min=pl.col("value").min(),
             replicate_max=pl.col("value").max(),
             n_perturbations_min=pl.col("n_perturbations").min(),
             n_perturbations_max=pl.col("n_perturbations").max())
        .with_columns(estimator=pl.when(pl.col("metric").is_in(nmae_names))
                      .then(pl.lit(FULL_GATE_RAW, dtype=pl.Utf8))
                      .otherwise(pl.lit(SPLIT_HALF_RAW, dtype=pl.Utf8)))
        .select(list(_ANCHOR_SCHEMA)).sort("metric")
    )
    splits.write_parquet(OUT / "anchor_splits.parquet")
    anchor.write_parquet(OUT / "anchor_agg.parquet")
    n = splits["split_index"].n_unique()
    print(f"n_splits={n}  seeds={sorted(set(splits['seed'].to_list()))}")
    with pl.Config(tbl_cols=-1, tbl_width_chars=200, fmt_str_lengths=50):
        print(anchor)


# --- 两端刻度 ---------------------------------------------------------------

#: docs/01-scoring.md:167-174 从实时榜反算出的官方 b / r（三个官方 context 的区间）。
#: 六个指标的官方名 -> (b_lo, b_hi, r_lo, r_hi)
OFFICIAL_BR = {
    "pds_cosine":                              (0.500, 0.500, 0.927, 0.984),
    "expr_mse_unbiased_capped_norm":           (0.986, 0.992, 0.028, 0.045),
    "de_wilcoxon_sig_jaccard":                 (0.021, 0.037, 0.375, 0.423),
    "de_wilcoxon_lfc_nmae":                    (1.0009, 1.0017, 0.369, 0.431),
    "de_wilcoxon_direction_fidelity_yield_raw": (0.505, 0.522, 0.795, 0.832),
    "de_wilcoxon_direction_reach_raw":         (0.047, 0.097, 0.958, 0.978),
}


def _two_ended(raw: dict, b: dict, r: dict) -> dict:
    """用 cell_eval2 自己的 score_one + catalog policy 算 (u-b)/(r-b)。

    不手写算术：把 policy 的 anchor 换成实测 r（score.py:431 的同一次 replace），
    再交给 scoring.score_one（score.py:270 的同一次调用）。
    """
    from dataclasses import replace as dc_replace

    from cell_eval2.catalog import CATALOG, _NAME_TO_CANONICAL
    from cell_eval2.scoring import is_degenerate, score_one

    out = {}
    for m in SCORED:
        canon = _NAME_TO_CANONICAL.get(m, m)
        spec = CATALOG[canon]
        policy = dc_replace(spec.scoring, anchor=float(r[m]), allow_negative_baseline=False)
        deg = is_degenerate(float(b[m]), policy)
        out[m] = {"score": None if deg else float(score_one(raw[m], float(b[m]), policy)),
                  "degenerate": bool(deg)}
    vals = [v["score"] for v in out.values() if v["score"] is not None]
    out["avg_score"] = {"score": float(sum(vals) / len(vals)) if vals else None,
                        "n": len(vals)}
    return out


def official_two_ended_from_raw(raw: dict) -> dict:
    """公开入口（供兄弟 agent 只读调用）：给定六个原始均值，返回官方两端刻度的分数。


    参数 raw：{metric_name: raw_mean}，键用官方名（见 SCORED）。多余的键会被忽略，
    缺任何一个 SCORED 键都会 KeyError —— 故意的，缺项会让 avg_score 在更少的成员上求均值。

    返回 {"b_lo/r_lo": {...}, "b_hi/r_hi": {...}}，每个里面是
    {metric: {"score": float|None, "degenerate": bool}} 外加
    {"avg_score": {"score": float, "n": int}}。

    **两端都要报，不要取中点。** b/r 是从实时榜反算的区间（docs/01-scoring.md:167-174），
    跨三个官方 context；单点会把不确定度藏起来。
    """
    return {lab: _two_ended(raw, {m: v[bi] for m, v in OFFICIAL_BR.items()},
                            {m: v[ri] for m, v in OFFICIAL_BR.items()})
            for lab, bi, ri in (("b_lo/r_lo", 0, 2), ("b_hi/r_hi", 1, 3))}


def official_two_ended(agg_parquet) -> dict:
    """同上，但直接吃 aggregate_metrics 落盘的 parquet（列 metric/mean/agg）。"""
    raw = {r["metric"]: r["mean"]
           for r in pl.read_parquet(str(agg_parquet)).iter_rows(named=True)}
    out = official_two_ended_from_raw(raw)
    out["raw"] = {m: raw[m] for m in SCORED}
    return out


def _from_baseline(user_pq: Path, base_pq: Path, tag: str) -> dict:
    from cell_eval2.score import score_metrics
    u, b = wide(user_pq, OUT / f"_w_{tag}_u.csv"), wide(base_pq, OUT / f"_w_{tag}_b.csv")
    assert pl.read_csv(u).columns == pl.read_csv(b).columns
    res = score_metrics(str(u), str(b), comparison_statistic="mean")
    return {r["metric"]: r["from_baseline"] for r in res.iter_rows(named=True)}


#: cell_eval2.scales.SCALES 里唯一的冻结刻度（scales.py:310-452）。0 = 无技巧点
#: （pds 的均匀秩 0.5 / mse 的「原样贴对照」1.0 / nmae 的 lfc_hat=0 即 1.0 / jac 的下界 0 /
#: fid 的掷硬币 0.5 / reach 的最小值 0），1 = 完美。不需要任何官方 bundle。
FROZEN_SCALE = "low-random_high-1_v10"


def _frozen_scale(user_pq: Path, base_pq: Path, tag: str) -> dict:
    """冻结刻度列。scale 只 ADD 一列，绝不动 from_baseline（score.py:985-988）。"""
    from cell_eval2.score import score_metrics
    u, b = wide(user_pq, OUT / f"_w_{tag}_u.csv"), wide(base_pq, OUT / f"_w_{tag}_b.csv")
    res = score_metrics(str(u), str(b), comparison_statistic="mean", scale=FROZEN_SCALE)
    return {r["metric"]: r[FROZEN_SCALE] for r in res.iter_rows(named=True)}


def _from_replicate(user_pq: Path, base_pq: Path, anchor: pl.DataFrame, tag: str) -> dict:
    """走 score.py 内部那条路：_replicate_entries + _reference_column。"""
    from cell_eval2.score import _replicate_entries, _reference_column, score_metrics
    u, b = wide(user_pq, OUT / f"_w_{tag}_u.csv"), wide(base_pq, OUT / f"_w_{tag}_b.csv")
    du, db = pl.read_csv(u), pl.read_csv(b)
    urow = du.filter(pl.col("statistic") == "mean").drop("statistic")
    brow = db.filter(pl.col("statistic") == "mean").drop("statistic")
    base_by_name = dict(zip(brow.columns, brow.row(0)))
    entries = _replicate_entries(base_by_name, anchor)
    out = score_metrics(str(u), str(b), comparison_statistic="mean")
    col = _reference_column(out["metric"].to_list(), urow.row(0), urow.columns, entries,
                           column="from_replicate", label="the replicate anchor")
    out = out.with_columns(col)
    return {r["metric"]: (r["from_baseline"], r["from_replicate"])
            for r in out.iter_rows(named=True)}


VARIANTS = {
    "A":  E27 / "agg_pred.parquet",
    "V2": E28 / "agg_v2.parquet", "V3": E28 / "agg_v3.parquet",
    "V4": E28 / "agg_v4.parquet", "V5": E28 / "agg_v5.parquet",
    "V6": E28 / "agg_v6.parquet", "V7": E28 / "agg_v7.parquet",
    "V8": E28 / "agg_v8.parquet",
}


def stage_rescore() -> None:
    OLD = E27 / "agg_base.parquet"
    OFF = OUT / "agg_official_base.parquet"
    report: dict = {"scale_official_scraped": {}, "scale_local_anchor": {},
                    "from_baseline_degenerate_base": {}}

    raw = {k: {r["metric"]: r["mean"] for r in pl.read_parquet(v).iter_rows(named=True)}
           for k, v in list(VARIANTS.items()) + [("old_base", OLD)]
           + ([("off_base", OFF)] if OFF.exists() else [])}
    report["raw"] = raw

    # (2) 官方刻度：docs/01-scoring.md 抓到的 b / r，区间两端各算一次
    print("\n" + "=" * 104)
    print("官方两端刻度 (u-b)/(r-b)，b / r 取自 docs/01-scoring.md:167-174（实时榜反算）")
    print("=" * 104)
    for lab, bi, ri in (("b_lo/r_lo", 0, 2), ("b_hi/r_hi", 1, 3)):
        b = {m: v[bi] for m, v in OFFICIAL_BR.items()}
        r = {m: v[ri] for m, v in OFFICIAL_BR.items()}
        print(f"\n--- {lab} ---")
        hdr = f"{'指标':44s}" + "".join(f"{k:>9s}" for k in VARIANTS)
        print(hdr); print("-" * len(hdr))
        res = {k: _two_ended(raw[k], b, r) for k in VARIANTS}
        report["scale_official_scraped"][lab] = res
        for m in SCORED + ("avg_score",):
            cells = "".join(
                f"{(res[k][m]['score'] if res[k][m]['score'] is not None else float('nan')):9.4f}"
                for k in VARIANTS)
            print(f"{m:44s}{cells}")

    # (3) 本地 anchor 刻度
    apq = OUT / "anchor_agg.parquet"
    if apq.exists():
        anchor = pl.read_parquet(apq)
        nsp = pl.read_parquet(OUT / "anchor_splits.parquet")["split_index"].n_unique()
        rep = dict(zip(anchor["metric"].to_list(), anchor["replicate"].to_list()))
        print("\n" + "=" * 104)
        print(f"本地 split-half anchor（n_splits={nsp}，我们自己的 8 扰动 real.h5ad）")
        print("=" * 104)
        for base_tag, base_pq in (("退化基线", OLD),) + (
                (("官方 generic 基线", OFF),) if OFF.exists() else ()):
            print(f"\n--- 0 端 = {base_tag} ---")
            hdr = f"{'指标':44s}{'r(local)':>10s}" + "".join(f"{k:>9s}" for k in VARIANTS)
            print(hdr); print("-" * len(hdr))
            res = {k: _from_replicate(v, base_pq, anchor, f"{k}_{base_tag[:3]}")
                   for k, v in VARIANTS.items()}
            report["scale_local_anchor"][base_tag] = {
                k: {m: {"from_baseline": v[0], "from_replicate": v[1]}
                    for m, v in d.items()} for k, d in res.items()}
            for m in SCORED + ("avg_score",):
                cells = "".join(f"{(res[k].get(m, (None, None))[1] or float('nan')):9.4f}"
                                for k in VARIANTS)
                print(f"{m:44s}{rep.get(m, float('nan')):10.4f}{cells}")
        report["local_anchor"] = {"n_splits": nsp,
                                  "frame": anchor.to_dicts()}
    else:
        print("\n[本地 anchor 尚未就绪：anchor_agg.parquet 不存在]")

    # 参照：仓库内一直在读的 from_baseline（anchor 固定 0/1）
    print("\n" + "=" * 104)
    print("参照：仓库内一直在读的 from_baseline（anchor 固定在 0/1，0 端=退化基线）")
    print("=" * 104)
    hdr = f"{'指标':44s}" + "".join(f"{k:>9s}" for k in VARIANTS)
    print(hdr); print("-" * len(hdr))
    fb = {k: _from_baseline(v, OLD, f"fb_{k}") for k, v in VARIANTS.items()}
    report["from_baseline_degenerate_base"] = fb
    for m in SCORED + ("avg_score",):
        print(f"{m:44s}" + "".join(f"{fb[k].get(m, float('nan')):9.4f}" for k in VARIANTS))

    # 冻结刻度：0 = 无技巧点，1 = 完美。不需要 bundle，也不需要任何实测基线。
    print("\n" + "=" * 104)
    print(f"冻结刻度 {FROZEN_SCALE}（0 = 无技巧点，1 = 完美；scales.py:310-452，无需 bundle）")
    print("=" * 104)
    print(hdr); print("-" * len(hdr))
    fs = {k: _frozen_scale(v, OLD, f"fs_{k}") for k, v in VARIANTS.items()}
    report["frozen_scale"] = {"name": FROZEN_SCALE, "per_variant": fs}
    for m in SCORED + ("avg_score",):
        print(f"{m:44s}" + "".join(
            f"{(fs[k].get(m) if fs[k].get(m) is not None else float('nan')):9.4f}"
            for k in VARIANTS))

    # Main 的两个点问：mse 这一项是不是全是分母伪影
    from cell_eval2.catalog import CATALOG
    sc = CATALOG["expr_mse_unbiased_capped_norm"].scoring
    M = "expr_mse_unbiased_capped_norm"
    print("\n" + "=" * 104)
    print(f"expr_mse_unbiased_capped_norm 专项：policy anchor={sc.anchor} direction="
          f"{sc.direction} penalty={sc.penalty} clamp_low={sc.clamp_low}")
    print(f"冻结刻度 base = 1.0 = 「原样贴对照」；> 1.0 即已过无技巧点")
    print("=" * 104)
    print(f"{'变体':8s}{'raw':>10s}{'过无技巧点?':>13s}{'1 - raw/退化base':>18s}"
          f"{'冻结刻度分':>12s}")
    mse = {}
    for k in list(VARIANTS) + ["old_base"] + (["off_base"] if "off_base" in raw else []):
        u = raw[k][M]
        fbv = 1.0 - u / raw["old_base"][M]
        mse[k] = {"raw": u, "past_no_skill": bool(u > 1.0),
                  "one_minus_u_over_degenerate_base": fbv,
                  "frozen_scale": fs[k][M] if k in fs else None}
        print(f"{k:8s}{u:10.4f}{('是' if u > 1.0 else '否'):>13s}{fbv:18.4f}"
              f"{(mse[k]['frozen_scale'] if mse[k]['frozen_scale'] is not None else float('nan')):12.4f}")
    report["mse_audit"] = {"policy": {"anchor": sc.anchor, "direction": sc.direction,
                                      "penalty": sc.penalty, "clamp_low": sc.clamp_low},
                           "per_variant": mse}

    (OUT / "rescore.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\n已存 {OUT/'rescore.json'}")


if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "splits"
    if st == "splits":
        stage_splits(int(sys.argv[2]) if len(sys.argv) > 2 else N_SPLITS)
    else:
        {"assemble": stage_assemble, "rescore": stage_rescore}[st]()
