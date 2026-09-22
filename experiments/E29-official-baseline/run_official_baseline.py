"""E29：把分母换成 cell_eval2 自带的官方 generic-response 基线，重算 V8 / A 的 avg_score。

不改任何 V8 旋钮，不重建 V8。唯一变化是**比较器（分母）**：
  旧：对照均值 tile 400 遍（build_v8.py:199-203 的 X_base）→ E27/out/agg_base.parquet
  新：cell_eval2.build_generic_baseline(..., emit="dispersed")，即官方 0 端刻度

F29 单旋钮生成器纪律（build_v8.py 文本 + 恰好一次 str.replace + assert）在此**不适用**：
本实验不产生新的 variant build，V8 的 h5ad 一行代码都没重跑，改的是 score_metrics 的
第二个参数。可归因性由「两侧 agg 逐字复用已落盘的 parquet」保证。

磁盘：build_generic_baseline(save_pred=None) 在内存里建预测并直接评分，**一个字节的
.h5ad 都不落盘**（baseline.py:1464-1465 只在 save_pred 非 None 时写）。因此本实验对
23 GB 余量的占用是 0，无需 AUTO_CLEAN。

用法：
  .../python run_official_baseline.py baseline   # 建官方基线并算六指标（慢，~25 min）
  .../python run_official_baseline.py score      # 重算 V8 / A / 旧基线 的 from_baseline
"""
from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
E27 = ROOT / "experiments" / "E27-six-metrics" / "out"
E28 = ROOT / "experiments" / "E28-pds" / "out"
OUT = Path(__file__).resolve().parent / "out"
REAL = E27 / "real.h5ad"          # 只读共享真值，绝不删改

SCORED = ("de_wilcoxon_sig_jaccard", "de_wilcoxon_lfc_nmae",
          "de_wilcoxon_direction_fidelity_yield_raw", "de_wilcoxon_direction_reach_raw",
          "pds_cosine", "expr_mse_unbiased_capped_norm")


def cfg_v8():
    """与 score_v8.py 逐字相同的评分配置。"""
    from cell_eval2 import EvalConfig
    cfg = EvalConfig.from_preset("vcc2026")
    cfg = replace(cfg, pert_col="target_gene", device="cpu")
    cfg = replace(cfg, de=replace(cfg.de, backend="scanpy"))
    return cfg


def wide(pq: Path, csv: Path) -> Path:
    """score_v8.py 的同名 helper，逐字照抄，保证列序与 agg_v8_wide.csv 一致。"""
    d = pl.read_parquet(pq)
    w = {"statistic": ["mean"]}
    for r in d.iter_rows(named=True):
        w[r["metric"]] = [r["mean"]]
    pl.DataFrame(w).write_csv(csv)
    return csv


def stage_baseline() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    logfile = OUT / "baseline_build.log"
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(logfile, mode="w"),
                                  logging.StreamHandler(sys.stdout)])
    from cell_eval2 import aggregate_metrics, build_generic_baseline

    t0 = time.time()
    cfg = cfg_v8()
    print(f"=== stage=baseline ===\nreal={REAL}\nversion={cfg.version} "
          f"input_type={cfg.input_type} pert_col={cfg.pert_col} control={cfg.control}")
    caught: list[str] = []
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        res = build_generic_baseline(
            str(REAL),
            config=cfg,
            exclude_target_gene=True,   # vcc2026 preset 的 discrimination 侧也是 True
            emit="dispersed",           # 官方支持的 counts 构造；tile 是被弃用的偏置臂
            seed=0,
            save_pred=None,             # 不落盘：磁盘是硬约束，评分在内存里完成
            allow_degenerate=True,      # 先拿到六个原始值，退化与否交给 score_metrics 判
        )
        caught = [f"{x.category.__name__}: {x.message}" for x in wl]
    for c in caught:
        print("WARN(py) " + c)
    (OUT / "baseline_py_warnings.txt").write_text("\n".join(caught) + "\n")

    agg = aggregate_metrics(res.results)          # 与 agg_v8.parquet 结构逐字相同
    agg.write_parquet(OUT / "agg_official_base.parquet")
    res.agg.write_csv(OUT / "agg_official_base_libwide.csv")   # 库自带 wide，交叉核对
    res.results.write_parquet(OUT / "results_official_base.parquet")
    p = res.profile
    meta = dict(res.meta)
    meta["_profile"] = {"n_perturbations": p.n_perturbations,
                        "n_genes": int(p.genes.size),
                        "exclude_target_gene": bool(p.exclude_target_gene),
                        "n_excluded": int(p.n_excluded),
                        "values_sum": float(p.values.sum()),
                        "values_max": float(p.values.max())}
    (OUT / "baseline_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True,
                                                       default=str))
    print(f"\nprofile: n_pert={p.n_perturbations} n_genes={p.genes.size} "
          f"n_excluded={p.n_excluded} sum={p.values.sum():.1f}")
    print("degenerate_metrics:", json.dumps(meta.get("degenerate_metrics"), default=str))
    print(agg)
    print(f"\nbaseline 完成 {time.time()-t0:.0f}s")


def _score(user_pq: Path, base_pq: Path, tag: str) -> dict:
    from cell_eval2.score import score_metrics
    u = wide(user_pq, OUT / f"_wide_{tag}_user.csv")
    b = wide(base_pq, OUT / f"_wide_{tag}_base.csv")
    du, db = pl.read_csv(u), pl.read_csv(b)
    assert du.columns == db.columns, (du.columns, db.columns)
    res = score_metrics(str(u), str(b), comparison_statistic="mean")
    return {r["metric"]: r["from_baseline"] for r in res.iter_rows(named=True)}


def stage_score() -> None:
    OFF = OUT / "agg_official_base.parquet"
    OLD = E27 / "agg_base.parquet"
    V8, A = E28 / "agg_v8.parquet", E27 / "agg_pred.parquet"
    raw = {tag: {r["metric"]: r["mean"]
                 for r in pl.read_parquet(pq).iter_rows(named=True)}
           for tag, pq in (("off", OFF), ("old", OLD), ("v8", V8), ("a", A))}

    runs = {
        "v8_vs_old": _score(V8, OLD, "v8old"),
        "a_vs_old": _score(A, OLD, "aold"),
        "v8_vs_off": _score(V8, OFF, "v8off"),
        "a_vs_off": _score(A, OFF, "aoff"),
        "old_vs_off": _score(OLD, OFF, "oldoff"),
    }

    print(f"\n{'指标':44s} {'官方基线 raw':>13s} {'退化基线 raw':>13s} "
          f"{'V8 raw':>10s} {'A raw':>10s}")
    print("-" * 96)
    for k in SCORED:
        print(f"{k:44s} {raw['off'].get(k, float('nan')):13.6f} "
              f"{raw['old'].get(k, float('nan')):13.6f} "
              f"{raw['v8'].get(k, float('nan')):10.6f} {raw['a'].get(k, float('nan')):10.6f}")

    print(f"\n{'指标':44s} {'A/old':>9s} {'V8/old':>9s} {'A/off':>9s} {'V8/off':>9s} "
          f"{'old/off':>9s}")
    print("-" * 96)
    for k in SCORED + ("avg_score",):
        print(f"{k:44s} {runs['a_vs_old'].get(k, float('nan')):9.4f} "
              f"{runs['v8_vs_old'].get(k, float('nan')):9.4f} "
              f"{runs['a_vs_off'].get(k, float('nan')):9.4f} "
              f"{runs['v8_vs_off'].get(k, float('nan')):9.4f} "
              f"{runs['old_vs_off'].get(k, float('nan')):9.4f}")

    v8o, ao = runs["v8_vs_off"]["avg_score"], runs["a_vs_off"]["avg_score"]
    v8d, ad_ = runs["v8_vs_old"]["avg_score"], runs["a_vs_old"]["avg_score"]
    print(f"\n三步链在退化分母下 {ad_:.4f} -> {v8d:.4f}  ({(v8d/ad_-1)*100:+.1f}%)")
    print(f"三步链在官方分母下 {ao:.4f} -> {v8o:.4f}  "
          f"({(v8o/ao-1)*100:+.1f}%)" if ao != 0 else "")
    print(f"领先者 0.1899：官方分母下 V8 {'超过' if v8o > 0.1899 else '未达'} "
          f"(差 {v8o-0.1899:+.4f})")
    (OUT / "score_summary.json").write_text(
        json.dumps({"raw": raw, "from_baseline": runs}, indent=2, sort_keys=True))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    {"baseline": stage_baseline, "score": stage_score}[stage]()
