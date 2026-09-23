"""对 pred_v{tag} 跑官方六指标，与 E27 退化基线对比。

逐字照 E28-pds/score_v8.py：EvalConfig 三步、comparison_statistic="mean"、
分母 E27/agg_base.parquet、评分成功后立刻删除提交 .h5ad。
唯一差别是标签与输出目录参数化。

用法：/Users/chetianc/vcc2026/.venv/bin/python score_lambda.py v10
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
E27 = ROOT / "experiments" / "E27-six-metrics" / "out"
OUT = Path(__file__).resolve().parent / "out"
SCORED = ("de_wilcoxon_sig_jaccard", "de_wilcoxon_lfc_nmae",
          "de_wilcoxon_direction_fidelity_yield_raw", "de_wilcoxon_direction_reach_raw",
          "pds_cosine", "expr_mse_unbiased_capped_norm")


def wide(pq: Path, csv: Path):
    d = pl.read_parquet(pq)
    w = {"statistic": ["mean"]}
    for r in d.iter_rows(named=True):
        w[r["metric"]] = [r["mean"]]
    pl.DataFrame(w).write_csv(csv)
    return csv


def main(tag: str):
    from cell_eval2 import EvalConfig, aggregate_metrics, compute_metrics
    from cell_eval2.score import score_metrics
    t0 = time.time()
    pq = OUT / f"agg_{tag}.parquet"
    if not pq.exists():
        cfg = EvalConfig.from_preset("vcc2026")
        cfg = replace(cfg, pert_col="target_gene", device="cpu")
        cfg = replace(cfg, de=replace(cfg.de, backend="scanpy"))
        df = compute_metrics(str(OUT / f"pred_{tag}.h5ad"), str(E27 / "real.h5ad"),
                             config=cfg)
        aggregate_metrics(df).write_parquet(pq)
        print(f"compute_metrics 完成 {time.time()-t0:.0f}s")
    else:
        print(f"已有 agg_{tag}.parquet，跳过计算")

    res = score_metrics(str(wide(pq, OUT / f"agg_{tag}_wide.csv")),
                        str(wide(E27 / "agg_base.parquet", OUT / "agg_base_wide.csv")),
                        comparison_statistic="mean")
    old = score_metrics(str(wide(E27 / "agg_pred.parquet", OUT / "agg_a_wide.csv")),
                        str(OUT / "agg_base_wide.csv"), comparison_statistic="mean")
    fa = {r["metric"]: r["from_baseline"] for r in old.iter_rows(named=True)}
    fb = {r["metric"]: r["from_baseline"] for r in res.iter_rows(named=True)}
    raw = {r["metric"]: r["mean"] for r in pl.read_parquet(pq).iter_rows(named=True)}
    print(f"\n{'指标':42s} {tag+' 原始':>10s} {'A from_b':>9s} {tag+' from_b':>10s} {'变化':>9s}")
    print("-" * 88)
    for k in SCORED:
        print(f"{k:42s} {raw.get(k, float('nan')):10.4f} {fa.get(k, 0):9.4f} "
              f"{fb.get(k, 0):10.4f} {fb.get(k,0)-fa.get(k,0):+9.4f}")
    print("-" * 88)
    print(f"{'avg_score':42s} {'':10s} {fa['avg_score']:9.4f} {fb['avg_score']:10.4f} "
          f"{fb['avg_score']-fa['avg_score']:+9.4f}")
    print(f"\nV8 基准 0.2025   {tag} {'超过' if fb['avg_score'] > 0.2025 else '未达'}"
          f"  (差 {fb['avg_score']-0.2025:+.4f})")
    print(f"耗时 {time.time()-t0:.0f}s")

    # AUTO_CLEAN：聚合已落盘为 agg_*.parquet（几 KB），提交文件本身不再需要。
    # 磁盘只剩约 22 GB 而每份提交 292 MB（gzip），故评分成功后立刻删。
    sub = OUT / f"pred_{tag}.h5ad"
    if sub.exists():
        mb = sub.stat().st_size / 1e6
        sub.unlink()
        print(f"已清理 {sub.name}（{mb:.0f} MB）；结果保留在 agg_*.parquet")


if __name__ == "__main__":
    main(sys.argv[1])
