"""对 pred_v6 跑官方六指标，并与 E27 的 variant A / 退化基线对比。"""
from __future__ import annotations
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
    d = pl.read_parquet(pq); w = {"statistic": ["mean"]}
    for r in d.iter_rows(named=True):
        w[r["metric"]] = [r["mean"]]
    pl.DataFrame(w).write_csv(csv); return csv

def main():
    from cell_eval2 import EvalConfig, compute_metrics, aggregate_metrics
    from cell_eval2.score import score_metrics
    t0 = time.time()
    pq = OUT / "agg_v6.parquet"
    if not pq.exists():
        cfg = EvalConfig.from_preset("vcc2026")
        cfg = replace(cfg, pert_col="target_gene", device="cpu")
        cfg = replace(cfg, de=replace(cfg.de, backend="scanpy"))
        df = compute_metrics(str(OUT / "pred_v6.h5ad"), str(E27 / "real.h5ad"), config=cfg)
        aggregate_metrics(df).write_parquet(pq)
        print(f"compute_metrics 完成 {time.time()-t0:.0f}s")
    else:
        print("已有 agg_v6.parquet，跳过计算")

    res = score_metrics(str(wide(pq, OUT / "agg_v6_wide.csv")),
                        str(wide(E27 / "agg_base.parquet", OUT / "agg_base_wide.csv")),
                        comparison_statistic="mean")
    old = score_metrics(str(wide(E27 / "agg_pred.parquet", OUT / "agg_a_wide.csv")),
                        str(OUT / "agg_base_wide.csv"), comparison_statistic="mean")
    fa = {r["metric"]: r["from_baseline"] for r in old.iter_rows(named=True)}
    fb = {r["metric"]: r["from_baseline"] for r in res.iter_rows(named=True)}
    raw = {r["metric"]: r["mean"] for r in pl.read_parquet(pq).iter_rows(named=True)}
    print(f"\n{'指标':42s} {'V6 原始':>10s} {'A from_b':>9s} {'V6 from_b':>10s} {'变化':>9s}")
    print("-" * 88)
    for k in SCORED:
        print(f"{k:42s} {raw.get(k, float('nan')):10.4f} {fa.get(k, 0):9.4f} "
              f"{fb.get(k, 0):10.4f} {fb.get(k,0)-fa.get(k,0):+9.4f}")
    print("-" * 88)
    print(f"{'avg_score':42s} {'':10s} {fa['avg_score']:9.4f} {fb['avg_score']:10.4f} "
          f"{fb['avg_score']-fa['avg_score']:+9.4f}")
    print(f"\n领先者 0.1899   V6 {'超过 ✅' if fb['avg_score'] > 0.1899 else '未达'}"
          f"  (差 {fb['avg_score']-0.1899:+.4f})")
    print(f"耗时 {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
