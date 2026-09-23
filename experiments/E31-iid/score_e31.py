"""E31 打分器：逐字照 experiments/E28-pds/score_v8.py 的 scorer 配置，只把 tag 参数化，
并把 `cell_eval2.metrics.delta` 的 INFO/WARNING 全部落盘 —— #348 的 budget/claim/ratio
是本实验的**主要观测量**，ratio >= 0.7 时不再有 WARNING，只剩那条 INFO，必须抓住。

分母与 V8 完全相同：experiments/E27-six-metrics/out/agg_base.parquet（退化对照均值基线），
comparison_statistic="mean"，故 avg_score 与 V8 的 0.2025 同尺度、可直接相减。

用法： python score_e31.py v8 | v9
"""
from __future__ import annotations

import logging
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


def main() -> None:
    tag = sys.argv[1]
    log = OUT / f"log_{tag}.txt"
    fh = logging.FileHandler(log, mode="w")
    fh.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(fh)
    root.addHandler(sh)
    logging.getLogger("cell_eval2").setLevel(logging.INFO)

    from cell_eval2 import EvalConfig, aggregate_metrics, compute_metrics
    from cell_eval2.score import score_metrics

    t0 = time.time()
    pq = OUT / f"agg_{tag}.parquet"
    if not pq.exists():
        cfg = EvalConfig.from_preset("vcc2026")
        cfg = replace(cfg, pert_col="target_gene", device="cpu")
        cfg = replace(cfg, de=replace(cfg.de, backend="scanpy"))
        df = compute_metrics(str(OUT / f"pred_{tag}.h5ad"), str(E27 / "real.h5ad"), config=cfg)
        aggregate_metrics(df).write_parquet(pq)
        print(f"compute_metrics 完成 {time.time() - t0:.0f}s")
    else:
        print(f"已有 agg_{tag}.parquet，跳过计算")

    res = score_metrics(str(wide(pq, OUT / f"agg_{tag}_wide.csv")),
                        str(wide(E27 / "agg_base.parquet", OUT / "agg_base_wide.csv")),
                        comparison_statistic="mean")
    fb = {r["metric"]: r["from_baseline"] for r in res.iter_rows(named=True)}
    raw = {r["metric"]: r["mean"] for r in pl.read_parquet(pq).iter_rows(named=True)}
    print(f"\n{'指标':42s} {'原始':>10s} {'from_baseline':>14s}")
    print("-" * 70)
    for k in SCORED:
        print(f"{k:42s} {raw.get(k, float('nan')):10.4f} {fb.get(k, 0):14.4f}")
    print("-" * 70)
    print(f"{'avg_score':42s} {'':10s} {fb['avg_score']:14.4f}")
    print(f"V8 基准 0.2025  差 {fb['avg_score'] - 0.2025:+.4f}；领先者 0.1899  "
          f"差 {fb['avg_score'] - 0.1899:+.4f}")
    print(f"耗时 {time.time() - t0:.0f}s")

    # AUTO_CLEAN：聚合已落盘（几 KB），292 MB 的提交文件立刻删（磁盘是硬约束）。
    sub = OUT / f"pred_{tag}.h5ad"
    if sub.exists():
        mb = sub.stat().st_size / 1e6
        sub.unlink()
        print(f"已清理 {sub.name}（{mb:.0f} MB）；结果保留在 agg_{tag}.parquet")
    print(f"日志 {log}")


if __name__ == "__main__":
    main()
