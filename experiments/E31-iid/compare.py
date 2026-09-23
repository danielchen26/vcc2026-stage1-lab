"""V8 vs V9 并排对比表 + 「默认路径未变」的硬证据。

- 左半：E31 自己重跑的 V8（force_mean 默认 True）与 E28 原始 agg_v8.parquet 逐指标对比，
  两者应当逐位相同 —— 这就是「design_cells 默认路径未变」的端到端证明。
- 右半：V9（force_mean=False）的 from_baseline 与 V8 的差。
分母始终是 E27/agg_base.parquet，comparison_statistic="mean"，与 V8 的 0.2025 同尺度。
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
E27 = ROOT / "experiments" / "E27-six-metrics" / "out"
E28 = ROOT / "experiments" / "E28-pds" / "out"
OUT = Path(__file__).resolve().parent / "out"
SCORED = ("pds_cosine", "expr_mse_unbiased_capped_norm", "de_wilcoxon_direction_reach_raw",
          "de_wilcoxon_lfc_nmae", "de_wilcoxon_direction_fidelity_yield_raw",
          "de_wilcoxon_sig_jaccard")


def wide(pq: Path, csv: Path):
    d = pl.read_parquet(pq)
    w = {"statistic": ["mean"]}
    for r in d.iter_rows(named=True):
        w[r["metric"]] = [r["mean"]]
    pl.DataFrame(w).write_csv(csv)
    return csv


def fb(pq: Path, tag: str) -> tuple[dict, dict]:
    from cell_eval2.score import score_metrics
    res = score_metrics(str(wide(pq, OUT / f"_w_{tag}.csv")),
                        str(wide(E27 / "agg_base.parquet", OUT / "agg_base_wide.csv")),
                        comparison_statistic="mean")
    raw = {r["metric"]: r["mean"] for r in pl.read_parquet(pq).iter_rows(named=True)}
    return raw, {r["metric"]: r["from_baseline"] for r in res.iter_rows(named=True)}


def main() -> None:
    r8, f8 = fb(OUT / "agg_v8.parquet", "v8")
    r9, f9 = fb(OUT / "agg_v9.parquet", "v9")
    r8o, f8o = fb(E28 / "agg_v8.parquet", "v8orig")

    print("=== 默认路径未变：E31 重跑的 V8 vs E28 原始 V8（原始指标值）===")
    worst = 0.0
    for k in SCORED:
        d = abs(r8[k] - r8o[k])
        worst = max(worst, d)
        print(f"{k:42s} {r8[k]:12.8f} {r8o[k]:12.8f}  |diff| {d:.2e}")
    print(f"{'avg_score(from_baseline)':42s} {f8['avg_score']:12.8f} {f8o['avg_score']:12.8f}"
          f"  |diff| {abs(f8['avg_score'] - f8o['avg_score']):.2e}")
    print(f"最大逐指标偏差 {worst:.3e}\n")

    print("=== V8 vs V9 ===")
    print(f"{'指标':42s} {'V8 raw':>10s} {'V9 raw':>10s} {'V8 f_b':>9s} {'V9 f_b':>9s} {'Δ f_b':>9s}")
    print("-" * 96)
    for k in SCORED:
        print(f"{k:42s} {r8[k]:10.4f} {r9[k]:10.4f} {f8[k]:9.4f} {f9[k]:9.4f} "
              f"{f9[k] - f8[k]:+9.4f}")
    print("-" * 96)
    print(f"{'avg_score':42s} {'':10s} {'':10s} {f8['avg_score']:9.4f} "
          f"{f9['avg_score']:9.4f} {f9['avg_score'] - f8['avg_score']:+9.4f}")
    print(f"\n领先者 0.1899：V8 {f8['avg_score'] - 0.1899:+.4f}，V9 {f9['avg_score'] - 0.1899:+.4f}")


if __name__ == "__main__":
    main()
