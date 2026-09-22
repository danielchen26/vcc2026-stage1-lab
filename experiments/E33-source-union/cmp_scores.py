"""V8 的 from_baseline（复刻 score_v8.py 的 score_metrics 调用，逐字同参），
但把中间 wide csv 写进本实验目录，不碰 E28-pds/out（兄弟 agent 共用）。

score_metrics 只传 (pred_wide, base_wide, comparison_statistic="mean")，
**不传** anchor= / real_bundle=，故这里的 from_baseline 是仓库内部的
anchor-0/1 尺度、分母为「对照均值铺满」退化基线 —— 与 V8 的 0.2025 同尺度。
"""
from __future__ import annotations
import sys
from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
E27 = ROOT / "experiments" / "E27-six-metrics" / "out"
E28 = ROOT / "experiments" / "E28-pds" / "out"
OUT = Path(__file__).resolve().parent / "out"
SCORED = ("de_wilcoxon_sig_jaccard", "de_wilcoxon_lfc_nmae",
          "de_wilcoxon_direction_fidelity_yield_raw", "de_wilcoxon_direction_reach_raw",
          "pds_cosine", "expr_mse_unbiased_capped_norm")


def wide(pq: Path, csv: Path):
    d = pl.read_parquet(pq); w = {"statistic": ["mean"]}
    for r in d.iter_rows(named=True):
        w[r["metric"]] = [r["mean"]]
    pl.DataFrame(w).write_csv(csv); return csv


def main() -> None:
    from cell_eval2.score import score_metrics
    OUT.mkdir(parents=True, exist_ok=True)
    base = wide(E27 / "agg_base.parquet", OUT / "w_base.csv")
    tags = sys.argv[1:] or ["v8"]
    cols = {}
    for t in tags:
        pq = E28 / f"agg_{t}.parquet" if (E28 / f"agg_{t}.parquet").exists() \
            else OUT / f"agg_{t}.parquet"
        r = score_metrics(str(wide(pq, OUT / f"w_{t}.csv")), str(base),
                          comparison_statistic="mean")
        cols[t] = ({x["metric"]: x["from_baseline"] for x in r.iter_rows(named=True)},
                   {x["metric"]: x["mean"] for x in pl.read_parquet(pq).iter_rows(named=True)})
    hdr = "".join(f"{t+' raw':>12s}{t+' from_b':>14s}" for t in tags)
    print(f"{'指标':42s}{hdr}")
    print("-" * (42 + 26 * len(tags)))
    for k in SCORED:
        line = f"{k:42s}"
        for t in tags:
            fb, raw = cols[t]
            line += f"{raw.get(k, float('nan')):12.4f}{fb.get(k, 0):14.4f}"
        print(line)
    print("-" * (42 + 26 * len(tags)))
    line = f"{'avg_score':42s}"
    for t in tags:
        line += f"{'':12s}{cols[t][0]['avg_score']:14.4f}"
    print(line)


if __name__ == "__main__":
    main()
