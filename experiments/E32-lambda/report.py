"""汇总 lambda 曲线：两种尺度并列输出，供 RESULT.md 直接抄录。

内部尺度：score_metrics(agg_x, agg_base) 的 from_baseline（锚点 0/1，分母是
  E27 的 control-mean-tiled 退化基线）。
官方尺度：OfficialBaseline 的共享 helper
  experiments/E29-official-baseline/anchor_local.official_two_ended，
  两端全报，不取中点。

用法：/Users/chetianc/vcc2026/.venv/bin/python report.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "E29-official-baseline"))
import anchor_local as A  # noqa: E402

from cell_eval2.score import score_metrics  # noqa: E402

E27 = ROOT / "experiments" / "E27-six-metrics" / "out"
E28 = ROOT / "experiments" / "E28-pds" / "out"
HERE = Path(__file__).resolve().parent / "out"
TMP = Path(tempfile.mkdtemp())

# (标签, agg parquet, ordering, K 规则, lambda)
POINTS = [
    ("V5",  E28 / "agg_v5.parquet",  "|beta|", "29/288/G", 1.0),
    ("V6",  E28 / "agg_v6.parquet",  "|beta|", "29/288/G", 0.5),
    ("V7",  E28 / "agg_v7.parquet",  "|beta|", "288 flat", 0.5),
    ("V8",  E28 / "agg_v8.parquet",  "|beta|", "288 flat", 0.7),
    ("V12", HERE / "agg_v12.parquet", "|beta|", "288 flat", 1.0),
]
SHOW = ("de_wilcoxon_direction_reach_raw", "expr_mse_unbiased_capped_norm",
        "de_wilcoxon_lfc_nmae", "pds_cosine", "de_wilcoxon_sig_jaccard",
        "de_wilcoxon_direction_fidelity_yield_raw")


def wide(pq: Path, csv: Path) -> str:
    d = pl.read_parquet(pq)
    w = {"statistic": ["mean"]}
    for r in d.iter_rows(named=True):
        w[r["metric"]] = [r["mean"]]
    pl.DataFrame(w).write_csv(csv)
    return str(csv)


def main() -> None:
    base = wide(E27 / "agg_base.parquet", TMP / "base.csv")
    rows = []
    for tag, pq, order, krule, lam in POINTS:
        if not pq.exists():
            print(f"!! 缺 {pq}，跳过 {tag}")
            continue
        raw = {r["metric"]: r["mean"] for r in pl.read_parquet(pq).iter_rows(named=True)}
        fb = {r["metric"]: r["from_baseline"] for r in
              score_metrics(wide(pq, TMP / f"{tag}.csv"), base,
                            comparison_statistic="mean").iter_rows(named=True)}
        off = A.official_two_ended(str(pq))
        rows.append((tag, order, krule, lam, raw, fb, off))

    print("\n### 三旋钮标注（混淆一目了然）")
    print(f"{'变体':5s} {'ordering':9s} {'K 规则':9s} {'lambda':>6s} "
          f"{'内部 avg':>9s} {'官方 lo':>8s} {'官方 hi':>8s}")
    for tag, order, krule, lam, raw, fb, off in rows:
        print(f"{tag:5s} {order:9s} {krule:9s} {lam:6.2f} {fb['avg_score']:9.4f} "
              f"{off['b_lo/r_lo']['avg_score']['score']:8.4f} "
              f"{off['b_hi/r_hi']['avg_score']['score']:8.4f}")

    print("\n### 原始均值")
    print(f"{'变体':5s} " + " ".join(f"{k.split('_')[-2][:9]:>10s}" for k in SHOW))
    for tag, order, krule, lam, raw, fb, off in rows:
        print(f"{tag:5s} " + " ".join(f"{raw.get(k, float('nan')):10.4f}" for k in SHOW))

    print("\n### 内部 from_baseline（逐指标）")
    for tag, order, krule, lam, raw, fb, off in rows:
        print(f"{tag:5s} " + " ".join(f"{fb.get(k, 0.0):10.4f}" for k in SHOW))

    print("\n### 官方尺度逐指标（lo / hi）")
    for tag, order, krule, lam, raw, fb, off in rows:
        lo, hi = off["b_lo/r_lo"], off["b_hi/r_hi"]
        print(f"{tag:5s} " + " ".join(f"{lo[k]['score']:+.4f}/{hi[k]['score']:+.4f}"
                                     for k in SHOW))


if __name__ == "__main__":
    main()
