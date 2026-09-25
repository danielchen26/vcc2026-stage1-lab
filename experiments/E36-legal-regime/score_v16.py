"""V16 评分 —— 合法 regime 下的四方对比：V8 / V9 / V13 / V16。

四个配置正好是一个 2×2：

                    force_mean=True      force_mean=False（合法）
    lfc_all 关       V8   0.1215          V9   0.0780
    lfc_all 开       V13  0.1508          V16  ?

所以本次评分同时给出两件事：
  ① V16 本身（合法 regime 下开 lfc_all 的分数）；
  ② **交互项** (V16−V9) − (V13−V8) —— `lfc_all` 的收益在合法 regime 下保留了多少。
     若交互 ≈ 0，则 `lfc_all` 与 `force_mean` 可分，V13 的 pds 收益真的能带进合法提交；
     若交互强负，则 V13 的 +0.0293 里有一部分本来就是 F35 那批同号伪影，
     在合法提交里拿不到（[F39](../../docs/02-findings.md#f39) ③ 的怀疑成立）。

判定（build 前写下）：
  - V16 官方 lo > V9 的 0.0780 + 0.0070（1 个 pds 量子折算）⇒ 合法 regime 下应采用 V16；
  - 否则合法提交用 V9（即不开 lfc_all）。
  - 交互项 |·| < 0.0070 ⇒ 两个旋钮可分，记为可加性成立。

⚠️ 三列刻度全报（[T9](../../docs/06-traps.md#t9)）。另外必须从 compute_metrics 的日志里
读出抽样校正折扣倍数（issue #348）—— V16 必须 ≥ 0.7 才算合法，这一条聚合里读不到。

跑法：/Users/chetianc/vcc2026/.venv/bin/python experiments/E36-legal-regime/score_v16.py
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E27 = ROOT / "experiments" / "E27-six-metrics" / "out"
E28 = ROOT / "experiments" / "E28-pds" / "out"
E31 = ROOT / "experiments" / "E31-iid" / "out"
E34 = ROOT / "experiments" / "E34-pds-decouple" / "out"
OUT = HERE / "out"
sys.path.insert(0, str(ROOT / "experiments" / "E34-pds-decouple"))
sys.path.insert(0, str(ROOT / "experiments" / "E29-official-baseline"))
from score_v13 import SCORED, raw_of, wide  # noqa: E402

V8_LO, V9_LO, V13_LO = 0.1215, 0.0780, 0.1508
QUANTUM = 0.0070          # 1 个 pds 量子（1/56 raw）折算成 avg
LEADER = 0.1899


def main() -> None:
    import anchor_local as A
    from cell_eval2 import EvalConfig, aggregate_metrics, compute_metrics
    from cell_eval2.score import score_metrics

    t0 = time.time()
    pq = OUT / "agg_v16.parquet"
    if not pq.exists():
        cfg = EvalConfig.from_preset("vcc2026")
        cfg = replace(cfg, pert_col="target_gene", device="cpu")
        cfg = replace(cfg, de=replace(cfg.de, backend="scanpy"))
        df = compute_metrics(str(OUT / "pred_v16.h5ad"), str(E27 / "real.h5ad"), config=cfg)
        aggregate_metrics(df).write_parquet(pq)
        print(f"compute_metrics 完成 {time.time()-t0:.0f}s")
    else:
        print("已有 agg_v16.parquet，跳过计算")

    r = {"V8": raw_of(E28 / "agg_v8.parquet"), "V9": raw_of(E31 / "agg_v9.parquet"),
         "V13": raw_of(E34 / "agg_v13.parquet"), "V16": raw_of(pq)}
    o = {k: A.official_two_ended_from_raw(v) for k, v in r.items()}
    i16 = {x["metric"]: x["from_baseline"] for x in score_metrics(
        str(wide(pq, OUT / "_w_v16.csv")),
        str(wide(E27 / "agg_base.parquet", OUT / "_w_base.csv")),
        comparison_statistic="mean").iter_rows(named=True)}

    print("\n" + "=" * 118)
    print("2×2：lfc_all × force_mean —— 官方 b_lo/r_lo 刻度")
    print("=" * 118)
    hdr = (f"{'指标':40s}" + "".join(f"{k+' raw':>10s}" for k in r)
           + " |" + "".join(f"{k+' lo':>9s}" for k in r))
    print(hdr); print("-" * len(hdr))
    for m in SCORED:
        print(f"{m:40s}" + "".join(f"{r[k][m]:10.4f}" for k in r) + " |"
              + "".join(f"{o[k]['b_lo/r_lo'][m]['score']:9.4f}" for k in r))
    print("-" * len(hdr))
    avg = {k: o[k]["b_lo/r_lo"]["avg_score"]["score"] for k in r}
    print(f"{'avg_score (lo)':40s}" + " " * (10 * len(r)) + " |"
          + "".join(f"{avg[k]:9.4f}" for k in r))
    print(f"{'avg_score (hi)':40s}" + " " * (10 * len(r)) + " |"
          + "".join(f"{o[k]['b_hi/r_hi']['avg_score']['score']:9.4f}" for k in r))
    print(f"{'仓库内部（不可外推）':40s}" + " " * (10 * len(r))
          + f" | V16 {i16['avg_score']:.4f}")

    print("\n" + "=" * 118)
    print("① V16 vs 合法基线 V9    ② lfc_all 的收益在合法 regime 下保留了多少")
    print("=" * 118)
    d_legal = avg["V16"] - avg["V9"]
    d_illegal = avg["V13"] - avg["V8"]
    inter = d_legal - d_illegal
    print(f"  lfc_all 在 force_mean=True  下的收益 (V13−V8)  = {d_illegal:+.4f}")
    print(f"  lfc_all 在 force_mean=False 下的收益 (V16−V9)  = {d_legal:+.4f}")
    print(f"  交互项 = {inter:+.4f}   "
          f"{'可加（|交互| < 1 量子）' if abs(inter) < QUANTUM else '**不可加**'}")
    print(f"  保留比例 = {d_legal/d_illegal:.1%}" if d_illegal else "")

    use_v16 = avg["V16"] > V9_LO + QUANTUM
    print(f"\n判定（门在 build 前写下）：V16 lo {avg['V16']:.4f} vs V9 + 1 量子 "
          f"{V9_LO + QUANTUM:.4f} ⇒ **合法提交用 {'V16' if use_v16 else 'V9'}**")
    print(f"对榜首 {LEADER}：差 {avg['V16'] - LEADER:+.4f}")
    print("\n⚠️ 合法性本身要看 compute_metrics 日志里的抽样校正折扣倍数（issue #348），"
          "\n   须 ≥ 0.7。V9 是 0.952×，V8/V13/V14 是 0.484×。本脚本读不到它。")

    json.dump({"raw": r, "avg_lo": avg,
               "avg_hi": {k: o[k]["b_hi/r_hi"]["avg_score"]["score"] for k in r},
               "internal_v16": i16["avg_score"],
               "gain_illegal": d_illegal, "gain_legal": d_legal, "interaction": inter,
               "additive": bool(abs(inter) < QUANTUM), "use_v16": bool(use_v16)},
              open(OUT / "score_v16.json", "w"), indent=2, ensure_ascii=False)
    print(f"\n已存 {OUT/'score_v16.json'}   耗时 {time.time()-t0:.0f}s")

    sub = OUT / "pred_v16.h5ad"
    if sub.exists():
        mb = sub.stat().st_size / 1e6
        sub.unlink()
        print(f"已清理 {sub.name}（{mb:.0f} MB）")


if __name__ == "__main__":
    main()
