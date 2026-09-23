"""V13 评分 —— 按 SPEC §3 强制三列刻度，按 §5 自动判定。

与 E32-lambda/score_lambda.py 的差别（后者只报仓库内部刻度，那正是 T9 的坑）：
本脚本三列都报，且把 SPEC §5 的停止条件写成代码里的断言/告警，不靠人眼。

跑法：/Users/chetianc/vcc2026/.venv/bin/python experiments/E34-pds-decouple/score_v13.py
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
E27 = ROOT / "experiments" / "E27-six-metrics" / "out"
E28 = ROOT / "experiments" / "E28-pds" / "out"
OUT = Path(__file__).resolve().parent / "out"
sys.path.insert(0, str(ROOT / "experiments" / "E29-official-baseline"))

SCORED = ("pds_cosine", "expr_mse_unbiased_capped_norm", "de_wilcoxon_sig_jaccard",
          "de_wilcoxon_lfc_nmae", "de_wilcoxon_direction_fidelity_yield_raw",
          "de_wilcoxon_direction_reach_raw")

# SPEC §4 的预测与 §5 的门，全部在 build 之前写下（git 可证）
PRED = {"pds_cosine": 0.8929}
GATE_GO = 0.1394          # = V8 的 0.1215 + 1 个 pds 量子 0.0179
V8_LO, V8_HI, V8_INT = 0.1215, 0.0848, 0.2025
LEADER = 0.1899


def wide(pq: Path, csv: Path) -> Path:
    d = pl.read_parquet(pq)
    w = {"statistic": ["mean"]}
    for r in d.iter_rows(named=True):
        w[r["metric"]] = [r["mean"]]
    pl.DataFrame(w).write_csv(csv)
    return csv


def raw_of(pq: Path) -> dict:
    return {r["metric"]: r["mean"] for r in pl.read_parquet(pq).iter_rows(named=True)}


def main() -> None:
    import anchor_local as A
    from cell_eval2 import EvalConfig, aggregate_metrics, compute_metrics
    from cell_eval2.score import score_metrics

    t0 = time.time()
    pq = OUT / "agg_v13.parquet"
    if not pq.exists():
        cfg = EvalConfig.from_preset("vcc2026")
        cfg = replace(cfg, pert_col="target_gene", device="cpu")
        cfg = replace(cfg, de=replace(cfg.de, backend="scanpy"))
        df = compute_metrics(str(OUT / "pred_v13.h5ad"), str(E27 / "real.h5ad"), config=cfg)
        aggregate_metrics(df).write_parquet(pq)
        print(f"compute_metrics 完成 {time.time()-t0:.0f}s")
    else:
        print("已有 agg_v13.parquet，跳过计算")

    r13, r8 = raw_of(pq), raw_of(E28 / "agg_v8.parquet")

    # ---- 列 1/2：官方两端刻度（唯一可与榜首并排看的）----
    o13, o8 = A.official_two_ended_from_raw(r13), A.official_two_ended_from_raw(r8)
    # ---- 列 3：仓库内部刻度（与 V5-V8 的历史数字对齐用）----
    bw = wide(E27 / "agg_base.parquet", OUT / "_w_base.csv")
    i13 = {r["metric"]: r["from_baseline"] for r in score_metrics(
        str(wide(pq, OUT / "_w_v13.csv")), str(bw), comparison_statistic="mean"
    ).iter_rows(named=True)}

    print("\n" + "=" * 110)
    print("V13 vs V8 —— SPEC §3 强制的三列刻度")
    print("=" * 110)
    hdr = (f"{'指标':42s} {'V8 raw':>9s} {'V13 raw':>9s} | {'V8 lo':>8s} {'V13 lo':>8s} "
           f"{'Δlo':>8s} | {'V8 hi':>8s} {'V13 hi':>8s} | {'V13 内部':>9s}")
    print(hdr); print("-" * len(hdr))
    for k in SCORED:
        a, b = o8["b_lo/r_lo"][k]["score"], o13["b_lo/r_lo"][k]["score"]
        print(f"{k:42s} {r8[k]:9.4f} {r13[k]:9.4f} | {a:8.4f} {b:8.4f} {b-a:+8.4f} | "
              f"{o8['b_hi/r_hi'][k]['score']:8.4f} {o13['b_hi/r_hi'][k]['score']:8.4f} | "
              f"{i13.get(k, float('nan')):9.4f}")
    lo = o13["b_lo/r_lo"]["avg_score"]["score"]
    hi = o13["b_hi/r_hi"]["avg_score"]["score"]
    print("-" * len(hdr))
    print(f"{'avg_score':42s} {'':9s} {'':9s} | {V8_LO:8.4f} {lo:8.4f} {lo-V8_LO:+8.4f} | "
          f"{V8_HI:8.4f} {hi:8.4f} | {i13['avg_score']:9.4f}")
    print(f"\n⚠️ 分子是本 8 扰动 / 1 context 的 raw，分母是官方 300 扰动 / 3 context 的 b/r。"
          f"\n   可以并排看榜首 {LEADER}，但**不是同一回事**（SPEC §3）。")

    # ================= SPEC §5 判定 =================
    print("\n" + "=" * 110)
    print("SPEC §5 判定（门在 build 前定死，事后不许挪）")
    print("=" * 110)
    halts = []
    if r13["pds_cosine"] <= r8["pds_cosine"]:
        halts.append(f"停止条件 1 触发：pds raw {r13['pds_cosine']:.4f} "
                     f"<= V8 的 {r8['pds_cosine']:.4f} —— 本地闭式探针不迁移")
    for k in ("de_wilcoxon_sig_jaccard", "de_wilcoxon_direction_reach_raw"):
        if r13[k] != r8[k]:
            halts.append(f"停止条件 2 触发：{k} 变了（{r8[k]:.6f} -> {r13[k]:.6f}）—— "
                         f"lfc_all 影响了显著性通道，decoder 的两通道分离是假的。"
                         f"**这比 pds 输掉严重**")
    print(f"预测 pds raw {PRED['pds_cosine']:.4f}，实测 {r13['pds_cosine']:.4f}，"
          f"差 {r13['pds_cosine']-PRED['pds_cosine']:+.4f}"
          f"（{round((r13['pds_cosine']-PRED['pds_cosine'])*56)} 个量子）")
    for h in halts:
        print("  ❌ " + h)
    if not halts:
        print("  ✅ 两条停止条件均未触发")

    verdict = ("GO" if lo >= GATE_GO else
               "不确定（1 个量子内，不构成证据）" if lo > V8_LO else "NO-GO")
    print(f"\navg_lo = {lo:.4f}   门 GO >= {GATE_GO:.4f}   V8 = {V8_LO:.4f}"
          f"   =>  **{verdict}**")

    json.dump({"raw": r13, "official_lo": {k: o13["b_lo/r_lo"][k]["score"] for k in SCORED},
               "official_hi": {k: o13["b_hi/r_hi"][k]["score"] for k in SCORED},
               "avg_lo": lo, "avg_hi": hi, "internal": i13,
               "halts": halts, "verdict": verdict,
               "k_off": 550, "lambda_off": 1.0},
              open(OUT / "score_v13.json", "w"), indent=2, ensure_ascii=False)
    print(f"\n已存 {OUT/'score_v13.json'}   耗时 {time.time()-t0:.0f}s")

    sub = OUT / "pred_v13.h5ad"
    if sub.exists():
        mb = sub.stat().st_size / 1e6
        sub.unlink()
        print(f"已清理 {sub.name}（{mb:.0f} MB）；聚合保留在 agg_v13.parquet")


if __name__ == "__main__":
    main()
