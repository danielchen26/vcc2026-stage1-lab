"""V14 评分 —— 三列刻度 + SPEC §8e 的两个门（机制门优先于分数门）。

复用 score_v13.py 的 helper（`wide` / `raw_of` / `SCORED`），不复制算术。
与 V13 评分的差别：基线是 **V13**（当前最优）而不是 V8，且判定按 §8e 的两个独立门。

跑法：/Users/chetianc/vcc2026/.venv/bin/python experiments/E34-pds-decouple/score_v14.py
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments" / "E29-official-baseline"))

from score_v13 import SCORED, raw_of, wide  # noqa: E402

E27 = ROOT / "experiments" / "E27-six-metrics" / "out"
E28 = ROOT / "experiments" / "E28-pds" / "out"
OUT = HERE / "out"

# SPEC §8d/§8e —— 全部在 build 之前写死（git: 4601d28）
PRED = {"pds_cosine": 0.8393, "de_wilcoxon_direction_reach_raw": 0.1482}
GATE_MECH_REACH = 0.14      # 机制门（主）：排序不变量救回纯前缀
GATE_SCORE_LO = 0.1578      # 分数门（次）：V13 的 0.1508 + 1 个 pds 量子折算 0.0070
V8_LO, V13_LO, V13_HI = 0.1215, 0.1508, 0.1069
REACH_NOSKILL = 0.0355      # T13 修正后的 reach 零假设
LEADER = 0.1899


def main() -> None:
    import anchor_local as A
    from cell_eval2 import EvalConfig, aggregate_metrics, compute_metrics
    from cell_eval2.score import score_metrics

    t0 = time.time()
    pq = OUT / "agg_v14.parquet"
    if not pq.exists():
        cfg = EvalConfig.from_preset("vcc2026")
        cfg = replace(cfg, pert_col="target_gene", device="cpu")
        cfg = replace(cfg, de=replace(cfg.de, backend="scanpy"))
        df = compute_metrics(str(OUT / "pred_v14.h5ad"), str(E27 / "real.h5ad"), config=cfg)
        aggregate_metrics(df).write_parquet(pq)
        print(f"compute_metrics 完成 {time.time()-t0:.0f}s")
    else:
        print("已有 agg_v14.parquet，跳过计算")

    r14 = raw_of(pq)
    r13 = raw_of(OUT / "agg_v13.parquet")
    r8 = raw_of(E28 / "agg_v8.parquet")
    o14, o13, o8 = (A.official_two_ended_from_raw(r) for r in (r14, r13, r8))
    i14 = {r["metric"]: r["from_baseline"] for r in score_metrics(
        str(wide(pq, OUT / "_w_v14.csv")),
        str(wide(E27 / "agg_base.parquet", OUT / "_w_base.csv")),
        comparison_statistic="mean").iter_rows(named=True)}

    print("\n" + "=" * 118)
    print("V14（压幅）vs V13（原幅）vs V8 —— SPEC §3 强制三列刻度")
    print("=" * 118)
    hdr = (f"{'指标':40s} {'V8 raw':>8s} {'V13 raw':>8s} {'V14 raw':>8s} | "
           f"{'V8 lo':>8s} {'V13 lo':>8s} {'V14 lo':>8s} {'Δ vs13':>8s} | {'V14 hi':>8s}")
    print(hdr); print("-" * len(hdr))
    for k in SCORED:
        a, b, c = (o8["b_lo/r_lo"][k]["score"], o13["b_lo/r_lo"][k]["score"],
                   o14["b_lo/r_lo"][k]["score"])
        print(f"{k:40s} {r8[k]:8.4f} {r13[k]:8.4f} {r14[k]:8.4f} | "
              f"{a:8.4f} {b:8.4f} {c:8.4f} {c-b:+8.4f} | "
              f"{o14['b_hi/r_hi'][k]['score']:8.4f}")
    lo = o14["b_lo/r_lo"]["avg_score"]["score"]
    hi = o14["b_hi/r_hi"]["avg_score"]["score"]
    print("-" * len(hdr))
    print(f"{'avg_score':40s} {'':8s} {'':8s} {'':8s} | {V8_LO:8.4f} {V13_LO:8.4f} "
          f"{lo:8.4f} {lo-V13_LO:+8.4f} | {hi:8.4f}")
    print(f"{'仓库内部（T9：不可外推）':40s} {i14['avg_score']:>8.4f}")
    print(f"\n⚠️ 分子 = 本 8 扰动 / 1 context 的 raw；分母 = 官方 300 扰动 / 3 context 的 b/r。"
          f"\n   可并排看榜首 {LEADER}，但不是同一回事（SPEC §3）。")

    # ================= SPEC §8e =================
    print("\n" + "=" * 118)
    print("SPEC §8e 判定 —— 机制门优先于分数门（门在 build 前定死：git 4601d28）")
    print("=" * 118)
    reach = r14["de_wilcoxon_direction_reach_raw"]
    mech = reach >= GATE_MECH_REACH
    print(f"预测 pds raw {PRED['pds_cosine']:.4f}，实测 {r14['pds_cosine']:.4f}，"
          f"差 {r14['pds_cosine']-PRED['pds_cosine']:+.4f}"
          f"（{round((r14['pds_cosine']-PRED['pds_cosine'])*56)} 个量子）")
    print(f"预测 reach raw {PRED['de_wilcoxon_direction_reach_raw']:.4f}，实测 {reach:.4f}"
          f"   零假设（T13）{REACH_NOSKILL:.4f}   "
          f"{'高于' if reach > REACH_NOSKILL else '**低于**'}零假设")
    print(f"\n机制门（主）reach >= {GATE_MECH_REACH:.2f}：{'✅ 过' if mech else '❌ 未过'}"
          f"   —— V13 是 {r13['de_wilcoxon_direction_reach_raw']:.4f}")
    score_pass = lo >= GATE_SCORE_LO
    print(f"分数门（次）avg_lo >= {GATE_SCORE_LO:.4f}：{'✅ 过' if score_pass else '❌ 未过'}"
          f"   （实测 {lo:.4f}，V13 {V13_LO:.4f}）")

    halts = []
    djac = abs(r14["de_wilcoxon_sig_jaccard"] - r8["de_wilcoxon_sig_jaccard"])
    if djac > 2e-5:
        halts.append(f"停止条件 2：sig_jaccard 相对 V8 变了 {djac:.2e} > 2e-5 —— "
                     f"敞口比 V13 观察到的更大，需单独定位")
    if not mech:
        halts.append("停止条件 1：不变量已在 build 内断言通过，但 reach 仍未恢复 —— "
                     "纯前缀还有第三个来源，SPEC §8a 仍不完整")
    for h in halts:
        print("  ❌ " + h)
    if not halts:
        print("  ✅ 停止条件均未触发")

    verdict = ("GO：V14 取代 V13" if mech and score_pass else
               "机制成立但分数未超 V13（保留 V13 为提交配置，V14 的 reach/mse 头寸留给后续组合）"
               if mech else "机制未成立 —— 回头改 §8a")
    print(f"\n=> **{verdict}**")

    json.dump({"raw": r14,
               "official_lo": {k: o14["b_lo/r_lo"][k]["score"] for k in SCORED},
               "official_hi": {k: o14["b_hi/r_hi"][k]["score"] for k in SCORED},
               "avg_lo": lo, "avg_hi": hi, "internal": i14,
               "mech_gate_pass": mech, "score_gate_pass": score_pass,
               "halts": halts, "verdict": verdict,
               "k_off": 550, "lambda_off": 1.0, "cap_margin": 0.99},
              open(OUT / "score_v14.json", "w"), indent=2, ensure_ascii=False)
    print(f"\n已存 {OUT/'score_v14.json'}   耗时 {time.time()-t0:.0f}s")

    sub = OUT / "pred_v14.h5ad"
    if sub.exists():
        mb = sub.stat().st_size / 1e6
        sub.unlink()
        print(f"已清理 {sub.name}（{mb:.0f} MB）；聚合保留在 agg_v14.parquet")


if __name__ == "__main__":
    main()
