"""V9 的官方刻度换算 + i.i.d. 伪影是否真的消失。

为什么单独一个脚本：`score_e31.py` 只报仓库内部 `from_baseline`（分母是 E27 的退化基线），
那正是 [T9](../../docs/06-traps.md#t9) 判为不可外推的那把尺子。提交合法性的决策必须在
**官方刻度**上做，且两端都报。算术全部走 `anchor_local.official_two_ended_from_raw`，
不自写。

判定（写在跑之前）：
  V9 的意义不是提分，是**去掉一个组织者明确在看的可检测伪影**（[F30](../../docs/02-findings.md#f30)）。
  所以门是「代价可接受」，不是「必须变好」：

    - `expr_mse_unbiased_capped` 的跨扰动离散度 / 声称的抽样校正 之比须 >= 0.7
      （V6/V7/V8 实测 0.484，被打分器告警并按 0.484 倍打折发放）。
      这个比值是**告警本身的判据**，所以它过了才算伪影消失。
    - 官方 avg_lo 的下降须落在 [F36](../../docs/02-findings.md#f36) 的棒 ±0.0423 之内 ——
      否则「合法性」是用真实分数换来的，需要明确权衡而不是默认接受。

⚠️ 本脚本只读 agg_*.parquet，不重算指标、不碰 .h5ad。

跑法：/Users/chetianc/vcc2026/.venv/bin/python experiments/E31-iid/official_v9.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
E28 = ROOT / "experiments" / "E28-pds" / "out"
OUT = Path(__file__).resolve().parent / "out"
sys.path.insert(0, str(ROOT / "experiments" / "E29-official-baseline"))

SCORED = ("pds_cosine", "expr_mse_unbiased_capped_norm", "de_wilcoxon_sig_jaccard",
          "de_wilcoxon_lfc_nmae", "de_wilcoxon_direction_fidelity_yield_raw",
          "de_wilcoxon_direction_reach_raw")
BAR = 0.0423          # F36：V8 官方 lo 的 bootstrap sd
V8_LO, V8_HI = 0.1215, 0.0848


def raw_of(pq: Path) -> dict:
    return {r["metric"]: r["mean"] for r in pl.read_parquet(pq).iter_rows(named=True)}


def main() -> None:
    import anchor_local as A

    pq9 = OUT / "agg_v9.parquet"
    assert pq9.exists(), f"缺 {pq9} —— 先跑 build_run.py v9 && score_e31.py v9"
    r9, r8 = raw_of(pq9), raw_of(E28 / "agg_v8.parquet")
    o9, o8 = A.official_two_ended_from_raw(r9), A.official_two_ended_from_raw(r8)

    print("=" * 104)
    print("V9（force_mean=False，诚实抽样）vs V8（force_mean=True，钉死经验均值）")
    print("=" * 104)
    hdr = (f"{'指标':42s} {'V8 raw':>9s} {'V9 raw':>9s} | {'V8 lo':>8s} {'V9 lo':>8s} "
           f"{'Δlo':>8s} | {'V8 hi':>8s} {'V9 hi':>8s}")
    print(hdr); print("-" * len(hdr))
    for k in SCORED:
        a, b = o8["b_lo/r_lo"][k]["score"], o9["b_lo/r_lo"][k]["score"]
        print(f"{k:42s} {r8[k]:9.4f} {r9[k]:9.4f} | {a:8.4f} {b:8.4f} {b-a:+8.4f} | "
              f"{o8['b_hi/r_hi'][k]['score']:8.4f} {o9['b_hi/r_hi'][k]['score']:8.4f}")
    lo9 = o9["b_lo/r_lo"]["avg_score"]["score"]
    hi9 = o9["b_hi/r_hi"]["avg_score"]["score"]
    print("-" * len(hdr))
    print(f"{'avg_score':42s} {'':9s} {'':9s} | {V8_LO:8.4f} {lo9:8.4f} {lo9-V8_LO:+8.4f} | "
          f"{V8_HI:8.4f} {hi9:8.4f}")

    d = lo9 - V8_LO
    print(f"\n代价 {d:+.4f}，F36 的棒 ±{BAR:.4f} ⇒ "
          f"{'落在棒内，合法性基本免费' if abs(d) <= BAR else '**超出棒，合法性是用真实分数换的**'}")
    print("\n⚠️ 还需要看官方打分器有没有继续发 i.i.d. 告警："
          "\n   V6/V7/V8 的离散度/校正之比是 0.484（< 0.7 阈值）。"
          "\n   该比值由打分器在评分时打印，见 out/v9_chain.log —— "
          "本脚本只读聚合，读不到它。")

    json.dump({"raw_v9": r9, "raw_v8": r8,
               "official_lo": {k: o9["b_lo/r_lo"][k]["score"] for k in SCORED},
               "official_hi": {k: o9["b_hi/r_hi"][k]["score"] for k in SCORED},
               "avg_lo": lo9, "avg_hi": hi9, "cost_lo": d, "bar_f36": BAR,
               "within_bar": bool(abs(d) <= BAR)},
              open(OUT / "official_v9.json", "w"), indent=2, ensure_ascii=False)
    print(f"\n已存 {OUT/'official_v9.json'}")


if __name__ == "__main__":
    main()
