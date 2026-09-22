"""把 nmae 的 raw 换算成官方两端刻度与 avg_score —— 用 OfficialBaseline 的 helper，
不自己写算术（Main 明令：两份实现会漂移）。

对每个假想臂只替换 de_wilcoxon_lfc_nmae 一个 raw，其余五个成员取 V8 实测值。
⚠️ 这是**成员级反事实**，不是对该臂 avg_score 的预测：V11/V12 会把 lfc_all 通道打开，
从而同时移动 expr_mse / direction_fidelity / direction_reach（见 transfer_probe 的
SSE 代理：GW-only 是「预测 0」的 1.78 倍）。故下表回答的是「若 nmae 走到该 raw，
且其余成员不动，值多少钱」，用来给覆盖率定价，不用来宣称收益。

跑法：~/vcc2026/.venv/bin/python experiments/E33-source-union/currency.py <raw> ...
"""
from __future__ import annotations
import sys
from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "E29-official-baseline"))
import anchor_local as A  # noqa: E402

M = "de_wilcoxon_lfc_nmae"
# 全部取自 calib_probe.py 的实测/离线算术（7 个进入均值的扰动，MAT2A 被
# min_gate_size=10 省略）。离线模型对 V8 的复现误差 +0.000675，见 calib_probe 末行。
ARMS = {
    "V8 实测（对照：应复现 0.1215 / 0.0848）": None,
    "V11 预测：lfc_all = K562GW only": 1.0234,
    "V12 预测：lfc_all = K562GW + Essential 填充": 1.0234,
    "天花板 A：K562GW 覆盖处给真值": 0.5285,
    "天花板 B：并集覆盖处给真值": 0.5285,
    "排行榜第一名的 nmae raw（0.892，供定位）": 0.892,
}


def main() -> None:
    v8 = {r["metric"]: r["mean"] for r in
          pl.read_parquet(ROOT / "experiments" / "E28-pds" / "out"
                          / "agg_v8.parquet").iter_rows(named=True)}
    print(f"{'臂':44s} {'nmae raw':>9s} {'nmae b_lo/r_lo':>15s} {'nmae b_hi/r_hi':>15s}"
          f" {'avg b_lo/r_lo':>14s} {'avg b_hi/r_hi':>14s}")
    print("-" * 116)
    base = None
    for name, u in ARMS.items():
        raw = dict(v8)
        if u is not None:
            raw[M] = u
        res = A.official_two_ended_from_raw(raw)
        lo, hi = res["b_lo/r_lo"], res["b_hi/r_hi"]
        if base is None:
            base = (lo["avg_score"]["score"], hi["avg_score"]["score"])
        print(f"{name:44s} {raw[M]:9.4f} {lo[M]['score']:15.4f} {hi[M]['score']:15.4f}"
              f" {lo['avg_score']['score']:14.4f} {hi['avg_score']['score']:14.4f}"
              f"  (Δavg {lo['avg_score']['score']-base[0]:+.4f} / "
              f"{hi['avg_score']['score']-base[1]:+.4f})")
    print("-" * 116)
    r = A.official_two_ended_from_raw(v8)
    print("V8 逐成员（官方两端刻度）：")
    for m in A.SCORED:
        a, b = r["b_lo/r_lo"][m], r["b_hi/r_hi"][m]
        print(f"  {m:44s} raw {v8[m]:9.4f}  {a['score']:8.4f} / {b['score']:8.4f}"
              f"{'  (degenerate)' if a['degenerate'] else ''}")
    print(f"  {'avg_score':44s} {'':13s}  "
          f"{r['b_lo/r_lo']['avg_score']['score']:8.4f} / "
          f"{r['b_hi/r_hi']['avg_score']['score']:8.4f}")

    # ---- 仓库内部尺度（与 V8 的 0.2025 同尺度：score_metrics(pred, agg_base)，
    #      不传 anchor= / real_bundle=，分母为「对照均值铺满」退化基线）----
    from cell_eval2.score import score_metrics
    OUT = Path(__file__).resolve().parent / "out"
    OUT.mkdir(parents=True, exist_ok=True)
    bw = OUT / "w_base.csv"
    if not bw.exists():
        d = pl.read_parquet(ROOT / "experiments" / "E27-six-metrics" / "out"
                            / "agg_base.parquet")
        w = {"statistic": ["mean"]}
        for x in d.iter_rows(named=True):
            w[x["metric"]] = [x["mean"]]
        pl.DataFrame(w).write_csv(bw)
    print("\n仓库内部尺度（from_baseline，与 V8 的 0.2025 同尺度）：")
    print(f"{'臂':44s} {'nmae raw':>9s} {'nmae from_b':>12s} {'avg_score':>10s} {'Δavg':>8s}")
    print("-" * 88)
    b0 = None
    for name, u in ARMS.items():
        raw = dict(v8)
        if u is not None:
            raw[M] = u
        w = {"statistic": ["mean"]}
        for k, val in raw.items():
            w[k] = [val]
        f = OUT / "_w_arm.csv"
        pl.DataFrame(w).write_csv(f)
        res = score_metrics(str(f), str(bw), comparison_statistic="mean")
        fb = {x["metric"]: x["from_baseline"] for x in res.iter_rows(named=True)}
        if b0 is None:
            b0 = fb["avg_score"]
        print(f"{name:44s} {raw[M]:9.4f} {fb[M]:12.4f} {fb['avg_score']:10.4f} "
              f"{fb['avg_score']-b0:+8.4f}")


if __name__ == "__main__":
    main()
