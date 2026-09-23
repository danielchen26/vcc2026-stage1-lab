"""Per-perturbation metrics for V8's 8-panel, then the sampling error of its avg_score.

Design (C1), after the 8-of-24 design was abandoned:
  The quantity of interest is the sampling error of a MEAN over 8 perturbations. That is
  estimable from the 8 per-perturbation rows themselves by bootstrap/jackknife; it does
  NOT require a larger panel. Resampling 8-of-24 would have estimated something else and
  estimated it worse, because `np.linspace(0, 40, N).round()` makes the N=8 and N=24
  panels DIFFERENT samples rather than nested ones (measured: 4/8 overlap, see
  panel_check.py) -- an arm that moves the thing it is supposed to hold constant.

  The baseline b is held FIXED at E27's published agg_base.parquet and is never resampled.
  Justification: on the real leaderboard b and r are published constants, not
  per-submission quantities, so resampling the baseline would model a source of variation
  that does not exist in the competition.

Two scales, official first:
  official   (u - b)/(r - b) via experiments/E29-official-baseline/anchor_local.py, the
             shared helper, at BOTH ends of the official b/r ranges. Never a midpoint.
  repo       score_metrics(agg_u, agg_base) `from_baseline`, i.e. the anchor-0/1 column
             this repo scored every variant on. V8 = 0.2025 here.

Aggregation is delegated to cell_eval2.aggregate_metrics per resample and never
reimplemented, because `expr_mse_unbiased_capped_norm` aggregates as `ratio_of_sums`
(measured in agg_v8.parquet), not as a mean: a np.nanmean over its per-perturbation rows
would be silently wrong.

pds_cosine is EXCLUDED from the clean bar. It is a panel-relative RANK metric:
metrics/discrimination.py:215 sets D = n - 1 over the panel's non-control perturbations
(rank_denominator="n-1") and :248 drops EVERY panel target gene from both operands
(exclusion_scope="panel"). Both depend on panel size and membership, so resampling
perturbs the effective panel and a pds row does not transfer. Reported both ways, labelled.

Scorer config is reproduced VERBATIM from experiments/E28-pds/score_v8.py; score_metrics is
called with comparison_statistic="mean" and NO anchor=/real_bundle=, so the `repo` numbers
sit on the same scale as V8's 0.2025. Threading is untouched.

Stages:
  metrics <tag>   compute_metrics for one tag against E27's real.h5ad, persist the
                  UN-AGGREGATED per-perturbation frame, aggregate, unlink the .h5ad.
  analyze         bootstrap + jackknife over the per-perturbation rows.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
E27 = ROOT / "experiments" / "E27-six-metrics" / "out"
E28 = ROOT / "experiments" / "E28-pds" / "out"
OUT = Path(__file__).resolve().parent / "out"
sys.path.insert(0, str(ROOT / "experiments" / "E29-official-baseline"))

SCORED = ("de_wilcoxon_sig_jaccard", "de_wilcoxon_lfc_nmae",
          "de_wilcoxon_direction_fidelity_yield_raw",
          "de_wilcoxon_direction_reach_raw", "pds_cosine",
          "expr_mse_unbiased_capped_norm")
PANEL_RELATIVE = ("pds_cosine",)
CLEAN = tuple(m for m in SCORED if m not in PANEL_RELATIVE)
N_BOOT, BOOT_SEED = 2000, 20260922
V8_REPO_AVG = 0.2025          # measured, experiments/E28-pds score_v8.py
V8_OFFICIAL = {"b_lo/r_lo": 0.1215, "b_hi/r_hi": 0.0848}   # measured, E29 anchor_local
LEADER = 0.1899


def cfg_official():
    from cell_eval2 import EvalConfig
    cfg = EvalConfig.from_preset("vcc2026")
    cfg = replace(cfg, pert_col="target_gene", device="cpu")
    cfg = replace(cfg, de=replace(cfg.de, backend="scanpy"))
    return cfg


def wide(agg: pl.DataFrame) -> pl.DataFrame:
    w = {"statistic": ["mean"]}
    for r in agg.iter_rows(named=True):
        w[r["metric"]] = [r["mean"]]
    return pl.DataFrame(w)


def drop_h5ad(p: Path) -> None:
    if p.exists():
        mb = p.stat().st_size / 1e6
        p.unlink()
        print(f"  cleaned {p.name} ({mb:.0f} MB)")


def stage_metrics() -> None:
    from cell_eval2 import aggregate_metrics, compute_metrics
    tag = sys.argv[2]
    t0 = time.time()
    real = E27 / "real.h5ad"          # shared, READ-ONLY, never written or deleted here
    assert real.exists(), f"missing shared ground truth {real}"
    agg_p = OUT / f"agg_{tag}.parquet"
    sub = OUT / f"{tag}.h5ad"
    if agg_p.exists():
        print(f"{tag}: have {agg_p.name}, skipping compute")
        drop_h5ad(sub)
        return
    assert sub.exists(), f"missing {sub}; run `build_c1.py build <pred|base>` first"
    df = compute_metrics(str(sub), str(real), config=cfg_official())
    df.write_parquet(OUT / f"per_pert_{tag}.parquet")     # the point of the whole slice
    aggregate_metrics(df).write_parquet(agg_p)
    print(f"{tag}: compute_metrics + aggregate done ({time.time()-t0:.0f}s); "
          f"per-pert rows {df.height:,}; schema {dict(df.schema)}")
    drop_h5ad(sub)                                        # the instant the parquet lands


def _check_reproduces_v8(agg: pl.DataFrame) -> dict:
    """My rebuilt pred must BE V8: its aggregate has to match agg_v8.parquet."""
    mine = {r["metric"]: r["mean"] for r in agg.iter_rows(named=True)}
    ref = {r["metric"]: r["mean"]
           for r in pl.read_parquet(E28 / "agg_v8.parquet").iter_rows(named=True)}
    print(f"\n{'reproduction check vs E28 agg_v8.parquet':44s} "
          f"{'mine':>13s} {'E28 V8':>13s} {'abs diff':>11s}")
    print("-" * 84)
    worst, rep = 0.0, {}
    for m in SCORED:
        d = abs(mine[m] - ref[m])
        worst = max(worst, d)
        rep[m] = {"mine": mine[m], "e28": ref[m], "abs_diff": d}
        print(f"{m:44s} {mine[m]:13.6f} {ref[m]:13.6f} {d:11.2e}")
    print("-" * 84)
    print(f"worst abs diff {worst:.2e}  -> "
          f"{'IDENTICAL (bit-for-bit)' if worst == 0 else 'MATCH within tolerance' if worst < 1e-9 else 'MISMATCH - per-pert rows are NOT V8'}")
    rep["worst_abs_diff"] = worst
    return rep


def _q(v) -> dict:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return dict(n=int(v.size), mean=float(v.mean()), sd=float(v.std(ddof=1)),
                p2_5=float(np.percentile(v, 2.5)), p5=float(np.percentile(v, 5)),
                p50=float(np.percentile(v, 50)), p95=float(np.percentile(v, 95)),
                p97_5=float(np.percentile(v, 97.5)),
                min=float(v.min()), max=float(v.max()))


def stage_analyze() -> None:
    import anchor_local as A
    from cell_eval2 import aggregate_metrics
    from cell_eval2.score import score_metrics
    logging.disable(logging.WARNING)
    t0 = time.time()

    dp = pl.read_parquet(OUT / "per_pert_pred_8.parquet")
    pcol = next(c for c in ("target_gene", "perturbation", "pert") if c in dp.columns)
    perts = sorted(set(dp[pcol].to_list()))
    print(f"per-pert frame: {dp.height:,} rows, pert column {pcol!r}, "
          f"{len(perts)} perturbations")
    print(f"perturbations: {perts}")
    have = sorted(set(dp['metric'].to_list()))
    print(f"metrics present per-perturbation: {have}")
    missing = [m for m in SCORED if m not in have]
    print(f"scored metrics WITHOUT per-perturbation rows (derived): {missing}")

    # b is FIXED at the published baseline aggregate and never resampled.
    b_fixed = wide(pl.read_parquet(E27 / "agg_base.parquet"))
    sub = {p: dp.filter(pl.col(pcol) == p) for p in perts}

    def evaluate(names) -> dict:
        """One panel -> {repo6, repo5, off6/off5 at both ends}."""
        agg = aggregate_metrics(pl.concat([sub[p] for p in names]))
        raw = {r["metric"]: r["mean"] for r in agg.iter_rows(named=True)}
        res = score_metrics(wide(agg), b_fixed, comparison_statistic="mean")
        fb = dict(zip(res["metric"].to_list(), res["from_baseline"].to_list()))
        out = {"repo6": fb["avg_score"],
               "repo5": float(np.mean([fb[m] for m in CLEAN]))}
        off = A.official_two_ended_from_raw({m: raw[m] for m in SCORED})
        for end in ("b_lo/r_lo", "b_hi/r_hi"):
            e = off[end]
            out[f"off6 {end}"] = e["avg_score"]["score"]
            out[f"off5 {end}"] = float(np.mean([e[m]["score"] for m in CLEAN]))
        return out, raw, agg

    # ---- point estimate on the full 8-panel -----------------------------------------
    point, raw8, agg8 = evaluate(perts)
    rep = _check_reproduces_v8(agg8)
    print(f"\nfull 8-panel point estimates")
    for k, v in point.items():
        print(f"  {k:22s} {v:9.4f}")
    assert abs(point["repo6"] - V8_REPO_AVG) < 5e-4, (
        f"repo avg_score {point['repo6']:.4f} does not reproduce V8's {V8_REPO_AVG}")
    print(f"  -> reproduces V8's repo-internal {V8_REPO_AVG} and official "
          f"{V8_OFFICIAL['b_lo/r_lo']}/{V8_OFFICIAL['b_hi/r_hi']}")

    # ---- per-perturbation values: concentration vs spread ----------------------------
    # This is what settles whether a metric's panel mean is carried by one perturbation.
    # `de_direction_reach` returns dict[str, float] (direction.py:862-875) -- one value per
    # target, no k* and no N_conf -- so depth is not recoverable here. What IS recoverable
    # is concentration, and for reach that is the whole question: direction.py:946-968
    # records that reach_raw's no-skill point is ~c/N_conf, NOT 0, with fitted c ~0.96 for
    # a coin-flip predictor, so a small confident budget buys a large chance reach.
    pp = {}
    for m in sorted(set(dp["metric"].to_list())):
        sel = dp.filter(pl.col("metric") == m)
        pp[m] = {r[pcol]: r["value"] for r in sel.iter_rows(named=True)}
    show = [m for m in SCORED if m in pp]
    print(f"\nper-perturbation raw values (derived metrics absent by construction)")
    print(f"{'perturbation':12s} " + " ".join(f"{m[:22]:>22s}" for m in show))
    print("-" * (13 + 23 * len(show)))
    for p in perts:
        print(f"{p:12s} " + " ".join(f"{pp[m].get(p, float('nan')):22.4f}"
                                    for m in show))
    print("-" * (13 + 23 * len(show)))
    print(f"{'panel mean':12s} " +
          " ".join(f"{np.nanmean([pp[m].get(q, np.nan) for q in perts]):22.4f}"
                   for m in show))
    # Share of the panel's reach total held by its single largest contributor.
    rk = "de_wilcoxon_direction_reach_raw"
    if rk in pp:
        vals = np.array([pp[rk].get(p, np.nan) for p in perts], float)
        tot = np.nansum(vals)
        j = int(np.nanargmax(vals))
        print(f"\nreach concentration: total {tot:.4f} over {np.isfinite(vals).sum()} "
              f"perturbations; largest single contributor {perts[j]} = {vals[j]:.4f} "
              f"({100*vals[j]/tot:.1f}% of the total); "
              f"{int((vals == 0).sum())} of {len(vals)} are exactly 0")

    KEYS = ["repo6", "repo5", "off6 b_lo/r_lo", "off5 b_lo/r_lo",
            "off6 b_hi/r_hi", "off5 b_hi/r_hi"]

    # ---- bootstrap over perturbations (b fixed) --------------------------------------
    rng = np.random.default_rng(BOOT_SEED)
    boot = {k: [] for k in KEYS}
    for i in range(N_BOOT):
        names = [perts[j] for j in rng.integers(0, len(perts), len(perts))]
        v, _, _ = evaluate(names)
        for k in KEYS:
            boot[k].append(v[k])
        if (i + 1) % 250 == 0:
            print(f"  bootstrap {i+1}/{N_BOOT} ({time.time()-t0:.0f}s)")
    bs = {k: _q(boot[k]) for k in KEYS}

    # ---- jackknife: which perturbation carries the score? ----------------------------
    jack = {}
    for p in perts:
        v, _, _ = evaluate([q for q in perts if q != p])
        jack[p] = v
    jk_sd = {k: float(np.sqrt((len(perts) - 1) / len(perts)
                              * sum((jack[p][k] - np.mean([jack[q][k] for q in perts]))**2
                                    for p in perts)))
             for k in KEYS}

    hdr = (f"{'quantity':26s} {'point':>9s} {'boot mean':>10s} {'boot sd':>9s} "
           f"{'p5':>9s} {'p95':>9s} {'jack sd':>9s}")
    print(f"\n{'='*len(hdr)}\nsampling error of V8's avg_score over the choice of 8 "
          f"perturbations\n(bootstrap B={N_BOOT}, seed {BOOT_SEED}, baseline b FIXED)"
          f"\n{'='*len(hdr)}\n{hdr}\n{'-'*len(hdr)}")
    for k in KEYS:
        tail = "  CONTAMINATED (incl pds)" if k.startswith("off6") or k == "repo6" \
            else "  clean (5 metrics)"
        print(f"{k:26s} {point[k]:9.4f} {bs[k]['mean']:10.4f} {bs[k]['sd']:9.4f} "
              f"{bs[k]['p5']:9.4f} {bs[k]['p95']:9.4f} {jk_sd[k]:9.4f}{tail}")
    print("-" * len(hdr))

    print(f"\nper-perturbation jackknife (leave-one-out), official b_lo/r_lo 6-metric:")
    base = point["off6 b_lo/r_lo"]
    for p in perts:
        d = jack[p]["off6 b_lo/r_lo"] - base
        print(f"  drop {p:10s} -> {jack[p]['off6 b_lo/r_lo']:7.4f}  ({d:+.4f})")

    # ---- does the error bar separate V8 from the leader? -----------------------------
    v = np.asarray(boot["off6 b_lo/r_lo"], float)
    p_exceed = float((v > LEADER).mean())
    print(f"\nleader comparison, official b_lo/r_lo scale (the decision-grade one):")
    print(f"  V8 point {base:.4f}   leader {LEADER}   gap {base-LEADER:+.4f}")
    print(f"  bootstrap P(V8 > leader) = {p_exceed:.4f}   "
          f"90% interval [{bs['off6 b_lo/r_lo']['p5']:.4f}, "
          f"{bs['off6 b_lo/r_lo']['p95']:.4f}]")
    print(f"  -> {'NOT separated' if p_exceed > 0.05 else 'V8 is below the leader with the sampling error accounted for'}")

    json.dump({"n_boot": N_BOOT, "seed": BOOT_SEED, "perts": perts,
               "baseline": "E27 agg_base.parquet, held FIXED (not resampled)",
               "point": point, "raw_8panel": raw8, "reproduction_vs_v8": rep,
               "per_perturbation": pp,
               "bootstrap": bs, "jackknife_sd": jk_sd,
               "jackknife_leave_one_out": jack,
               "p_exceed_leader_off6_blo": p_exceed},
              open(OUT / "analyze.json", "w"), indent=2)
    pl.DataFrame(boot).write_parquet(OUT / "bootstrap.parquet")
    print(f"\nwrote analyze.json + bootstrap.parquet; elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    {"metrics": stage_metrics, "analyze": stage_analyze}[sys.argv[1]]()
