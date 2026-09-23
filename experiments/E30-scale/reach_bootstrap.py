"""Sampling error of V8's avg_score attributable to de_wilcoxon_direction_reach_raw alone.

This is the ZERO-COMPUTE partial answer, run while C1 (the full six-metric bootstrap) is
queued behind a resource ruling. It needs no .h5ad, no compute_metrics, and ~40 MB of RAM.

Input: the eight per-perturbation reach_raw values, MEASURED by SourceUnion's validated reach
estimator (which reproduced cell_eval2's aggregate to -0.0000). Self-check below: their mean
must reproduce the 0.148158 that experiments/E28-pds/out/agg_v8.parquet records, and
agg_v8.parquet marks this metric agg="mean", so the plain mean IS its aggregate.

What this gives and does not give: reach is one of six equally-weighted members of avg_score,
so bootstrapping it alone yields the sd of ITS contribution. That is a LOWER BOUND on the sd
of the full avg_score, not the answer -- the other five members carry their own sampling
variance, and variances add for weakly-correlated terms. It is reported as a bound and
labelled as one.

The official-scale arithmetic is not hand-rolled: it takes the catalog policy, replaces the
anchor with the official r exactly as score.py:431 does, and calls scoring.score_one exactly
as score.py:270 does -- the same path experiments/E29-official-baseline/anchor_local.py uses.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace as dc_replace
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out"
sys.path.insert(0, str(ROOT / "experiments" / "E29-official-baseline"))

METRIC = "de_wilcoxon_direction_reach_raw"
N_BOOT, SEED = 20000, 20260922
N_MEMBERS = 6          # avg_score is an unweighted mean over the six vcc2026 members

#: MEASURED per-perturbation reach_raw (SourceUnion, validated estimator, -0.0000 vs
#: cell_eval2's aggregate). N_conf and k* included where reported.
PER_PERT = {
    "MAT2A":  {"reach": 1.0000, "N_conf": 5,   "k_star": 5},
    "ZNF581": {"reach": 0.0625, "N_conf": 16,  "k_star": 1},
    "COX4I1": {"reach": 0.0574, "N_conf": 122, "k_star": 7},
    "SLIRP":  {"reach": 0.0435, "N_conf": 253, "k_star": 11},
    "VCL":    {"reach": 0.0184, "N_conf": None, "k_star": None},
    "GNG12":  {"reach": 0.0032, "N_conf": None, "k_star": None},
    "PAXIP1": {"reach": 0.0008, "N_conf": None, "k_star": None},
    "TCF7L2": {"reach": 0.0000, "N_conf": None, "k_star": None},
}


def official_reach_score(u: float, b: float, r: float) -> float:
    from cell_eval2.catalog import CATALOG, _NAME_TO_CANONICAL
    from cell_eval2.scoring import score_one
    spec = CATALOG[_NAME_TO_CANONICAL.get(METRIC, METRIC)]
    policy = dc_replace(spec.scoring, anchor=float(r), allow_negative_baseline=False)
    return float(score_one(float(u), float(b), policy))


def q(v) -> dict:
    v = np.asarray(v, float)
    return dict(n=int(v.size), mean=float(v.mean()), sd=float(v.std(ddof=1)),
                p2_5=float(np.percentile(v, 2.5)), p5=float(np.percentile(v, 5)),
                p50=float(np.percentile(v, 50)), p95=float(np.percentile(v, 95)),
                p97_5=float(np.percentile(v, 97.5)),
                min=float(v.min()), max=float(v.max()))


def main() -> None:
    import anchor_local as A
    perts = list(PER_PERT)
    vals = np.array([PER_PERT[p]["reach"] for p in perts], float)

    # ---- self-check against the shipped aggregate ------------------------------------
    ref = {r["metric"]: r["mean"] for r in pl.read_parquet(
        ROOT / "experiments" / "E28-pds" / "out" / "agg_v8.parquet").iter_rows(named=True)}
    mine, shipped = float(vals.mean()), ref[METRIC]
    # The eight inputs were reported to 4 decimals, so their mean carries up to ~5e-5 of
    # rounding error. The tolerance is set to that rounding floor, not to float epsilon:
    # a tighter bar would fail on the reporting precision rather than on the arithmetic.
    # The floor is ~3 orders of magnitude below the sd this script measures, so it does not
    # affect any conclusion.
    print(f"self-check: mean of the 8 per-perturbation values = {mine:.6f}")
    print(f"            agg_v8.parquet {METRIC} = {shipped:.6f}")
    print(f"            abs diff {abs(mine-shipped):.2e} (4-decimal rounding floor ~5e-5)"
          f"  -> {'CONSISTENT' if abs(mine-shipped) < 2e-4 else 'INCONSISTENT'}")
    assert abs(mine - shipped) < 2e-4, \
        f"per-perturbation values do not reproduce the aggregate: {mine} vs {shipped}"

    b_lo, b_hi, r_lo, r_hi = A.OFFICIAL_BR[METRIC]
    ends = {"b_lo/r_lo": (b_lo, r_lo), "b_hi/r_hi": (b_hi, r_hi)}
    print(f"\nofficial b/r for {METRIC}: b {b_lo}-{b_hi}, r {r_lo}-{r_hi}")

    # ---- contribution concentration --------------------------------------------------
    print(f"\n{'pert':10s} {'reach':>9s} {'N_conf':>7s} {'k*':>4s} {'contrib':>9s} {'share':>8s}")
    print("-" * 52)
    for p in perts:
        d = PER_PERT[p]
        print(f"{p:10s} {d['reach']:9.4f} "
              f"{d['N_conf'] if d['N_conf'] is not None else '-':>7} "
              f"{d['k_star'] if d['k_star'] is not None else '-':>4} "
              f"{d['reach']/len(perts):9.4f} {100*d['reach']/vals.sum():7.1f}%")
    print("-" * 52)
    print(f"{'mean':10s} {mine:9.4f}")

    # ---- bootstrap -------------------------------------------------------------------
    rng = np.random.default_rng(SEED)
    draws = vals[rng.integers(0, len(vals), (N_BOOT, len(vals)))].mean(axis=1)
    raw_q = q(draws)
    # Probability a resampled panel excludes MAT2A entirely: (7/8)^8 analytically.
    p_no_mat2a = (1 - 1 / len(vals)) ** len(vals)
    print(f"\nbootstrap B={N_BOOT}, seed {SEED}: raw reach_raw panel mean")
    print(f"  point {mine:.4f}  mean {raw_q['mean']:.4f}  sd {raw_q['sd']:.4f}  "
          f"90% [{raw_q['p5']:.4f}, {raw_q['p95']:.4f}]")
    print(f"  P(resample contains no MAT2A) = (7/8)^8 = {p_no_mat2a:.4f}")

    res = {"self_check": {"mean_of_per_pert": mine, "agg_v8": shipped,
                          "abs_diff": abs(mine - shipped)},
           "per_perturbation": PER_PERT, "n_boot": N_BOOT, "seed": SEED,
           "raw_reach": raw_q, "p_resample_without_mat2a": p_no_mat2a, "ends": {}}

    hdr = (f"{'end':12s} {'point':>9s} {'boot mean':>10s} {'sd':>9s} {'p5':>9s} "
           f"{'p95':>9s} {'sd/avg_score':>13s}")
    print(f"\nofficial-scale reach member, and its contribution to a 6-member avg_score")
    print(f"{hdr}\n{'-'*len(hdr)}")
    for lab, (b, r) in ends.items():
        pt = official_reach_score(mine, b, r)
        sc = np.array([official_reach_score(u, b, r) for u in draws], float)
        sq = q(sc)
        contrib = q(sc / N_MEMBERS)
        print(f"{lab:12s} {pt:9.4f} {sq['mean']:10.4f} {sq['sd']:9.4f} "
              f"{sq['p5']:9.4f} {sq['p95']:9.4f} {contrib['sd']:13.4f}")
        res["ends"][lab] = {"b": b, "r": r, "point": pt, "member": sq,
                            "avg_score_contribution": contrib}
    print("-" * len(hdr))
    print("last column = sd of this ONE member's contribution to avg_score;")
    print("it is a LOWER BOUND on the sd of the full six-metric avg_score.")

    # ---- the same bound on the repo-internal from_baseline scale (V8 = 0.2025) --------
    # Same catalog policy, but the CATALOG's own anchor and the repo's measured baseline b
    # from E27's control-mean-tiled aggregate -- i.e. exactly what score_metrics computes
    # for this member when score_v8.py calls it.
    from cell_eval2.catalog import CATALOG, _NAME_TO_CANONICAL
    from cell_eval2.scoring import score_one
    b_repo = {r["metric"]: r["mean"] for r in pl.read_parquet(
        ROOT / "experiments" / "E27-six-metrics" / "out" / "agg_base.parquet"
    ).iter_rows(named=True)}[METRIC]
    spec = CATALOG[_NAME_TO_CANONICAL.get(METRIC, METRIC)]
    pt_repo = float(score_one(mine, b_repo, spec.scoring))
    sc_repo = np.array([float(score_one(u, b_repo, spec.scoring)) for u in draws], float)
    rq, rc = q(sc_repo), q(sc_repo / N_MEMBERS)
    print(f"\nrepo-internal from_baseline scale (the one V8's 0.2025 lives on)")
    print(f"  baseline b for this member (E27 agg_base) = {b_repo:.6f}")
    print(f"  member: point {pt_repo:.4f}  sd {rq['sd']:.4f}  "
          f"90% [{rq['p5']:.4f}, {rq['p95']:.4f}]")
    print(f"  contribution to avg_score: sd {rc['sd']:.4f}")
    print(f"  -> V8 repo-internal avg_score 0.2025 +/- AT LEAST {rc['sd']:.4f} "
          f"(1 sd, reach member alone)")
    res["repo_internal"] = {"b": b_repo, "point": pt_repo, "member": rq,
                            "avg_score_contribution": rc}

    # ---- what the bound means against the leader -------------------------------------
    lo = res["ends"]["b_lo/r_lo"]["avg_score_contribution"]["sd"]
    hi = res["ends"]["b_hi/r_hi"]["avg_score_contribution"]["sd"]
    print(f"\nV8 official avg_score 0.1215 (b_lo/r_lo) / 0.0848 (b_hi/r_hi); "
          f"leader {0.1899}")
    print(f"  gap to leader: {0.1215-0.1899:+.4f} / {0.0848-0.1899:+.4f}")
    print(f"  reach-only sampling sd: {lo:.4f} / {hi:.4f}  -> the gap is "
          f"{abs(0.1215-0.1899)/lo:.1f} / {abs(0.0848-0.1899)/hi:.1f} reach-only sds wide")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "reach_bootstrap.json", "w"), indent=2)
    print(f"\nwrote {OUT/'reach_bootstrap.json'}")


if __name__ == "__main__":
    main()
