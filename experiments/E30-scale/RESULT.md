# E30 — How much of V8's 0.2025 is sampling noise, and can the scale mismatches be closed locally?

**Scale statement, read this first.** Every `avg_score` labelled `repo` below sits on the
**repo-internal anchor-0/1 `from_baseline` scale** produced by
`score_metrics(agg_pred, agg_base, comparison_statistic="mean")` with **no** `anchor=` or
`real_bundle=` argument, against the **control-mean-tiled degenerate baseline**
(`experiments/E27-six-metrics/out/agg_base.parquet`). That is the identical call every
variant A..V8 in this repo was scored through, so `repo` numbers are comparable to V8's
0.2025 and to each other. Every `avg_score` labelled `official` is the two-ended
`(u - b)/(r - b)` scale computed through the shared helper
`experiments/E29-official-baseline/anchor_local.py`, reported at **both ends** of the
official b/r ranges and never as a midpoint.

**Scale mismatches this slice's numbers still carry.** Of the three mismatches between our
harness and the leaderboard — baseline definition, 8 perturbations vs 300, 1 context vs 3 —
this experiment **closes none of them** and instead **measures the size of the second** and
**proves the third is not closable locally**. Numerators are 8-perturbation / 1-context
measurements; the official denominators b and r are 300-perturbation / 3-context
quantities. These numbers may be placed beside the leader's 0.1899 but are **not** apples
to apples.

---

## Part 2 — the context axis: CLOSED, not locally closable

**Verdict.** The 1-context-vs-3 mismatch **cannot be closed locally, by any amount of
compute**, because the perturbed ground truth for contexts A/B/C does not exist on this
machine. It cannot be closed before the organizers release the finals cell lines on
**2026-10-22** (`docs/00-problem.md:131`), and that release is of *new* contexts D/E/F plus
a *different* 300 genes, not the missing A/B/C labels.

All evidence below is `measured` by `context_probe.py` (h5py metadata only — no matrix was
loaded); raw log in `context_probe.log`, machine-readable dump in `context_evidence.json`.

### The three context files contain only controls

| file | shape | `obs` columns | `target_gene` categories | `context` categories |
|---|---|---|---|---|
| `/Users/chetianc/vcc2026/context_A.h5ad` | 18,400 x 18,533 | `_index, context, ntc_id, target_gene` | **1**: `non-targeting` (18,400 cells) | 1: `A` |
| `/Users/chetianc/vcc2026/context_B.h5ad` | 18,400 x 18,533 | `_index, context, ntc_id, target_gene` | **1**: `non-targeting` (18,400 cells) | 1: `B` |
| `/Users/chetianc/vcc2026/context_C.h5ad` | 18,400 x 18,533 | `_index, context, ntc_id, target_gene` | **1**: `non-targeting` (18,400 cells) | 1: `C` |

`ntc_id` carries 46 categories in each file, every one named `non-targeting-NN` and every
one holding exactly 400 cells (46 x 400 = 18,400). So each file is a **pure control block**:
**zero** perturbed cells, and therefore zero ground-truth labels for any of the 300
constructs. `18,400` is exactly the `control_cells` field the manifest declares.

### What the manifest advertises but the files do not contain

`/Users/chetianc/vcc2026/manifest.json` declares, per context:
`n_perturbations: 300`, `control_cells: 18400`, `ground_truth_cells: 138400`, `n_ntc_ids: 46`,
with `cells_per_pert: 400` and `n_constructs: 300`. Note
`138,400 = 18,400 + 300 x 400`. The 18,400 controls are present; the **120,000 perturbed
cells are absent from all three files**. `pert_counts.csv` is a `(300, 1)` frame whose only
column is `target_gene` — a list of construct **names**, with no counts and no expression.
`de_pred_official.parquet` is a 3-target fixture (`ABCD1`, `ADNP`, `ACLY`; 29,787 rows),
and `parity_real.h5ad` / `parity_pred.h5ad` are 19,600 x 18,533 parity fixtures whose
`target_gene` has 4 categories (`ABCD1`, `ACLY`, `ADNP`, `non-targeting`) and which carry
**no `context` column at all**. Nothing in that directory supplies labelled perturbed cells
for a second or third context.

### Our own ground truth has no context axis whatsoever

`data/vcc2025/adata_Validation.h5ad` (98,927 x 18,080) has
`obs = [_index, batch, guide_id, target_gene]`. There is **no cell-line or context column**.
`batch` is 48 categories, all of the form `Flex_N_NN` — 10x Flex **sequencing batches**, not
cell lines. `target_gene` has 51 categories: `non-targeting` (38,176 cells) plus **50**
non-control perturbations. `uns`, `layers` and `obsm` are all empty, so no metadata hides a
cell-line annotation elsewhere.

There is also a **gene-space mismatch** that would block reuse even of the control blocks:
the context files carry 18,533 genes, our validation ground truth carries 18,080.

### Consequence

Our harness is single-context by construction and cannot be made otherwise locally. Any
3-context number is `inferred` at best. What *would* measure it: labelled perturbed cells
for at least two contexts in a shared gene space — i.e. the organizers' held-out ground
truth, which is exactly what is withheld.

---

## The perturbation axis: 41 is the hard local ceiling

`panel_check.py` (measured; log in `panel_check.log`) reproduces `build_v8.py:147-154`'s
panel selection without touching a matrix:

| quantity | value |
|---|---|
| validation non-control `target_gene` categories | 50 |
| K562GW source columns (`data/nadig2025/K562GW_p.csv.gz`) | 9,866 |
| intersection, `usable` | 47 |
| ... **and** >= `VCC_PERT_CELLS` = 400 real cells -> `cand` | **41** |
| leaderboard scale (`manifest.json: n_constructs`) | 300 |

So **mismatch #2 (8 vs 300) is also not closable locally**: the largest panel this harness
can ever score is **41 perturbations, 13.7% of the leaderboard's 300**. The binding
constraint is not disk or time, it is that only 41 perturbations are simultaneously (a)
present in the validation ground truth, (b) present in the Nadig K562 genome-wide source,
and (c) backed by >= 400 real cells.

Cost of that implied maximum, for the record: a 41-perturbation panel is
`41 x 400 + 18,400 = 34,800` cells, so ~470 MB per gzip file at the measured 13.5 KB/cell
[`inferred` from the measured rate below], with peak disk `real + one other` ~940 MB, and
`compute_metrics` at roughly 1,289 s per 8 perturbations scaling ~linearly -> ~1.8 h per
side, ~3.7 h for pred and base. Affordable in isolation; it was not affordable against five
concurrent agents on this machine.

### The N=24 panel is not a superset of the N=8 panel — the original design was confounded

The assignment asked whether `N_PERT=24` yields a stratified **superset** of the original 8.
**It does not.** `pick = [cand[int(i)] for i in np.linspace(0, len(cand)-1, N_PERT).round()]`
with `len(cand) = 41` gives:

- `N=8` indices `[0, 6, 11, 17, 23, 29, 34, 40]` -> `TCF7L2, GNG12, VCL, COX4I1, MAT2A, PAXIP1, SLIRP, ZNF581` (reproduces `experiments/E28-pds/out/perts.csv` exactly — checked)
- `N=24` indices `[0, 2, 3, 5, 7, 9, 10, 12, 14, 16, 17, 19, 21, 23, 24, 26, 28, 30, 31, 33, 35, 37, 38, 40]`

Overlap is **4 of 8**. `GNG12`, `VCL`, `PAXIP1` and `SLIRP` are **dropped** by the 24-panel.
`round()` shifts the grid, so the two panels are **different samples, not nested ones**.

This is why the assigned 8-of-24 resampling design was abandoned rather than merely
descoped: an arm that moves the very thing it is supposed to hold constant is a designed-in
confound of the F29 family. The replacement design is stated below. `build_scale.py`
(the `N_PERT = 8 -> 24` knob) was still generated and syntax-validated, and is committed
unrun as evidence of the mandated single-knob transformation.

---

## Commands run

```bash
cd /Users/chetianc/code/vcc2026-stage1-lab/experiments/E30-scale
PY=/Users/chetianc/vcc2026/.venv/bin/python

# panel selection + superset check (no matrices)
$PY panel_check.py                 # -> panel_check.log

# context-axis evidence (h5py metadata only)
$PY context_probe.py               # -> context_probe.log, context_evidence.json

# generate both builds from build_v8.py by asserted str.replace
$PY gen_build.py                   # -> build_c1.py, build_scale.py
```

### The mandated single-knob transformation

`gen_build.py` separates the two kinds of edit so attribution is auditable. Both generated
files assert V8's three modelling knobs survive verbatim (`LAMBDA = 0.7`, `K = 288`,
`score = np.where(good, np.abs(b), -np.inf)`), and every replacement asserts it landed
(source string gone — or reintroduced exactly once for a wrapping insert — and replacement
present).

- **`build_scale.py`** = harness + the **one knob**, `N_PERT = 8` -> `N_PERT = 24`.
  `diff build_c1.py build_scale.py` is **exactly one line**, the `N_PERT` line. Generated
  and validated; **not run**.
- **`build_c1.py`** = harness **only**; `N_PERT` stays 8, so it **reproduces V8 exactly**.
  It exists to obtain the per-perturbation metric rows that V8's original run aggregated
  away.

The harness edits are plumbing with zero modelling effect: redirect `OUT` into this
experiment; let one invocation write one tag; drop the blocks a pass does not write (peak
RSS ~1.6 GB -> ~0.6 GB); and check the 5 GB disk floor *immediately before* each
`write_h5ad` rather than once at launch. The RAM edit is reproducibility-safe because every
block is still **constructed** — only retention is refused — so the shared `rng` stream is
consumed identically regardless of which tag is written; and independently, `pred` uses
`design_cells(..., seed=SEED)` with a fixed seed while `base` is deterministic, so neither
reads `rng` at all.

---

## Measured file-size model (corrects the dispatch estimate)

Submission size is dominated by **controls**, not perturbations: every file carries
`VCC_CTRL_CELLS = 18,400` control cells plus only `N_PERT x 400` perturbed ones.

| | cells | size | source |
|---|---|---|---|
| `N_PERT=8` | 21,600 | **292.4 MB** | `measured`; identical across `E28/pred_v8.h5ad`, `E31/pred_v8.h5ad`, `E32/pred_v10.h5ad`, `E32/pred_v12.h5ad` |
| rate | | **13.5 KB/cell** | derived from the above |
| `N_PERT=16` | 24,800 | ~335 MB | `inferred` from the rate |
| `N_PERT=24` | 28,000 | ~379 MB | `inferred` from the rate |

The dispatch estimate of ~875 MB per 24-perturbation file scaled on perturbation count and
is **wrong by 2.3x**. The corollary matters more than the correction: **cutting `N_PERT`
buys almost nothing** (24 -> 16 saves ~11%), so perturbation count is the wrong thing to
trade away for space.

---

## Part 1 — the sampling error of V8's avg_score

**Status: DONE** (2026-09-23). The memory ruling that queued this has cleared. Run as
`build_c1.py build pred` (198 s, `pred_8.h5ad` 292 MB, shape (21600, 18080)) ->
`score_scale.py metrics pred_8` (1530 s, **71** per-perturbation rows) ->
`score_scale.py analyze` (8 s, B = 2000, seed 20260922, baseline `b` FIXED).
Artifacts: `out/analyze.json`, `out/bootstrap.parquet`, `out/per_pert_pred_8.parquet`
(the last two `*.parquet` and therefore gitignored — every number below is also in
`analyze.json`, which is not), `score_scale_metrics.log`, `score_scale_analyze.log`.
The `.h5ad` was unlinked the instant the parquet landed; this experiment again holds zero.

### The rebuilt rows ARE V8, bit-for-bit

All six scored metrics reproduce `E28-pds/out/agg_v8.parquet` at **`abs diff = 0.00e+00`**
(`measured`, not "within tolerance"): `sig_jaccard` 0.048234, `lfc_nmae` 1.000361,
`direction_fidelity_yield_raw` 0.491874, `direction_reach_raw` 0.148158, `pds_cosine`
0.750000, `expr_mse_unbiased_capped_norm` 1.081883. The assembled point estimates
reproduce V8's repo-internal **0.2025** and official **0.1215 / 0.0848**. So the
per-perturbation frame is V8's own decomposition, not a lookalike rebuild.

⚠️ The uncommitted `src/vcclab/decoder.py` change in the tree at run time (scalar `shift`
generalised to an optional per-gene vector, for E34's V15) is **bit-identical on the scalar
path**: `np.broadcast_to(np.asarray(0.10, float), (n,))[i]` is exactly 0.10, so
`ut = 0.5 + sign(l) * shift` is unchanged. The 0.00e+00 above is the proof, not the claim.

### The answer: "0.2025 +/- what"

| quantity | point | boot mean | boot sd | p5 | p95 | jack sd |
|---|---|---|---|---|---|---|
| `repo6` **contaminated** | 0.2025 | 0.1963 | **0.0348** | 0.1418 | 0.2552 | 0.0369 |
| `repo5` clean | 0.1430 | 0.1371 | 0.0320 | 0.0741 | 0.1771 | 0.0246 |
| `off6 b_lo/r_lo` **contaminated** | 0.1215 | 0.1342 | **0.0423** | 0.0680 | 0.2022 | 0.0441 |
| `off5 b_lo/r_lo` clean | 0.0287 | 0.0458 | 0.0321 | −0.0002 | 0.0973 | 0.0311 |
| `off6 b_hi/r_hi` **contaminated** | 0.0848 | 0.0984 | 0.0395 | 0.0379 | 0.1605 | 0.0409 |
| `off5 b_hi/r_hi` clean | −0.0015 | 0.0163 | 0.0330 | −0.0314 | 0.0684 | 0.0323 |

> **V8 = 0.2025 ± 0.035** (repo-internal, 1 sd over the choice of 8 perturbations), and on
> the only scale comparable to the leaderboard, **0.1215 ± 0.042** (lo) / **0.0848 ± 0.040** (hi).
> Bootstrap and jackknife agree to within 0.002–0.008, so the bar is not an artifact of
> either resampler.

The `reach_bootstrap.py` lower bound was 0.0211 from that one member; the full six-metric
sd is **0.0423**, i.e. 2.0x the single-member bound. Variances adding was the right model.

#### Why the bootstrap mean sits ABOVE the point on the official scale (measured, B = 500)

Per-member, official `b_lo/r_lo`, `point -> boot mean (shift)`:

| member | point | boot mean | shift | P(score ≤ 0) |
|---|---|---|---|---|
| `expr_mse_unbiased_capped_norm` | **0.0000** | **0.0834** | **+0.0834** | 0.566 |
| `pds_cosine` | 0.5855 | 0.5696 | −0.0159 | 0.002 |
| `de_wilcoxon_sig_jaccard` | 0.0769 | 0.0734 | −0.0035 | 0.240 |
| `de_wilcoxon_direction_fidelity_yield_raw` | −0.0453 | −0.0432 | +0.0021 | 0.938 |
| `de_wilcoxon_lfc_nmae` | 0.0009 | −0.0007 | −0.0015 | 0.524 |
| `de_wilcoxon_direction_reach_raw` | 0.1110 | 0.1104 | −0.0007 | 0.450 |

The whole +0.0106 shift is `expr_mse_unbiased_capped_norm` (+0.0834/6 = +0.0139, against
−0.0195/6 = −0.0033 from the other five). Its point value is **exactly** 0.0000 because the
policy carries `clamp_low = 0.0` and V8's raw 1.0819 exceeds the official `b ≈ 0.986` — the
member is pinned to a **one-sided floor**, so 43.4% of resamples can only move it UP and
`E[clamp(x)] > clamp(E[x])`. **This is a clamping artifact, not estimator bias: do not
bias-correct the point.** `fid`'s 0.938 is a different thing — it is genuinely negative, not
clamped; only `mse` carries a floor.

### Consequence 1 — V8 is NOT separated from the leader

`P(V8 > 0.1899) = 0.0885`, 90% interval **[0.0680, 0.2022]**, which **contains** the leader.
The roadmap's "缺 0.0391" is smaller than the panel-choice noise on the very panel that
produced it. This does **not** mean we are level with the leader — their 0.1899 is measured
on 300 perturbations x 3 contexts and carries a far smaller bar. It means **an
8-perturbation panel cannot resolve a 0.039 gap**, so no ranking decision may rest on one.

### Consequence 2 — one perturbation is worth the entire V13 "improvement"

Leave-one-out on `off6 b_lo/r_lo`:

| drop | score | Δ | | drop | score | Δ |
|---|---|---|---|---|---|---|
| GNG12 | 0.1502 | **+0.0287** | | ZNF581 | 0.1186 | −0.0029 |
| TCF7L2 | 0.1471 | **+0.0256** | | COX4I1 | 0.1195 | −0.0020 |
| PAXIP1 | 0.1019 | −0.0196 | | SLIRP | 0.1205 | −0.0011 |
| MAT2A | 0.1023 | −0.0192 | | VCL | 0.1220 | +0.0005 |

Dropping GNG12 alone moves the score **+0.0287**; V13's entire claimed gain over V8 is
**+0.0293**. Deleting one of eight perturbations is worth as much as the change the project
adopted a new configuration for.

### Consequence 3 — on the five members that survive resampling, V13 and V14 are WORSE than V8

`out/clean5_vs_full6.json` (`measured`, official rescale of each variant's published raw):

| variant | `avg6` scored (lo / hi) | `avg5` clean (lo / hi) | `pds` member (lo / hi) |
|---|---|---|---|
| V8 | 0.1215 / 0.0848 | **0.0287 / −0.0015** | 0.5855 / 0.5165 |
| V13 | 0.1508 / 0.1069 | **0.0054 / −0.0267** | 0.8782 / 0.7748 |
| V14 | 0.1158 / 0.0765 | **0.0052 / −0.0263** | 0.6691 / 0.5903 |

Δ vs V8: V13 **+0.0293 / +0.0221** on the scored `avg6` but **−0.0234 / −0.0251** on clean-5;
V14 **−0.0057 / −0.0084** and **−0.0235 / −0.0248**. V13's entire advantage lives inside
`pds_cosine` — the one member section "Mandatory caveat" above forbids reading as
transferable.

Since `avg6 = (Δpds + 5·Δavg5)/6`, adoption has an exact break-even
(`out/breakeven_pds_retention.json`):

| variant | end | Δavg5 | Δpds on 8 perts | must retain | as a fraction |
|---|---|---|---|---|---|
| V13 | lo | −0.0234 | +0.2927 | +0.1168 | **39.9%** |
| V13 | hi | −0.0251 | +0.2583 | +0.1256 | **48.6%** |
| V14 | lo | −0.0235 | +0.0836 | +0.1177 | **impossible** |
| V14 | hi | −0.0248 | +0.0738 | +0.1239 | **impossible** |

> **V13 beats V8 on the scored metric only if `pds_cosine` retains ≥ 40% (lo) / ≥ 49% (hi)
> of its 8-panel advantage when the panel becomes 300 perturbations x 3 contexts.** The three
> measured reasons in the scale warning all say that is the member least likely to survive.
> **V14 is dominated unconditionally** — it needs +0.1177 from a gain of only +0.0836, so no
> retention level saves it. That is strictly stronger than E34 SPEC §8e's "机制未成立":
> V14 is not merely a failed mechanism, it is worse than V8 at both ends on **both** scales.

### Consequence 4 — the transferable core has NOT moved across V8 -> V13 -> V14

Consequence 3 says V13 pays for `pds` out of the other five. Decomposing **where** that
payment comes from changes the reading (`measured`, official `b_lo/r_lo`, per member):

| member | V8 | V13 | Δ | transferable? |
|---|---|---|---|---|
| `de_wilcoxon_direction_reach_raw` | 0.1110 | −0.0241 | **−0.1352** | **NO** — MAT2A renormalization coincidence ([F35](../../docs/02-findings.md#f35)) |
| `pds_cosine` | 0.5855 | 0.8782 | **+0.2927** | **NO** — panel-relative, this section |
| `de_wilcoxon_lfc_nmae` | 0.0009 | −0.0168 | −0.0177 | yes |
| `de_wilcoxon_direction_fidelity_yield_raw` | −0.0453 | −0.0092 | +0.0361 | yes |
| `de_wilcoxon_sig_jaccard` | 0.0769 | 0.0770 | +0.0000 | yes |
| `expr_mse_unbiased_capped_norm` | 0.0000 | 0.0000 | +0.0000 | yes |

**Essentially the whole clean-5 loss is the `reach` member** (−0.1352 of −0.1166 total,
partly offset by `fid` +0.0361). And F35 established, independently of this slice, that
V8's `reach` credit was never skill either. So **both** variants' largest members are
non-transferable — V8's `reach` +0.1110 (a coincidence) and V13's `pds` +0.2927
(panel-relative). Removing both leaves the four members whose value would actually carry to
the finals (`out/transferable_core4.json`):

| variant | core-4 lo | core-4 hi | Δ vs V8 (lo / hi) |
|---|---|---|---|
| V8 | 0.0081 | −0.0164 | — |
| V13 | 0.0127 | −0.0129 | +0.0046 / +0.0035 |
| V14 | 0.0137 | −0.0113 | +0.0055 / +0.0052 |

V8's core-4 bootstrap (B = 2000, same rows, same seed): **sd 0.0389**, 90%
[−0.0086, 0.0960]. The V13-vs-V8 difference on the transferable core is therefore
**0.12 sd**.

> **Three decoder configurations, three official scores spanning 0.1158–0.1508, and the
> transferable content is flat at 0.008–0.014 ± 0.039 — negative at the pessimistic end for
> all three.** V14 is nominally highest on core-4 (+0.0055) and lowest on the scored `avg6`;
> at 0.14 sd that ordering is noise and must not be acted on either.
>
> This is the measured version of the roadmap's own argument that the real P0 is finding a
> transferable lfc **sign** source. E34's decoder work was not wasted — it retired `reach`
> and mapped the `pds` knob — but it has **not** moved transferable skill, and no further
> `(k_off, λ_off)` sweep can, because `nmae`/`fid`/`jac`/`mse` are the only members it could
> move and all four are flat inside a 0.039 bar.

### reach is confirmed hollow — now from cell_eval2's own per-perturbation rows

| pert | `sig_jaccard` | `lfc_nmae` | `fid_yield_raw` | `reach_raw` | `pds_cosine` |
|---|---|---|---|---|---|
| COX4I1 | 0.0122 | 0.9671 | 0.4838 | 0.0574 | 0.8571 |
| GNG12 | 0.0606 | 1.0081 | 0.4983 | 0.0033 | 0.2857 |
| MAT2A | 0.0006 | `nan` | 0.5174 | **1.0000** | 0.7143 |
| PAXIP1 | **0.2527** | 1.0077 | 0.4437 | 0.0006 | 1.0000 |
| SLIRP | 0.0239 | 1.0322 | 0.5276 | 0.0435 | 1.0000 |
| TCF7L2 | 0.0265 | 1.0110 | 0.4696 | **0.0000** | 0.4286 |
| VCL | 0.0081 | 1.0026 | 0.4889 | 0.0180 | 0.8571 |
| ZNF581 | 0.0012 | 0.9738 | 0.5057 | 0.0625 | 0.8571 |
| panel mean | 0.0482 | 1.0004 | 0.4919 | 0.1482 | 0.7500 |

MAT2A alone is **84.4%** of the panel's reach total (1.0000 of 1.1853) and TCF7L2 is exactly
0. The earlier "~5/6 artifact" verdict rested on a hand-built estimator; it now rests on the
official per-perturbation rows. Same shape in `sig_jaccard`: PAXIP1's 0.2527 is 65% of that
metric's total, and it is our only member ahead of the whole field.

### Three properties of the bar that must be quoted with it

- `expr_mse_unbiased_capped_norm` has **no per-perturbation rows**. The frame carries nine
  metrics x 8 perturbations minus MAT2A's absent `nmae` = **71** rows, and this member is not
  among them: it is derived from `expr_mse_unbiased_capped` and `expr_real_mass_ratio`, and it
  aggregates `ratio_of_sums`, not `mean`. Every resample re-derives it through
  `cell_eval2.aggregate_metrics`; a `np.nanmean` over rows would have been silently wrong,
  which is why the script never reimplements aggregation.
- `de_wilcoxon_lfc_nmae` has **7** rows, not 8: MAT2A is omitted for fewer than 10 gated
  genes, and the gate is real-side only so that omission is identical for every submission.
  Resamples drawing MAT2A repeatedly therefore thin that member. Faithful, not a defect.
- The bar measures variability over the **choice of 8 perturbations**. It does not apply to a
  paired same-panel comparison: V13 vs V8 on these exact cells is deterministic. What the bar
  licenses is the transfer statement — an ordering measured on one 8-panel is weak evidence
  about the ordering on the finals' panel.


### Design actually used (C1)

The quantity of interest is the sampling error of a **mean over 8 perturbations**, which is
estimable from the 8 per-perturbation rows themselves by bootstrap and jackknife. It does
**not** require a larger panel. This is both cheaper than the 8-of-24 design and a better
estimator of the right thing, for the superset reason above.

`compute_metrics` returns per-perturbation rows *before* `aggregate_metrics` averages them,
and **no un-aggregated frame exists anywhere on disk** from any prior run: every cell_eval2
output in `experiments/` is a 10-row aggregate (`metric`, `mean`, `agg`) — nine files,
`E27/agg_pred`, `E27/agg_base`, `E28/agg_v2..agg_v8`. The per-perturbation `result.csv`
files in `E10..E26` hold our own in-house proxy metrics (`h`, `jac`), not cell_eval2's six.
So one new `compute_metrics` pass is unavoidable; `score_scale.py metrics` persists the
tidy frame this time.

**The baseline `b` is held FIXED** at `E27/agg_base.parquet` and is never resampled.
Justification, as required: *on the real leaderboard b and r are published constants, not
per-submission quantities, so resampling the baseline would model a source of variation
that does not exist in the competition.* Resampling it (C2) is therefore not merely
optional but arguably the **less** faithful design.

Aggregation per resample is delegated to `cell_eval2.aggregate_metrics` and never
reimplemented, because `expr_mse_unbiased_capped_norm` aggregates as **`ratio_of_sums`**,
not as a mean (`measured`: the `agg` column of `agg_v8.parquet`). A `np.nanmean` over its
per-perturbation rows would have been silently wrong.

### Mandatory caveat: pds_cosine does not survive resampling

`pds_cosine` is a **panel-relative rank metric**. `cell_eval2/metrics/discrimination.py:215`
sets `D = n` or `n - 1` over the panel's non-control perturbations — the `vcc2026` preset
uses `rank_denominator="n-1"` — and `:248` drops **every** panel target gene from both
operands before any distance is computed (`exclusion_scope="panel"`, the v2 default). Both
the denominator and the excluded gene set depend on panel **size** and **membership**.

Therefore a per-perturbation `pds_cosine` row measured on one panel is **not** the value it
would take on a resampled panel, and any resampling distribution that includes it is
**contaminated**. Every table below is reported twice: a **clean 5-metric** variant
(`de_wilcoxon_sig_jaccard`, `de_wilcoxon_lfc_nmae`,
`de_wilcoxon_direction_fidelity_yield_raw`, `de_wilcoxon_direction_reach_raw`,
`expr_mse_unbiased_capped_norm`) which is valid, and a **contaminated 6-metric** variant
including `pds_cosine` which is reported for continuity with the published `avg_score` and
must not be read as a clean error bar.

### Official-scale wiring, verified

`anchor_local.official_two_ended("experiments/E28-pds/out/agg_v8.parquet")` reproduces
OfficialBaseline's published reference values exactly (`measured`):

| end | V8 `avg_score` | reference | match |
|---|---|---|---|
| `b_lo/r_lo` | 0.1215078150966954 | 0.1215 | yes |
| `b_hi/r_hi` | 0.0848115347085640 | 0.0848 | yes |

V8 per-metric on the official scale, `b_lo/r_lo` / `b_hi/r_hi` (`measured`):

| metric | raw (8-pert) | `b_lo/r_lo` | `b_hi/r_hi` |
|---|---|---|---|
| `pds_cosine` | 0.750000 | 0.585480 | 0.516529 |
| `expr_mse_unbiased_capped_norm` | 1.081883 | 0.000000 | 0.000000 |
| `de_wilcoxon_sig_jaccard` | 0.048234 | 0.076933 | 0.029104 |
| `de_wilcoxon_lfc_nmae` | 1.000361 | 0.000854 | 0.002347 |
| `de_wilcoxon_direction_fidelity_yield_raw` | 0.491874 | -0.045260 | -0.097179 |
| `de_wilcoxon_direction_reach_raw` | 0.148158 | 0.111041 | 0.058068 |
| **`avg_score`** | | **0.121508** | **0.084812** |

`expr_mse_unbiased_capped_norm` reads exactly 0.0000 at both ends because its policy carries
`clamp_low=0.0` and V8's raw 1.0819 exceeds the 1.0 "paste the control unchanged" no-skill
point. That is the catalog's behaviour, not a defect in this run.

### `de_wilcoxon_direction_reach_raw` is hollow, and my first null estimate was wrong by 10x

This section records a correction I introduced and then had to retract, because the retracted
number reached the coordinator and briefly became the project's conclusion.

**What `reach_raw`'s no-skill point actually is.** Not zero, and not a constant.
`cell_eval2/metrics/direction.py:946-949`, verbatim: *"`reach_raw`'s no-skill point is
`~c/N_conf`, NOT a constant, and for a no-skill submission it is decided almost entirely at
the HEAD of a ranking the submission controls. A first-pair match is SUFFICIENT for
`k* >= 1` (purity 1/1 >= P0), one coin flip independent of depth, and it dominates the
no-skill probability."* Their own measurement, `direction.py:965-968`: over 200 replicates
per cell, `E[reach_raw]` runs **0.5100 at `N_conf = 1` to 0.0019 at `N_conf = 500`**, with
`P(reach_raw > 0) = 0.385-0.565` at **every** `N_conf`, and fitted `c ~ 0.96` for a
coin-flip predictor.

So the intuition "a prefix cannot be 90% pure if the ordering is at chance" is false: a
single lucky first pair clears the purity floor, and that event dominates the null mean.
Any argument comparing `reach_raw` against an assumed zero baseline is invalid.

**My error.** I estimated the per-perturbation null as `0.96 / N_conf` using `n_sig` from
`experiments/E10-h1-erp/result.csv` as a stand-in for `N_conf`, and got a panel-mean null of
0.0036, hence "V8's 0.1482 is 41x the null — reach is real skill". That file carries **two
rows per gene**, `mode="downsampled"` (400 cells, gate 10,779) and `mode="native"` (full
cell count, gate 10,780); my dict comprehension silently kept the **last** row, i.e.
`native`. The harness scores 400 downsampled cells (`VCC_PERT_CELLS = 400`), so
`downsampled` is the correct row. For MAT2A the two rows are `n_sig = 5` and `n_sig = 41`.

Corrected (`measured` from the correct rows):

| pert | E10 `n_sig` (downsampled) | measured `N_conf` | `0.96/N_conf` null | measured `reach` |
|---|---|---|---|---|
| MAT2A | 5 | 5 | 0.1920 | **1.0000** |
| ZNF581 | 15 | 16 | 0.0640 | 0.0625 |
| COX4I1 | 171 | 122 | 0.0056 | 0.0574 |
| SLIRP | 259 | 253 | 0.0037 | 0.0435 |
| VCL | 69 | - | 0.0139 | - |
| TCF7L2 | 298 | - | 0.0032 | - |
| GNG12 | 585 | - | 0.0016 | - |
| PAXIP1 | 3,200 | - | 0.0003 | - |

Panel-mean null **0.0355**, not 0.0036. V8's 0.1482 is therefore **4.17x** the null, not
41x. **The 41x figure is retracted.**

Note the direction of the failure: the proxy overstated `N_conf` for exactly the
perturbation where the null goes as `1/N_conf` and `N_conf` is smallest, understating that
cell's null by 53x (0.0036 vs 0.1920). That is precisely the sensitivity I had flagged in
writing when I labelled the proxy `inferred` — the caveat was correct and I was still bitten
by it, because I checked the caveat and not the row selection. The `N_conf`/`k*` columns
themselves are `measured` by SourceUnion, not by me; `de_direction_reach` is declared
`-> dict[str, float]` (`direction.py:862-875`) and exposes neither, and
`direction.py:897-902` records that even the authors' own calibration sweep had to thread its
own `p0` through the private `_k_star` / `_purity_curve` / `_reference_stats` /
`_ontarget_excluded_frame` because the public entry point has no way in. I did not call
those private helpers.

**Why reach is hollow anyway.** The measured decomposition of V8's 0.1482:
MAT2A `N_conf=5`, `k*=5`, `reach=1.0000`, contributing 0.1250 = **84.3% of the panel mean**;
then ZNF581 0.0078, COX4I1 0.0072, SLIRP 0.0054, and VCL/GNG12/PAXIP1/TCF7L2 together
0.0028. Median `k*` is **2**. MAT2A's confident pool contains **zero** of our 288 recruited
genes: its perfect 1.0000 comes from the decoder's near-zero prediction being systematically
slightly negative after mass renormalization, which happens to match all five of its
reference-significant genes, whose real lfc are all negative. That is **mass conservation,
not directional skill**.

Two corroborations from my own corrected proxy, which is the part this slice adds:
**ZNF581 sits exactly on its null** (0.0640 predicted, 0.0625 measured, ratio 0.98) — not
"near chance" but at chance, and it is the largest non-MAT2A contributor. And removing MAT2A
drops the panel mean to `(0.1482*8 - 1.0)/7 = 0.0265` against a null mean over the same
seven of `(0.0355*8 - 0.1920)/7 = 0.0131`, about **2x** — soft, since only three of those
seven have a measured `N_conf`. The jackknife replaces this with an exact public-API
leave-one-out figure.

**Verdict.** `reach_raw` is ~5/6 artifact carried by one perturbation with a 5-gene
confident budget. On the official scale it is worth 0.1110 / 0.0581, of which roughly
0.023 / 0.012 is non-MAT2A content. Combined with `expr_mse_unbiased_capped_norm` being
floored at exactly 0.0000 at both ends, and `nmae`/`fidelity`/`jaccard` being noise (two
negative at the pessimistic end), **V8 scores on `pds_cosine` and essentially nothing else**.

### Scorer configuration (verbatim from `experiments/E28-pds/score_v8.py`)

```python
cfg = EvalConfig.from_preset("vcc2026")
cfg = replace(cfg, pert_col="target_gene", device="cpu")
cfg = replace(cfg, de=replace(cfg.de, backend="scanpy"))
...
score_metrics(wide(agg_pred), wide(agg_base), comparison_statistic="mean")
```

No `anchor=`, no `real_bundle=`, `cfg.num_threads` untouched, no BLAS/OMP environment
variable set.

---

## Disk

Left as found apart from small text artifacts. This experiment currently holds **zero**
`.h5ad`. `experiments/E27-six-metrics/out/real.h5ad` and `agg_base.parquet` were read only
and never written or deleted. One 236 MB `_ctrl.h5ad` was orphaned when the OOM reaper
SIGKILLed the first build between `ControlRef.load` and `build_v8.py:136`'s `unlink`; it has
been deleted.
