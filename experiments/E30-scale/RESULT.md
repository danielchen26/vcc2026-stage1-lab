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

**Status: PENDING.** Queued behind the machine's memory ceiling by coordinator decision; the
two heavy `compute_metrics` slots were occupied and `vm.swapusage` free was 540 MB against a
2 GB launch gate. Numbers and the "0.2025 +/- what" line land here when the run completes.
Nothing in this section is inferred in the meantime.

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
