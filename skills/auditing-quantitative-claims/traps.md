# The Eighteen Traps

Magnitudes are measured consequences from real projects, not estimates. Grouped by what they corrupt.

## A. Validity — the number answers the wrong question

| # | Trap | Detection signal | Fix |
|---|---|---|---|
| 1 | **Leakage across the deployment boundary.** Folds split on a convenient axis while deployment lacks *all* labels in the target domain. | Ask: "what is unavailable at deployment?" If any fold uses it, the score is fiction. | Split on the deployment boundary (leave-one-domain-out). **Cost: 50% of measured performance; every leak-free variant then failed to beat the trivial baseline.** |
| 2 | **Procedure mismatch.** Substituting a quantity computed by a different procedure for the one the metric defines. | Two sources report the "same" quantity with a large ratio. | The metric's value is defined by the metric's own procedure. Recompute with *that* procedure. **Cost: 26×; caused three successive corrections of one number.** |
| 3 | **Aggregation mismatch.** Comparing your mean against a published median (or vice versa). | Your baseline reproduction misses the published baseline by a suspicious factor, while a *different* aggregation of your own numbers lands inside their interval. | Reproduce the published baseline first; whichever aggregation matches it is the right one. **Cost: 4×, and it reversed which method ranked best.** |
| 11 | **Mechanism attributed without documentation or an isolating measurement.** | A tidy causal story explains the gap; you never read the source's stated method. | Read their published procedure first. A published stratified-sampling description falsified a "selection bias" story that had already been written into three documents. |

## B. Inference — the estimate is contaminated

| # | Trap | Detection signal | Fix |
|---|---|---|---|
| 4 | **Thresholding a noisy estimate.** `|estimate| > threshold` when the estimate has known standard error. | Count of "significant" items ≈ `2·Φ(−t/se) × n_items`. | Deconvolve (empirical Bayes) before counting. **Cost: 981 discoveries, all noise — matching the predicted noise floor of ~1080.** |
| 5 | **Uniform null where nuisance structure exists.** `K/N` as chance overlap when some items are systematically popular. | Null seems implausibly low. | Permutation null: shuffle the labels you care about, preserve every marginal. **Cost: the honest null was 2/3 of the raw signal.** |
| 6 | **Result lands exactly on the analytic null.** | Measured value ≈ computed chance level to 2 digits. | This is an alignment/indexing bug, not a finding. `usecols`-style selectors often return **file order, not requested order** — verify keyed joins actually reordered, and assert in a test that the sources differ in order. |
| 15 | **Selection contamination from iterating on one evaluation set.** | A borderline p (0.01–0.05) after many configurations, no correction, no untouched holdout. | Report how many configurations were tried; reserve a set you never look at. CV protecting one inner parameter does not protect the outer design choices. |
| 18 | **Effective sample size per fitted cell.** Nested structure (folds × bins × items) leaves single digits per estimated parameter. | Divide items by (folds × cells) before trusting any "learned" setting. | Reduce cells, pool, or report the sensitivity curve and pick a flat region instead of an argmax. |

## C. Interpretation — the arithmetic is right, the reading is wrong

| # | Trap | Detection signal | Fix |
|---|---|---|---|
| 7 | **Simpson reversal from pooling.** | Method A wins pooled; you never stratified. | Stratify by the variable driving the metric's scale, then claim. **Cost: A won pooled by 5.4×, lost in the stratum carrying the score.** |
| 8 | **Correlation used as a set-overlap ratio.** | You multiplied a correlation ratio by a set-overlap anchor. | The map is nonlinear in the effect/threshold joint distribution. Measure it or drop the claim. A forward model built to fix this failed too — it assumed independence between effect size and threshold; the measured association was +0.148, and the model overstated the null by 23×. |
| 9 | **Identity applied outside its precondition.** | A closed form relates two metrics; you never checked when it holds. | Derive the precondition and assert it in code. `jac = h/(2−h)` holds only when the predicted and true sets are the same size; a call-everything strategy has `h = 1` and `jac ≈ 0.12`. |
| 16 | **Beating a floor baseline mistaken for the method working.** | Your only comparator is a degenerate/trivial strategy. | Add ablations isolating the *claimed* contribution (adaptive component vs one global setting vs random ranking at the same budget). Ablation ties → contribution is zero. |
| 17 | **Test statistic doesn't match the reported claim.** | A paired rank test is cited to support a ratio of means. | The test endorses only what it tests. Report the estimand the test addresses, plus a bootstrap interval for the one you want to claim. |

## D. Decision — the estimate is fine, the action is wrong

| # | Trap | Detection signal | Fix |
|---|---|---|---|
| 10 | **Optimizing prediction accuracy under asymmetric loss.** | You report RMSE/ρ of an intermediate quantity, then plug the point estimate into the decision. | Minimize expected loss over the decision grid. **Cost: identical information, 2× difference in the final objective.** Check the loss ratio first: one project's was ~150:1, so the optimal policy was strongly biased, not accurate. |

## E. Method — the claim about a technique is wrong

| # | Trap | Detection signal | Fix |
|---|---|---|---|
| 12 | **Distributional claim tested as an algebraic identity.** | An independence/unbiasedness test passes with degenerate inputs. | Independence depends on the input's *variance*, not the algebra: `Cov = Var(estimate) − σ²` is zero only when the input carries its full sampling variance. Simulate under the real generative model. |

## F. Infrastructure — the pipeline lies

| # | Trap | Detection signal | Fix |
|---|---|---|---|
| 13 | **Transient failure read as absent capability.** | Two failures → "blocked/unsupported". | Probe systematically. Intermittent proxies return both 200 and timeouts for one endpoint; a "blocked" verdict was reversed by one systematic sweep. |
| 14 | **Fragile provenance.** Derived artifacts in `/tmp`; two importable scripts sharing a module name. | A rerun fails on a missing file; imports resolve to the wrong module. | Persist derived artifacts beside the data; unique module name per importable script. |

## Auditing Your Own Audit

When building a scenario to check whether someone (or a future you) would catch a trap: **write the scenario before you know which trap it contains, or have someone else write it.**

Verified failure mode — a hand-built scenario written *after* diagnosing a leak stated the leak's two ingredients almost verbatim. The no-guidance control caught it 3/3, making the test worthless for that trap. The same scenario stayed discriminating for a trap that had *not* been telegraphed: 1/3 control vs 3/3 with guidance. **Telegraphing is leakage applied to test design.**

Corollary: run a no-guidance control on every scenario. If the control catches it, you learned nothing about your guidance.

## The Meta-Pattern

Three of the four costliest errors (#1, #2, #3) share one shape: **a quantity was compared against another quantity that was not the same quantity.** Different domains (leakage), different procedures (26×), different aggregations (4×).

Before any comparison, write down what each side literally is — the procedure, the population, the aggregation. If the two sentences differ in any clause, the comparison is invalid until you fix the clause.
