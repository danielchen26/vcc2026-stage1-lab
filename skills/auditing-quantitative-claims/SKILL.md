---
name: auditing-quantitative-claims
description: Use when a measurement, benchmark, cross-validation score, effect size, or p-value is about to be used as evidence for a decision — especially when the number supports the hypothesis you were hoping for, when comparing against a published baseline or leaderboard, when a validation split was designed for convenience, when the same quantity has been reported with different values, or when you have been iterating on one evaluation set.
---

# Auditing Quantitative Claims

## Overview

A measurement can be arithmetically correct and still mean nothing. Eighteen recurring failure modes each produced a *plausible, self-consistent, wrong* number that survived casual review.

**Core principle: a number is not evidence until two independent paths reach it.**

## Required Before Reporting Any Number

1. **Independent cross-check.** Reproduce by a second route; report both. Disagreement > 20% means neither is usable yet.
2. **Reproduce the published baseline** before comparing to any published score. Can't hit their baseline → your scale is wrong, not their leaderboard.
3. **Name the deployment boundary** in one sentence; confirm no fold crossed it.
4. **State the aggregation** (mean / median / per-item then averaged), and report both when they diverge.
5. **Show the chance level** next to every overlap, accuracy, or enrichment number.
6. **Show the ablation** that isolates the claimed mechanism.
7. **Count the configurations tried** on this evaluation set.

Any check you skip, say which and why.

## Red Flags — STOP

- "This number confirms the hypothesis" and you have only one estimate of it
- Cross-validation designed around what was convenient to split
- Comparing to a leaderboard without reproducing its baseline
- An improvement stated as a pooled mean/median with no stratification
- Mean and median of the same result differ by >2× and you quote the flattering one
- Counting significant items straight from point estimates + standard errors
- A borderline p after weeks on one evaluation set
- Your only comparator is a trivial strategy
- A causal story that arrived before the isolating measurement
- The same quantity reported with two very different values, and you picked one

**Any of these: read [traps.md](traps.md) before reporting.**

## Rationalizations

| Excuse | Reality |
|--------|---------|
| "The CV split is standard for this data" | Standard ≠ matched to *your* deployment. Leakage cost 50%. |
| "Both are effect sizes, so they're comparable" | Different procedures = different quantities. 26× apart. |
| "The identity is well known" | Every identity has a precondition. Assert it. |
| "Stratifying leaves too few per stratum" | Then you cannot claim the improvement. Say so. |
| "High correlation implies high overlap" | Post-threshold overlap is a different, nonlinear functional. |
| "Prediction accuracy improved, so the decision improves" | Not under asymmetric loss. Optimize the decision. |
| "We beat the baseline, so it works" | Beating a floor proves the floor is low. Ablate. |
| "p < 0.05 after cross-validation" | CV protected the inner parameter, not your outer design search. |
| "One number is enough, the pipeline is tested" | Tested pipelines give correct answers to the wrong question. |

## Reference

[traps.md](traps.md) — 18 failure modes with detection signal, fix, and measured cost. Read it when a red flag fires, when designing a validation split, or when a number disagrees with a published anchor.

## When NOT to Use

Exploratory scans where no decision hangs on the number, and unit tests of deterministic functions with known exact answers.
