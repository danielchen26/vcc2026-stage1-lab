"""Generate the E30 builds from E28's build_v8.py by explicit, asserted str.replace.

Finding F29 discipline: a variant build MUST be `build_v8.py` text plus a single
`str.replace` for the KNOB, with an assert that the replacement landed. This module
separates the two kinds of edit so the attribution is auditable:

  HARNESS  plumbing only. Redirects OUT into this experiment, lets one invocation write
           one tag, drops the blocks a pass does not write (RAM), and checks the disk
           floor immediately before each write. NONE of these touches ordering, K,
           lambda, gene selection, or any number that enters a prediction.
  KNOB     the one modelling decision a variant changes.

Two outputs:
  build_c1.py     HARNESS only. N_PERT stays 8, so this REPRODUCES V8 exactly; it exists
                  to obtain the per-perturbation metric rows that V8's original run
                  aggregated away. Verified by requiring its aggregate to equal
                  experiments/E28-pds/out/agg_v8.parquet.
  build_scale.py  HARNESS + the N_PERT 8 -> 24 knob. Generated and syntax-validated, but
                  NOT RUN: abandoned on the machine's memory ceiling, and separately
                  superseded because resampling 8-of-24 answers a worse question than
                  bootstrapping the 8 rows directly (the two panels are different
                  samples, not nested ones -- see panel_check.py).

Why `X_real`/`X_base` may be dropped without disturbing reproducibility: inside the
per-perturbation loop, pred uses `design_cells(..., seed=SEED)` with a FIXED seed and base
uses only the deterministic `hamilton(base_row)`; neither reads the shared `rng`. Only the
real branch draws from `rng`. The _Sink below therefore still CONSTRUCTS every block, so
the rng stream is consumed identically no matter which tag is written -- it only refuses to
retain the two it will not write. A pass that writes `real` is bit-identical to `real` from
a three-way pass.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "experiments" / "E28-pds" / "build_v8.py"

# ---- HARNESS: plumbing, zero modelling effect --------------------------------------
HARNESS = [
    ("harness:OUT",
     'OUT = Path(__file__).resolve().parents[1] / "E28-pds" / "out"',
     'OUT = Path(__file__).resolve().parent / "out"'),
    # Drop the blocks this pass will not write: peak RSS ~1.6 GB -> ~0.6 GB. `sys` is
    # already imported at module scope. Tag names carry N_PERT so two panel sizes cannot
    # collide in one out/ directory.
    ("harness:sink",
     "    X_pred, X_real, X_base, obs_p, obs_r = [], [], [], [], []",
     '    _sel = sys.argv[2:] or ["pred", "real", "base"]\n'
     "\n"
     "    class _Sink(list):\n"
     '        """Discards what it is given: the tags this invocation does not write."""\n'
     "\n"
     "        def append(self, x):\n"
     "            return None\n"
     "\n"
     '    X_pred = [] if "pred" in _sel else _Sink()\n'
     '    X_real = [] if "real" in _sel else _Sink()\n'
     '    X_base = [] if "base" in _sel else _Sink()\n'
     "    obs_p, obs_r = [], []"),
    ("harness:write-one",
     '    for tag, blocks in (("pred_v8", X_pred),):',
     '    for tag, blocks in [(f"{t}_{N_PERT}", bl) for t, bl in\n'
     '                        (("pred", X_pred), ("real", X_real), ("base", X_base))\n'
     "                        if t in _sel]:"),
    # The floor is checked HERE, not at launch: siblings can eat 900 MB during the ~6 min
    # of source-side compute that precedes this line.
    ("harness:dfguard",
     '        a.write_h5ad(OUT / f"{tag}.h5ad", compression="gzip")',
     "        import shutil\n"
     '        _free = shutil.disk_usage("/System/Volumes/Data").free / 1e9\n'
     '        print(f"  df immediately before writing {tag}: {_free:.1f} GB free")\n'
     '        assert _free > 5.0, f"ABORT pre-write: {_free:.1f} GB free (<5 GB floor)"\n'
     '        a.write_h5ad(OUT / f"{tag}.h5ad", compression="gzip")'),
]

# ---- KNOBS: one modelling decision each --------------------------------------------
KNOB_N24 = ("knob:N_PERT", "N_PERT = 8", "N_PERT = 24")

TARGETS = {
    "build_c1.py": [],            # harness only -> reproduces V8 at N_PERT=8
    "build_scale.py": [KNOB_N24],  # harness + the one knob
}

# V8's three modelling knobs, asserted verbatim in every generated file.
PINNED = ("LAMBDA = 0.7", "K = 288",
          "score = np.where(good, np.abs(b), -np.inf)")


def apply(text: str, edits) -> str:
    for label, old, new in edits:
        assert text.count(old) == 1, \
            f"{label}: {old!r} not unique ({text.count(old)}x)"
        text = text.replace(old, new)
        # An edit that PREPENDS a guard and keeps the original line reintroduces `old`.
        # The invariant is therefore "exactly as many occurrences as `new` puts back":
        # 0 for a true substitution, 1 for a wrapping insert.
        expect = new.count(old)
        assert text.count(old) == expect, \
            f"{label}: {old!r} occurs {text.count(old)}x, expected {expect}x"
        assert new in text, f"{label}: new string {new!r} absent"
        print(f"    ok {label}")
    return text


def main() -> None:
    src = SRC.read_text()
    for name, knobs in TARGETS.items():
        print(f"{name}  (harness {len(HARNESS)} + knobs {len(knobs)}):")
        text = apply(src, HARNESS + knobs)
        for pinned in PINNED:
            assert pinned in text, f"{name}: V8 knob lost: {pinned!r}"
        if not knobs:
            assert "N_PERT = 8" in text, f"{name}: must stay at N_PERT = 8"
        (HERE / name).write_text(text)
        print(f"    wrote {name} ({len(text)} chars, src {len(src)})\n")


if __name__ == "__main__":
    main()
