"""Part 2 evidence: is a 2nd/3rd context evaluable locally at all?

Inspects obs of every context .h5ad with h5py only (never touches X), plus our harness
ground truth data/vcc2025/adata_Validation.h5ad, and quantifies the source-side
perturbation pool ceiling.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "E10-h1-erp"))
from run import VCC_PERT_CELLS  # noqa: E402

VCC = Path("/Users/chetianc/vcc2026")
DATA = ROOT / "data"
VAL = DATA / "vcc2025" / "adata_Validation.h5ad"


def cats_of(g):
    """Category labels + code histogram for an AnnData categorical, or raw uniques."""
    if isinstance(g, h5py.Group) and "categories" in g:
        c = [x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]
        codes = g["codes"][:]
        cnt = np.bincount(codes[codes >= 0], minlength=len(c))
        return c, cnt
    v = g[:]
    v = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in v])
    u, cnt = np.unique(v, return_counts=True)
    return list(u), cnt


def dump(path: Path, label: str) -> dict:
    out = {"path": str(path), "label": label}
    print(f"\n{'='*78}\n{label}: {path}\n  size {path.stat().st_size/1e6:.0f} MB\n{'='*78}")
    with h5py.File(path, "r") as f:
        print(f"  root keys: {sorted(f.keys())}")
        if "X" in f:
            X = f["X"]
            if isinstance(X, h5py.Group):
                sh = tuple(X.attrs.get("shape", ()))
                print(f"  X: sparse {X.attrs.get('encoding-type')} shape={sh} "
                      f"nnz={X['data'].shape[0]:,}")
                out["shape"] = list(sh)
            else:
                print(f"  X: dense shape={X.shape}")
                out["shape"] = list(X.shape)
        n_var = None
        if "var" in f:
            print(f"  var keys: {sorted(f['var'].keys())}")
            vi = f["var"].get("_index")
            if isinstance(vi, h5py.Dataset):
                n_var = int(vi.shape[0])
            elif isinstance(vi, h5py.Group) and "categories" in vi:
                n_var = int(vi["codes"].shape[0])
        print(f"  n_var = {n_var}")
        out["n_var"] = n_var
        obs = f["obs"]
        cols = [k for k in obs.keys() if k != "__categories"]
        print(f"  obs columns: {cols}")
        out["obs_columns"] = cols
        n_obs = None
        for k in cols:
            g = obs[k]
            try:
                c, cnt = cats_of(g)
            except Exception as e:                       # noqa: BLE001
                print(f"    - {k}: <unreadable: {e}>")
                continue
            n = int(cnt.sum())
            n_obs = n_obs or n
            print(f"    - {k}: n_obs={n:,} n_categories={len(c)}")
            if len(c) <= 12:
                for lab, q in zip(c, cnt):
                    print(f"        {lab!r:34s} {int(q):>9,}")
            else:
                order = np.argsort(cnt)[::-1]
                print(f"        top 6 of {len(c)}:")
                for i in order[:6]:
                    print(f"        {c[i]!r:34s} {int(cnt[i]):>9,}")
                print(f"        ... e.g. 'non-targeting' present: "
                      f"{'non-targeting' in c}")
                out.setdefault("categories", {})[k] = c
            out.setdefault("ncat", {})[k] = len(c)
        out["n_obs"] = n_obs
    return out


def main() -> None:
    res = {}
    print("### manifest.json ###")
    man = json.loads((VCC / "manifest.json").read_text())
    print(json.dumps(man, indent=2))
    res["manifest"] = man

    for tag in ("A", "B", "C"):
        res[f"context_{tag}"] = dump(VCC / f"context_{tag}.h5ad", f"context_{tag}")
    res["validation"] = dump(VAL, "harness ground truth (adata_Validation)")

    print(f"\n{'='*78}\npert_counts.csv (head)\n{'='*78}")
    pc = pd.read_csv(VCC / "pert_counts.csv")
    print(pc.head(10).to_string())
    print(f"  shape {pc.shape}  columns {list(pc.columns)}")
    res["pert_counts_shape"] = list(pc.shape)
    res["pert_counts_columns"] = list(pc.columns)

    # ---- source-side ceiling: how many perturbations can our harness ever score? -----
    print(f"\n{'='*78}\nsource-side perturbation pool ceiling\n{'='*78}")
    with h5py.File(VAL, "r") as f:
        tg = obs_tg = f["obs"]["target_gene"]
        c, cnt = cats_of(tg)
    tg_counts = dict(zip(c, [int(x) for x in cnt]))
    cats = sorted(set(c) - {"non-targeting"})
    gw_cols = [str(x) for x in pd.read_csv(DATA / "nadig2025" / "K562GW_p.csv.gz",
                                           index_col=0, nrows=1).columns]
    usable = sorted(set(gw_cols) & set(cats))
    cand = [p for p in usable if tg_counts[p] >= VCC_PERT_CELLS]
    print(f"  validation non-control target_gene categories : {len(cats)}")
    print(f"  K562GW source columns (nadig2025)             : {len(gw_cols)}")
    print(f"  intersection (usable)                         : {len(usable)}")
    print(f"  ... and >= {VCC_PERT_CELLS} real cells (cand)          : {len(cand)}")
    print(f"  leaderboard scale (manifest n_constructs)     : {man['n_constructs']}")
    res["ceiling"] = {"val_noncontrol_cats": len(cats), "gw_cols": len(gw_cols),
                      "usable": len(usable), "cand": len(cand),
                      "leaderboard": man["n_constructs"]}

    (Path(__file__).resolve().parent / "context_evidence.json").write_text(
        json.dumps(res, indent=2, default=str))
    print("\nwrote context_evidence.json")


if __name__ == "__main__":
    main()
