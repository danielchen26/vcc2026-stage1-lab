"""Cheap pre-flight: is the N_PERT=24 stratified panel a superset of the N_PERT=8 one?

Reproduces build_v8.py:147-154 verbatim (the `pick` computation) without touching any
matrix, so the answer is known before a ~2.6 GB build is started.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "E10-h1-erp"))
from run import VCC_PERT_CELLS  # noqa: E402

DATA = ROOT / "data"
H5 = DATA / "vcc2025" / "adata_Validation.h5ad"


def as_str(g):
    if isinstance(g, h5py.Group) and "categories" in g:
        c = [x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]
        return np.asarray(c)[g["codes"][:]]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g[:]])


def main() -> None:
    with h5py.File(H5, "r") as f:
        tg_all = as_str(f["obs"]["target_gene"])
    cats = sorted(set(tg_all) - {"non-targeting"})
    gw_cols = [str(c) for c in pd.read_csv(DATA / "nadig2025" / "K562GW_p.csv.gz",
                                           index_col=0, nrows=1).columns]
    usable = sorted(set(gw_cols) & set(cats))
    real_n = {p: int((tg_all == p).sum()) for p in usable}
    cand = [p for p in usable if real_n[p] >= VCC_PERT_CELLS]
    cand.sort(key=lambda p: real_n[p])
    print(f"len(cand) = {len(cand)}")

    def pick_at(n):
        idx = np.linspace(0, len(cand) - 1, n).round()
        return [cand[int(i)] for i in idx], [int(i) for i in idx]

    p8, i8 = pick_at(8)
    p24, i24 = pick_at(24)
    print(f"\nN=8  indices {i8}\n     {p8}")
    print(f"\nN=24 indices {i24}\n     {p24}")
    missing = [p for p in p8 if p not in p24]
    print(f"\nsuperset? {'YES' if not missing else 'NO'}   missing from 24-panel: {missing}")
    print(f"overlap = {len(set(p8) & set(p24))}/8")
    on_disk = pd.read_csv(ROOT / "experiments" / "E28-pds" / "out" / "perts.csv",
                          header=None)[0].tolist()
    print(f"\nE28 perts.csv on disk: {on_disk}")
    print(f"reproduces N=8 pick? {on_disk == p8}")
    print("\nreal cell counts (24-panel):")
    for p in p24:
        print(f"  {p:12s} {real_n[p]:6d}{'   <- also in 8-panel' if p in p8 else ''}")


if __name__ == "__main__":
    main()
