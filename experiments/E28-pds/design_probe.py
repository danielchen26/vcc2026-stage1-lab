"""只改 lfc 赋值规则（排序统计量 + K），管线逐字照 A，用已验证的 harness 本地测 pds。

为什么这样做：V2/V3 都因为同时改了生成模型而无法归因；V4 证明量化器毫无影响
（0.1109 vs A 的 0.1110）。真正与扫描里 0.6607 有差别的只剩排序统计量和 K。
harness 已独立验证两次（A 上 1.1e-16、V3 上精确 0.500000），只吃 pseudobulk、不碰 DE。
"""
from __future__ import annotations
import importlib.util, sys, time
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, h5py
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "E10-h1-erp"))
from vcclab.decoder import design_cells                      # noqa: E402
from vcclab.scorer import ControlRef, bh_adjust              # noqa: E402
from run import VCC_CTRL_CELLS, VCC_PERT_CELLS, VCC_UMI, read_rows, thin  # noqa: E402

E27 = ROOT / "experiments" / "E27-six-metrics"
DATA, SEED, ALPHA = ROOT / "data", 0, 0.05
spec = importlib.util.spec_from_file_location("e28", Path(__file__).parent / "run.py")
H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)


def main():
    t0 = time.time()
    perts = [l.strip() for l in (E27 / "out" / "perts.csv").read_text().split() if l.strip()]
    with h5py.File(DATA / "vcc2025" / "adata_Validation.h5ad", "r") as f:
        genes = H._load_beta.__globals__["np"].array(
            [x.decode() if isinstance(x, bytes) else str(x) for x in f["var"]["_index"][:]])
        tg_all = np.asarray([x.decode() if isinstance(x, bytes) else str(x)
                             for x in f["obs"]["target_gene"]["categories"][:]])[
            f["obs"]["target_gene"]["codes"][:]]
    rng = np.random.default_rng(SEED)
    ntc_rows = rng.choice(np.flatnonzero(tg_all == "non-targeting"), VCC_CTRL_CELLS, replace=False)
    ctrl = thin(read_rows(ntc_rows), VCC_UMI, rng)
    tmp = Path(__file__).parent / "_ctrl_probe.h5ad"
    ad.AnnData(X=sp.csr_matrix(ctrl), var=pd.DataFrame(index=pd.Index(genes))).write_h5ad(tmp)
    ref = ControlRef.load(tmp, list(genes)); tmp.unlink(missing_ok=True)
    gidx = np.asarray(ref.gidx)

    e2s = pd.read_csv(DATA / "external" / "ens2sym.csv").dropna()
    def rd(n):
        d = pd.read_csv(DATA / "nadig2025" / f"{n}.csv.gz", index_col=0,
                        usecols=["Unnamed: 0"] + perts, dtype={c: np.float32 for c in perts})
        d.index = d.index.astype(str); return d[perts]
    lfc_s, se_s, p_s = rd("K562GW_lfc"), rd("K562GW_se"), rd("K562GW_p")
    se_s, p_s = se_s.reindex(index=lfc_s.index), p_s.reindex(index=lfc_s.index)
    sym = pd.Index(lfc_s.index.map(dict(zip(e2s.ensembl, e2s.symbol))))
    k = pd.notna(sym); lfc_s, se_s, p_s, sym = lfc_s[k], se_s[k], p_s[k], sym[k]
    dd = ~sym.duplicated(); lfc_s, se_s, p_s, sym = lfc_s[dd], se_s[dd], p_s[dd], sym[dd]
    common = pd.Index(genes[gidx]).intersection(sym)
    gi_g = pd.Index(genes[gidx]).get_indexer(common); gi_s = sym.get_indexer(common)
    B = lfc_s.to_numpy(dtype=np.float64)[gi_s]
    S = se_s.to_numpy(dtype=np.float64)[gi_s]
    P = p_s.to_numpy(dtype=np.float64)[gi_s]
    print(f"gate {ref.G:,}  共同基因 {len(common):,}  载入 {time.time()-t0:.0f}s")

    real = ad.read_h5ad(E27 / "out" / "real.h5ad")
    bts = float(H.vcc_cfg().bulk_target_sum)
    real_bulk = H.pseudobulk_bulk_lognorm(real, H.PERT_COL, bulk_target_sum=bts)

    ctrl_csr = sp.csr_matrix(ctrl)
    def probe(stat: str, kmode: str):
        X, obs = [], []
        for j, p in enumerate(perts):
            b, s = B[:, j], S[:, j]
            good = np.isfinite(b) & np.isfinite(s) & (s > 0)
            r = (np.abs(b) / np.maximum(s, 1e-9)) if stat == "z" else np.abs(b)
            r = np.where(good, r, -np.inf)
            pc = P[:, j]; fin = np.isfinite(pc)
            n_src = int((bh_adjust(pc[fin]) < ALPHA).sum()) if fin.any() else 0
            K = 288 if kmode == "flat" else (29 if n_src < 5 else (288 if n_src < 100 else ref.G))
            sel = np.argsort(r)[::-1][:min(K, int(good.sum()))]
            X.append(sp.csr_matrix(design_cells(ref, gi_g[sel], b[sel],
                                               n_cells=VCC_PERT_CELLS, seed=SEED)))
            obs += [p] * VCC_PERT_CELLS
        Xs = sp.vstack(X + [ctrl_csr], format="csr")
        obs += ["non-targeting"] * ctrl_csr.shape[0]
        a = ad.AnnData(X=Xs, obs=pd.DataFrame({"target_gene": obs},
                       index=[str(i) for i in range(len(obs))]),
                       var=pd.DataFrame(index=pd.Index(genes)))
        got = H.pds(H.pseudobulk_bulk_lognorm(a, H.PERT_COL, bulk_target_sum=bts),
                    real_bulk, genes)
        return float(np.mean(list(got.values()))), got

    print(f"\n{'排序统计量':14s} {'K 规则':10s} {'pds':>8s} {'from_base':>10s}")
    print("-" * 48)
    for stat, lbl in (("z", "|beta|/SE"), ("abs", "|beta|")):
        for kmode, klbl in (("prior", "29/288/G"), ("flat", "288 全部")):
            v, per = probe(stat, kmode)
            mark = "  ← A 的设计" if (stat, kmode) == ("z", "prior") else ""
            print(f"{lbl:14s} {klbl:10s} {v:8.4f} {2*(v-0.5):+10.4f}{mark}")
    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
