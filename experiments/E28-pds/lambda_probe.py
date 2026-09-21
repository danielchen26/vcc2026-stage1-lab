"""lambda 收缩扫描：pds 用已验证的 harness，expr_mse 用 pseudobulk 上的局部代理。

动机：V5 把排序换成 |beta| 后 avg_score 0.1110 -> 0.1559，唯一代价是
expr_mse_unbiased_capped_norm 从 0.5219 掉到 0.3930（-0.1289）。
而 pds_cosine 的余弦**对尺度不变**（log1p 让它只是近似不变），
所以收缩 lambda 应当能救回 expr_mse 而几乎不动 pds —— 这是本脚本要验证的。

expr_mse 代理：在 bulk_lognorm 空间上 mean((pred_eff - real_eff)^2)。
官方 expr_mse_unbiased 还做了无偏校正与钳位，故代理只用于**设计间相对比较**，
绝对值不可与官方数字比（这正是 F29 里"拿不同量作比较"的陷阱）。
"""
from __future__ import annotations
import importlib.util, sys, time
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, h5py

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
        genes = np.array([x.decode() if isinstance(x, bytes) else str(x)
                          for x in f["var"]["_index"][:]])
        cats = np.asarray([x.decode() if isinstance(x, bytes) else str(x)
                           for x in f["obs"]["target_gene"]["categories"][:]])
        tg_all = cats[f["obs"]["target_gene"]["codes"][:]]
    rng = np.random.default_rng(SEED)
    ntc = rng.choice(np.flatnonzero(tg_all == "non-targeting"), VCC_CTRL_CELLS, replace=False)
    ctrl = thin(read_rows(ntc), VCC_UMI, rng)
    tmp = Path(__file__).parent / "_ctrl_lam.h5ad"
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

    real = ad.read_h5ad(E27 / "out" / "real.h5ad")
    bts = float(H.vcc_cfg().bulk_target_sum)
    rb = H.pseudobulk_bulk_lognorm(real, H.PERT_COL, bulk_target_sum=bts)
    rl, rm = np.asarray(rb[0]).astype(str), np.asarray(rb[1], dtype=np.float64)
    rci = int(np.flatnonzero(rl == H.CTRL)[0])
    real_eff = {p: rm[int(np.flatnonzero(rl == p)[0])] - rm[rci] for p in perts}
    print(f"gate {ref.G:,}  共同 {len(common):,}  载入 {time.time()-t0:.0f}s")

    ctrl_csr = sp.csr_matrix(ctrl)
    print(f"\n{'lambda':>7s} {'pds':>8s} {'pds_fb':>8s} {'MSE 代理':>11s} {'相对 V5':>9s}")
    print("-" * 50)
    base_mse = None
    for lam in (1.0, 0.7, 0.5, 0.3, 0.15):
        X, obs = [], []
        for j, p in enumerate(perts):
            b, s = B[:, j], S[:, j]
            good = np.isfinite(b) & np.isfinite(s) & (s > 0)
            sc = np.where(good, np.abs(b), -np.inf)          # V5 的排序
            pc = P[:, j]; fin = np.isfinite(pc)
            n_src = int((bh_adjust(pc[fin]) < ALPHA).sum()) if fin.any() else 0
            K = 29 if n_src < 5 else (288 if n_src < 100 else ref.G)
            sel = np.argsort(sc)[::-1][:min(K, int(good.sum()))]
            X.append(sp.csr_matrix(design_cells(ref, gi_g[sel], lam * b[sel],
                                               n_cells=VCC_PERT_CELLS, seed=SEED)))
            obs += [p] * VCC_PERT_CELLS
        Xs = sp.vstack(X + [ctrl_csr], format="csr")
        obs += ["non-targeting"] * ctrl_csr.shape[0]
        a = ad.AnnData(X=Xs, obs=pd.DataFrame({"target_gene": obs},
                       index=[str(i) for i in range(len(obs))]),
                       var=pd.DataFrame(index=pd.Index(genes)))
        pb = H.pseudobulk_bulk_lognorm(a, H.PERT_COL, bulk_target_sum=bts)
        got = H.pds(pb, rb, genes); v = float(np.mean(list(got.values())))
        pl_, pm = np.asarray(pb[0]).astype(str), np.asarray(pb[1], dtype=np.float64)
        pci = int(np.flatnonzero(pl_ == H.CTRL)[0])
        mse = float(np.mean([np.mean((pm[int(np.flatnonzero(pl_ == p)[0])] - pm[pci]
                                      - real_eff[p]) ** 2) for p in perts]))
        if base_mse is None: base_mse = mse
        print(f"{lam:7.2f} {v:8.4f} {2*(v-0.5):+8.4f} {mse:11.6f} {mse/base_mse:9.4f}")
    print(f"\n总耗时 {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
