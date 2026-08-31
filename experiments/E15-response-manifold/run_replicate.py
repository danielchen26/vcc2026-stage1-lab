"""E15b — 用独立重复当判据：投影到对照流形能否给 beta 去噪。

## E15 的混淆

E15 测得对照 PCA 前 200 维只解释 beta 能量的 12%（随机基线 1.9%，超出 6.4 倍），
判定「响应不在对照变异空间内」。**但这个判定有混淆。**

反推：若 200 个 PC 完美捕获了信号，则 f + 0.0186(1-f) = 0.120 → f = 0.103。
即 400 细胞下的 beta **最多只有 10% 是信号，90% 是噪声**。
任何固定 k 维基对各向同性噪声只能抓 k/G，所以低 R^2 分不清
「响应不在流形上」和「beta 本身是噪声」。

## 无混淆的判据：独立重复

把每个扰动的细胞对半劈开：
    beta_A、beta_B  互相独立（噪声独立，信号相同）
    corr(beta_A, beta_B)                = 原始可靠性（基线）
    corr(proj_k(beta_A), beta_B)        = 投影后的可靠性

**投影只能扔掉方向，不能增加信息。所以若投影后相关上升，
说明扔掉的是噪声方向 —— 流形是有用的去噪器。** 这与 R^2 无关，不受上面的混淆影响。

同时报操作层面的量（真正决定分数的）：
    用 beta_A 排序召集 vs 用 proj_k(beta_A) 排序召集，
    对靶 = 半 B 独立算出的 R_p，比 h 谁高。

若成立，这是一个**完全不需要跨系数据**的算法收益：靶 context 自己的 18,400 个
对照细胞就能构造去噪器，直接作用在任何来源的 beta 上。

跑法：  ~/vcc2026/.venv/bin/python experiments/E15-response-manifold/run_replicate.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr, wilcoxon
from sklearn.utils.extmath import randomized_svd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vcclab.scorer import ControlRef  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "E10-h1-erp"))
from run import VCC_CTRL_CELLS, VCC_PERT_CELLS, VCC_UMI, read_rows, thin, to_cpm  # noqa: E402


def logcpm(M: sp.csr_matrix, gidx: np.ndarray) -> np.ndarray:
    """行归一到 1e6 后 log1p，只取 gate 内基因。（与 run.py 同实现；
    两个实验的文件同名 run.py，不能跨目录 import，故内联。）"""
    A = np.asarray(M[:, gidx].todense(), dtype=np.float32)
    s = np.asarray(M.sum(1), dtype=np.float32)
    np.divide(A, np.maximum(s, 1.0), out=A)
    A *= 1e6
    return np.log1p(A, out=A)

H5 = ROOT / "data" / "vcc2025" / "adata_Validation.h5ad"
OUT = Path(__file__).parent
KS = (5, 10, 20, 50, 100, 200, 400)
N_PC = 400
ALPHA = 0.05
SEED = 0
EPS = 1.0


def main() -> None:
    t0 = time.time()
    print("=== E15b 投影到对照流形能否给 beta 去噪（用独立重复判定）===\n")

    with h5py.File(H5, "r") as f:
        genes = [g.decode() if isinstance(g, bytes) else str(g)
                 for g in f["var"]["_index"][:]]
        o = f["obs"]["target_gene"]
        cats = [c.decode() if isinstance(c, bytes) else str(c)
                for c in o["categories"][:]]
        codes = o["codes"][:]
    ntc = cats.index("non-targeting")
    rng = np.random.default_rng(SEED)

    rows = rng.choice(np.flatnonzero(codes == ntc), VCC_CTRL_CELLS, replace=False)
    ctrl_raw = thin(read_rows(rows), VCC_UMI, rng)
    p_tmp = OUT / "_ntc_rep.h5ad"
    ad.AnnData(X=sp.csr_matrix(ctrl_raw),
               var=pd.DataFrame(index=pd.Index(genes))).write_h5ad(p_tmp)
    ref = ControlRef.load(p_tmp, genes)
    gidx = np.asarray(ref.gidx)
    G = len(gidx)

    C = logcpm(ctrl_raw, gidx)
    C -= C.mean(0)
    _, _, Vt = randomized_svd(C, n_components=N_PC, random_state=SEED)
    V = Vt.T.astype(np.float32)
    del C
    ctrl_cpm = np.asarray(ctrl_raw[:, gidx].multiply(
        1e6 / np.maximum(np.asarray(ctrl_raw.sum(1)), 1.0)).mean(0)).ravel()
    print(f"gate = {G:,}   对照 PCA {N_PC} 维完成")

    def beta_of(sel: np.ndarray) -> np.ndarray:
        M = thin(read_rows(sel), VCC_UMI, rng)
        pm = np.asarray(M[:, gidx].multiply(
            1e6 / np.maximum(np.asarray(M.sum(1)), 1.0)).mean(0)).ravel()
        return np.log2((pm + EPS) / (ctrl_cpm + EPS)).astype(np.float32)

    need = 2 * VCC_PERT_CELLS
    print(f"\n只用细胞数 ≥ {need} 的扰动（两半各 {VCC_PERT_CELLS}）")
    print(f"{'扰动':>10} {'|R_p(B)|':>9} {'r 原始':>8} " +
          " ".join(f"r k={k:<4d}" for k in KS))
    recs = []
    for ci, name in enumerate(cats):
        if ci == ntc:
            continue
        idx = np.flatnonzero(codes == ci)
        if len(idx) < need:
            continue
        pick = rng.permutation(idx)[:need]
        a_sel, b_sel = pick[:VCC_PERT_CELLS], pick[VCC_PERT_CELLS:]
        ba, bb = beta_of(a_sel), beta_of(b_sel)
        # 半 B 的官方真集
        padj, _ = ref.de_table(
            to_cpm(thin(read_rows(b_sel), VCC_UMI, rng)), tie_correct=True)
        real = padj < ALPHA
        n_real = int(real.sum())
        if n_real < 5:
            continue

        r_raw = float(pearsonr(ba, bb).statistic)
        rec = dict(target_gene=name, n_real=n_real, r_raw=r_raw)
        # 原始 beta 的召集（按 |beta| 降序，等规模）
        h_raw = float(real[np.argsort(np.abs(ba))[::-1][:n_real]].sum()) / n_real
        rec["h_raw"] = h_raw
        for k in KS:
            proj = V[:, :k] @ (V[:, :k].T @ ba)
            rec[f"r_k{k}"] = float(pearsonr(proj, bb).statistic)
            rec[f"h_k{k}"] = float(
                real[np.argsort(np.abs(proj))[::-1][:n_real]].sum()) / n_real
        recs.append(rec)
        print(f"{name:>10} {n_real:9d} {r_raw:8.3f} " +
              " ".join(f"{rec[f'r_k{k}']:7.3f}" for k in KS))

    df = pd.DataFrame(recs)
    df.to_csv(OUT / "result_replicate.csv", index=False)
    p_tmp.unlink(missing_ok=True)

    print(f"\n{'='*80}\n结论（n = {len(df)} 个扰动）\n{'='*80}")
    print(f"{'':>10} {'相关 r(·, beta_B) 中位':>22} {'h 中位':>10} {'vs 原始 h':>12}")
    print(f"{'原始 beta':>10} {df.r_raw.median():22.3f} {df.h_raw.median():10.3f} "
          f"{'—':>12}")
    best_k, best_h = None, df.h_raw.median()
    for k in KS:
        rm, hm = df[f"r_k{k}"].median(), df[f"h_k{k}"].median()
        mark = ""
        if hm > best_h:
            best_h, best_k, mark = hm, k, "  ←"
        print(f"{'投影 k='+str(k):>10} {rm:22.3f} {hm:10.3f} "
              f"{hm/df.h_raw.median():11.2f}×{mark}")

    print(f"\n--- 配对检验（投影 vs 原始）---")
    for k in KS:
        dr = df[f"r_k{k}"] - df.r_raw
        dh = df[f"h_k{k}"] - df.h_raw
        pr = wilcoxon(df[f"r_k{k}"], df.r_raw).pvalue if dr.abs().sum() > 0 else 1.0
        ph = wilcoxon(df[f"h_k{k}"], df.h_raw).pvalue if dh.abs().sum() > 0 else 1.0
        print(f"  k={k:<4d} Δr 中位 {dr.median():+.4f} (p={pr:.4f})   "
              f"Δh 中位 {dh.median():+.4f} (p={ph:.4f})   h 胜 {(dh>0).sum()}/{len(df)}")

    if best_k is not None:
        dh = df[f"h_k{best_k}"] - df.h_raw
        p = wilcoxon(df[f"h_k{best_k}"], df.h_raw).pvalue
        good = p < 0.05
        print(f"\n>>> {'投影去噪有效' if good else '投影去噪不显著'}："
              f"最佳 k={best_k}，h 从 {df.h_raw.median():.3f} 提到 {best_h:.3f}"
              f"（{best_h/df.h_raw.median():.2f}×，p={p:.4f}）<<<")
    else:
        print(f"\n>>> 投影去噪无效：任何 k 都没提高 h <<<")
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
