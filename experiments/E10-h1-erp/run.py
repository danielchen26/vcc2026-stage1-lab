"""E10 — 用主办方自己的 2025 H1 数据实测 E|R_p|，终结 11 / 36 / 288 之争。

## 为什么只有这个数据能回答

三个观测跨了 26 倍，而 E|R_p| 是整个框架最大的杠杆（决定 K_p）：

    K562 GWPS 经验贝叶斯     11      tau/se=2.49 良态，可信
    K562 MASH（外部）         7      作者功效校正
    CD4+T Rest MASH（外部）  36      512 细胞/扰动，功效与本届可比
    F8 从官方 jac 基线反推   288     **欠定**（假设基线报满 9,863 个基因）

2025 H1 是唯一同时满足以下条件的公开数据：
  - **同一批主办方、同一套 pilot-screen 分层选基因流程**（按 DE 基因数分箱，
    negligible→strong 全谱采样，所以是无偏抽样而非「响应者集合」）
  - **同一套实验协议**：dual-guide CRISPRi + 10x Flex
  - **基因面板几乎相同**：18,080 vs 本届 18,533，共享 18,076
  - 深度是本届的 7.4 倍（59.35 M vs 8.00 M UMI/扰动）→ **可以降采样到本届功效**

## 降采样到本届的确切条件

| 量 | 本届 2026 | H1 2025 原始 | 本实验 |
|---|---|---|---|
| 对照细胞 | 18,400 | 38,176 | **抽 18,400** |
| 每扰动细胞 | 400 | 中位 1,090 | **抽 400**（不足者取全部） |
| UMI/细胞 | 20,000 | 54,447 | **二项稀释到 20,000** |

同时跑一版**不降采样**的（H1 原生功效），用来和 CD4+T 的 36 对照。

DE 用 `vcclab.scorer.ControlRef.de_table` —— 已由 E04 逐基因验证等于官方
`cell-eval2` 0.16.0 preset `vcc2026`（显著集对称差 0/0/0，需并列校正）。

跑法：  ~/vcc2026/.venv/bin/python experiments/E10-h1-erp/run.py
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vcclab.scorer import ControlRef  # noqa: E402

H5 = ROOT / "data" / "vcc2025" / "adata_Validation.h5ad"
OUT = Path(__file__).parent
VCC_CTRL_CELLS = 18_400
VCC_PERT_CELLS = 400
VCC_UMI = 20_000
ALPHA = 0.05
SEED = 0


def load_obs() -> tuple[np.ndarray, list[str]]:
    with h5py.File(H5, "r") as f:
        o = f["obs"]["target_gene"]
        cats = [c.decode() if isinstance(c, bytes) else str(c)
                for c in o["categories"][:]]
        codes = o["codes"][:]
    return codes, cats


def read_rows(rows: np.ndarray) -> sp.csr_matrix:
    """按行读 CSR 的一个子集，不把整个矩阵读进内存。"""
    rows = np.sort(np.asarray(rows))
    with h5py.File(H5, "r") as f:
        X = f["X"]
        indptr = X["indptr"][:]
        n_genes = int(X.attrs["shape"][1])
        starts, ends = indptr[rows], indptr[rows + 1]
        counts = (ends - starts).astype(np.int64)
        data = np.empty(int(counts.sum()), np.float32)
        idx = np.empty(int(counts.sum()), np.int32)
        pos = 0
        for s, e, n in zip(starts, ends, counts):
            if n:
                data[pos:pos + n] = X["data"][s:e]
                idx[pos:pos + n] = X["indices"][s:e]
            pos += int(n)
    new_ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return sp.csr_matrix((data, idx, new_ptr), shape=(len(rows), n_genes))


def thin(M: sp.csr_matrix, target_umi: int, rng: np.random.Generator
         ) -> sp.csr_matrix:
    """二项稀释到每细胞 target_umi。已低于目标的细胞不动。"""
    M = M.tocsr().astype(np.float32)
    tot = np.asarray(M.sum(1)).ravel()
    p = np.clip(target_umi / np.maximum(tot, 1.0), 0.0, 1.0)
    rep = np.diff(M.indptr)
    pr = np.repeat(p, rep)
    M.data = rng.binomial(M.data.astype(np.int64), pr).astype(np.float32)
    M.eliminate_zeros()
    return M


def to_cpm(M: sp.csr_matrix) -> np.ndarray:
    """转成行和恰为 1e6 的稠密 CPM —— de_table 的输入契约。"""
    A = np.asarray(M.todense(), dtype=np.float64)
    s = A.sum(1, keepdims=True)
    return np.divide(A, s, out=np.zeros_like(A), where=s > 0) * 1e6


def run(downsample: bool, rng: np.random.Generator, genes: list[str]) -> pd.DataFrame:
    tag = "降采样到本届" if downsample else "H1 原生功效"
    print(f"\n{'='*62}\n{tag}\n{'='*62}")
    codes, cats = load_obs()
    ntc_code = cats.index("non-targeting")

    ntc_rows = np.flatnonzero(codes == ntc_code)
    if downsample and len(ntc_rows) > VCC_CTRL_CELLS:
        ntc_rows = rng.choice(ntc_rows, VCC_CTRL_CELLS, replace=False)
    t = time.time()
    ctrl = read_rows(ntc_rows)
    if downsample:
        ctrl = thin(ctrl, VCC_UMI, rng)
    print(f"对照 {ctrl.shape[0]:,} 细胞  中位 UMI {np.median(np.asarray(ctrl.sum(1))):,.0f}"
          f"  ({time.time()-t:.0f}s)")

    ref_path = OUT / f"_ntc_{'ds' if downsample else 'full'}.h5ad"
    ad.AnnData(X=sp.csr_matrix(ctrl),
               var=pd.DataFrame(index=pd.Index(genes))).write_h5ad(ref_path)
    ref = ControlRef.load(ref_path, genes)
    print(f"ControlRef: gate={ref.G:,} 基因  对照={ref.n_ctrl:,} 细胞")

    rows = []
    for ci, name in enumerate(cats):
        if ci == ntc_code:
            continue
        idx = np.flatnonzero(codes == ci)
        n_avail = len(idx)
        if downsample and n_avail > VCC_PERT_CELLS:
            idx = rng.choice(idx, VCC_PERT_CELLS, replace=False)
        M = read_rows(idx)
        if downsample:
            M = thin(M, VCC_UMI, rng)
        padj, lfc = ref.de_table(to_cpm(M), tie_correct=True)
        n_sig = int((padj < ALPHA).sum())
        rows.append(dict(target_gene=name, n_cells_used=len(idx),
                         n_cells_avail=n_avail, n_sig=n_sig,
                         median_abs_lfc=float(np.median(np.abs(lfc)))))
        print(f"  {name:12s} {len(idx):5d}/{n_avail:5d} 细胞  |R_p| = {n_sig:5d}")
    df = pd.DataFrame(rows)
    ref_path.unlink(missing_ok=True)

    q = np.percentile(df.n_sig, [25, 50, 75, 90])
    print(f"\n{tag}的 |R_p|: 中位 {q[1]:.0f}  IQR [{q[0]:.0f}, {q[2]:.0f}]  "
          f"p90 {q[3]:.0f}  范围 [{df.n_sig.min()}, {df.n_sig.max()}]  gate={ref.G:,}")
    df["mode"] = "downsampled" if downsample else "native"
    df["gate"] = ref.G
    return df


def main() -> None:
    t0 = time.time()
    print("=== E10 用主办方自己的 2025 H1 数据实测 E|R_p| ===")
    with h5py.File(H5, "r") as f:
        genes = [g.decode() if isinstance(g, bytes) else str(g)
                 for g in f["var"]["_index"][:]]
    print(f"H1 validation: 98,927 细胞 × {len(genes):,} 基因")

    rng = np.random.default_rng(SEED)
    ds = run(True, rng, genes)
    nat = run(False, np.random.default_rng(SEED), genes)
    out = pd.concat([ds, nat], ignore_index=True)
    out.to_csv(OUT / "result.csv", index=False)

    md, mn = float(ds.n_sig.median()), float(nat.n_sig.median())
    print(f"\n{'='*62}\n结论\n{'='*62}")
    print(f"降采样到本届条件（400 细胞 · 18,400 对照 · 20k UMI）: E|R_p| 中位 = {md:.0f}")
    print(f"H1 原生功效（1,090 细胞 · 38,176 对照 · 54k UMI）  : E|R_p| 中位 = {mn:.0f}")
    print(f"\n对比三个先前观测：")
    print(f"  K562 GWPS 经验贝叶斯       11")
    print(f"  K562 MASH（外部）           7")
    print(f"  CD4+T Rest MASH（外部）    36   (p75 95)")
    print(f"  F8 从官方 jac 基线反推     288   ← 欠定")
    verdict = ("支持「几十」，F8 的 288 不成立" if md < 120
               else "支持 F8 的 288 量级" if md > 200 else "落在两者之间，需再判")
    print(f"\n>>> {verdict} <<<")
    print(f"\n对应最优召集集合大小 K_p 应定在 {md:.0f} 量级"
          f"（命题 2：预测不确定时 K* > E|R_p|）")
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
