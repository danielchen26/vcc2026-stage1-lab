"""E11 — 直接实测 h_replicate，验证整个项目最承重的锚点。

## 为什么这是最该验的一条

追平榜首所需的 h = 0.134 是这样来的：
    官方公布 replicate 锚点 r_jac = 0.399
    → h_replicate = 2*0.399/1.399 = 0.570        （用 jac = h/(2-h) 反推）
    → 追平线 = 0.23 × 0.570 = 0.134
**整条链条上没有一个数是直接测的。** 而所有 go/no-go 判断都挂在它身上。

现在有条件直接测：主办方自己的 2025 H1 真实数据 + 官方 DE 机器
（`ControlRef.de_table`，E04 已逐基因验证 = 官方 cell-eval2 preset vcc2026）。

## 方法：官方 1 分锚点的字面复现

「真实数据对半劈开，一半预测另一半」：
    对照 38,176 → 两个互不相交的 18,400（本届的对照规模）
    每个扰动的细胞 → 两个互不相交的 400（本届的每扰动规模）
    两半各自用自己的对照算 R_p，再比重叠

同时报三个口径，避免单一口径的偏差：
    h   = |A ∩ B| / mean(|A|,|B|)      我们一直用的对称重叠率
    jac = |A ∩ B| / |A ∪ B|            官方 de_wilcoxon_sig_jaccard 的口径
    并核对恒等式 jac = h/(2-h) 在 |A|≈|B| 时是否成立

跑法：  ~/vcc2026/.venv/bin/python experiments/E11-replicate-anchor/run.py
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

sys.path.insert(0, str(ROOT / "experiments" / "E10-h1-erp"))
from run import VCC_CTRL_CELLS, VCC_PERT_CELLS, VCC_UMI, read_rows, thin, to_cpm  # noqa: E402

H5 = ROOT / "data" / "vcc2025" / "adata_Validation.h5ad"
OUT = Path(__file__).parent
ALPHA = 0.05
SEED = 0
OFFICIAL_JAC = 0.399
OFFICIAL_H = 2 * OFFICIAL_JAC / (1 + OFFICIAL_JAC)


def main() -> None:
    t0 = time.time()
    print("=== E11 直接实测 h_replicate ===")
    print(f"官方锚点 r_jac = {OFFICIAL_JAC}  →  反推 h_replicate = {OFFICIAL_H:.3f}")
    print(f"本实验直接测，看是否吻合\n")

    with h5py.File(H5, "r") as f:
        genes = [g.decode() if isinstance(g, bytes) else str(g)
                 for g in f["var"]["_index"][:]]
        o = f["obs"]["target_gene"]
        cats = [c.decode() if isinstance(c, bytes) else str(c)
                for c in o["categories"][:]]
        codes = o["codes"][:]
    ntc_code = cats.index("non-targeting")
    rng = np.random.default_rng(SEED)

    # ---- 对照劈成两半，各 18,400（本届规模）----
    ntc = rng.permutation(np.flatnonzero(codes == ntc_code))
    need = 2 * VCC_CTRL_CELLS
    if len(ntc) < need:
        raise RuntimeError(f"NTC 只有 {len(ntc)}，两半各 {VCC_CTRL_CELLS} 需要 {need}")
    halves = (ntc[:VCC_CTRL_CELLS], ntc[VCC_CTRL_CELLS:need])
    print(f"NTC {len(ntc):,} → 两半各 {VCC_CTRL_CELLS:,}（互不相交）")

    refs = []
    for k, rows in enumerate(halves):
        M = thin(read_rows(rows), VCC_UMI, rng)
        p = OUT / f"_ntc_half{k}.h5ad"
        ad.AnnData(X=sp.csr_matrix(M),
                   var=pd.DataFrame(index=pd.Index(genes))).write_h5ad(p)
        r = ControlRef.load(p, genes)
        refs.append((r, p))
        print(f"  半 {k}: gate={r.G:,}  对照={r.n_ctrl:,}")

    if refs[0][0].G != refs[1][0].G:
        print(f"  ⚠️ 两半 gate 不同（{refs[0][0].G:,} vs {refs[1][0].G:,}）"
              f"—— 取共同下标比较")

    # ---- 逐扰动对半劈开 ----
    need_cells = 2 * VCC_PERT_CELLS
    rows_out = []
    print(f"\n只用细胞数 ≥ {need_cells} 的扰动（两半各 {VCC_PERT_CELLS}）")
    print(f"{'扰动':>12} {'细胞':>6} {'|A|':>6} {'|B|':>6} {'∩':>6} "
          f"{'h':>7} {'jac':>7} {'h/(2-h)':>8}")
    for ci, name in enumerate(cats):
        if ci == ntc_code:
            continue
        idx = np.flatnonzero(codes == ci)
        if len(idx) < need_cells:
            continue
        pick = rng.permutation(idx)[:need_cells]
        sig = []
        for k in (0, 1):
            M = thin(read_rows(pick[k * VCC_PERT_CELLS:(k + 1) * VCC_PERT_CELLS]),
                     VCC_UMI, rng)
            padj, _ = refs[k][0].de_table(to_cpm(M), tie_correct=True)
            sig.append(padj < ALPHA)
        # gate 可能不同 → 用两半 gate 的交集比较（按基因名对齐）
        g0 = np.asarray(refs[0][0].gidx) if hasattr(refs[0][0], "gidx") else None
        g1 = np.asarray(refs[1][0].gidx) if hasattr(refs[1][0], "gidx") else None
        if g0 is not None and g1 is not None and not np.array_equal(g0, g1):
            common = np.intersect1d(g0[sig[0]], g1[sig[1]])
            na, nb = int(sig[0].sum()), int(sig[1].sum())
            inter = len(common)
            union = na + nb - inter
        else:
            na, nb = int(sig[0].sum()), int(sig[1].sum())
            inter = int((sig[0] & sig[1]).sum())
            union = int((sig[0] | sig[1]).sum())
        if na == 0 or nb == 0:
            continue
        h = inter / (0.5 * (na + nb))
        jac = inter / union if union else np.nan
        rows_out.append(dict(target_gene=name, n_cells=len(idx), n_a=na, n_b=nb,
                             inter=inter, union=union, h=h, jac=jac))
        print(f"{name:>12} {len(idx):6d} {na:6d} {nb:6d} {inter:6d} "
              f"{h:7.3f} {jac:7.3f} {h/(2-h):8.3f}")

    df = pd.DataFrame(rows_out)
    df.to_csv(OUT / "result.csv", index=False)
    for _, p in refs:
        p.unlink(missing_ok=True)

    print(f"\n{'='*66}\n结论（n = {len(df)} 个扰动）\n{'='*66}")
    for col, off in (("h", OFFICIAL_H), ("jac", OFFICIAL_JAC)):
        v = df[col].dropna()
        q = np.percentile(v, [25, 50, 75])
        print(f"  实测 {col:4s} 中位 {q[1]:.3f}  IQR [{q[0]:.3f}, {q[2]:.3f}]"
              f"   官方 {off:.3f}   比值 {q[1]/off:.2f}×")
    # 恒等式核对
    ok = np.allclose(df.jac, df.h / (2 - df.h), atol=0.02)
    print(f"\n  恒等式 jac = h/(2-h) 在 |A|≈|B| 时成立: {ok}")
    print(f"  |A|/|B| 中位 = {np.median(df.n_a / df.n_b):.3f}"
          f"（越接近 1 恒等式越准）")

    hm = float(df.h.median())
    print(f"\n  追平榜首需要的 h：0.23 × {hm:.3f} = {0.23*hm:.3f}"
          f"   （原用官方反推值算得 0.134）")
    verdict = ("锚点得到证实，0.570 可用" if abs(hm - OFFICIAL_H) / OFFICIAL_H < 0.2
               else f"锚点偏离 {(hm-OFFICIAL_H)/OFFICIAL_H:+.0%}，需据实测值重算所有 go/no-go 阈值")
    print(f"\n>>> {verdict} <<<")
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
