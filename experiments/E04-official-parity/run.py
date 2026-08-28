"""E04 — 与官方 cell-eval2 0.16.0 (preset vcc2026) 逐基因对齐.

三件事:

1. **gate 一致**: 官方 `filter_gene_min_cpm_cell=5.0` 选出的基因集必须与
   `ControlRef.gidx` 完全一致 (context_A: 9,929).
2. **DE 一致**: 官方 `compute_de(backend="scanpy")` 的 log2_fold_change 与
   p_adj, 与 `ControlRef.de_table` 逐基因对齐; 显著集 R̂ 对称差必须为 0
   —— 只有做了并列校正 (tie correction) 才成立, 这是本实验的关键结论.
3. **六指标一致**: `compute_metrics` 的 de_wilcoxon_sig_jaccard 必须等于我们
   设计出来的重叠比 (|R_p ∩ R_r| / |R_p ∪ R_r| = 100/401 = 0.249377).

pred 与 real 都由 Stage 2 解码器构造 (我们没有目标细胞系的真答案), 故意让两边
共享 100 个响应基因、方向独立随机 -> jaccard 和 direction_fidelity 都可预判.

跑法::

    ~/vcc2026/.venv/bin/python experiments/E04-official-parity/run.py
    ~/vcc2026/.venv/bin/python experiments/E04-official-parity/run.py --dry-run

**耗时**: 完整跑约 10 分钟, 其中官方单侧 DE 表 ~297 s、compute_metrics ~300 s,
我们自己的部分 < 30 s. `--dry-run` 跳过两个官方调用 (只验证构造 + 我们的 DE +
官方 API 签名), 约 60 s.

产物写在 `out/` (h5ad / parquet 均已被 .gitignore 拦掉, 绝不入库).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from _common import DATA, gene_names, header, load_ref, pert_list

OUT = Path(__file__).resolve().parent / "out"
N_PERTS = 3            # 官方 DE 一个扰动就要 ~100 s, 3 个够做逐基因对齐
N_R = 250              # 每个扰动的意图响应基因数
N_SHARED = 100         # pred 与 real 故意共享的响应基因数 -> jaccard 可预判
ALPHA = 0.05
CTRL_LABEL = "non-targeting"
SEED = 2026

DE_KW = dict(                      # 与 preset vcc2026 的 DE 段逐字段一致
    backend="scanpy",
    groupby="target_gene",
    reference=CTRL_LABEL,
    mean_calc="arithmetic",
    epsilon=1e-9,
    input_type="counts",
    target_sum=1e6,
    clip_value=None,
    filter_gene_min_cpm_cell=5.0,
    fdr_scope="per_pert",
    threads=-1,
    device="cpu",
)


def raw_control(context: str = "A"):
    """官方发布的对照细胞原始整数计数 (不是 CPM). h5ad 的 X 是 csr."""
    import h5py
    from scipy import sparse

    with h5py.File(DATA / f"context_{context}.h5ad", "r") as f:
        return sparse.csr_matrix(
            (f["X/data"][:], f["X/indices"][:], f["X/indptr"][:]),
            shape=tuple(f["X"].attrs["shape"]),
        )


def write_h5ad(path: Path, ctrl, blocks: dict[str, np.ndarray], genes) -> None:
    """对照块在前 (原始计数), 各扰动块在后 (行和 1e6 的整数计数)."""
    import anndata as ad
    import pandas as pd
    from scipy import sparse

    X = sparse.vstack(
        [ctrl.astype(np.float32)]
        + [sparse.csr_matrix(b.astype(np.float32)) for b in blocks.values()],
        format="csr",
    )
    labels = [CTRL_LABEL] * ctrl.shape[0]
    for pert, b in blocks.items():
        labels += [pert] * b.shape[0]
    obs = pd.DataFrame(
        {"target_gene": pd.Categorical(labels)},
        index=[f"c{i}" for i in range(X.shape[0])],
    )
    a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=np.asarray(genes)))
    path.parent.mkdir(parents=True, exist_ok=True)
    a.write_h5ad(path)


def build_intents(ref, rg):
    """pred / real 的意图: 共享 N_SHARED 个响应基因, 方向各自独立随机."""
    perts = list(pert_list()[:N_PERTS])
    intents = {}
    for pert in perts:
        R_p = rg.choice(ref.G, N_R, replace=False)
        rest = np.setdiff1d(np.arange(ref.G), R_p)
        R_r = np.concatenate(
            [rg.choice(R_p, N_SHARED, replace=False),
             rg.choice(rest, N_R - N_SHARED, replace=False)]
        )
        intents[pert] = {
            "pred": (R_p, rg.choice([-1, 1], N_R) * rg.uniform(0.4, 2.2, N_R)),
            "real": (R_r, rg.choice([-1, 1], N_R) * rg.uniform(0.4, 2.2, N_R)),
        }
    return perts, intents


def main() -> int:
    dry = "--dry-run" in sys.argv
    header("E04 官方打分器逐基因对齐" + ("  [dry-run]" if dry else ""))
    genes = gene_names()
    ref, t_load = load_ref("A")
    print(f"ControlRef 加载 {t_load:.1f}s   gate={ref.G}   对照={ref.n_ctrl}")

    rg = np.random.default_rng(SEED)
    perts, intents = build_intents(ref, rg)

    blocks = {"pred": {}, "real": {}}
    t0 = time.time()
    for i, pert in enumerate(perts):
        for s, side in enumerate(("pred", "real")):
            R, lfc = intents[pert][side]
            # 固定种子: str 的 hash() 每个进程都不同, 绝不能用来当种子
            blocks[side][pert] = ref.design(R, lfc, seed=SEED + 10 * i + s)
    print(f"解码器构造 {2 * N_PERTS} 个扰动块: {time.time() - t0:.2f}s")

    ctrl = raw_control("A")
    paths = {}
    for side in ("pred", "real"):
        paths[side] = OUT / f"parity_{side}.h5ad"
        write_h5ad(paths[side], ctrl, blocks[side], genes)
        print(f"写出 {paths[side]}  ({ctrl.shape[0] + N_PERTS * 400} 个细胞)")

    # ---------- 我们的 DE (含 / 不含并列校正) ----------
    t0 = time.time()
    ours = {}
    for pert in perts:
        C = blocks["pred"][pert]
        padj_t, lfc_t = ref.de_table(C, tie_correct=True)
        padj_n, _ = ref.de_table(C, tie_correct=False)
        ours[pert] = (padj_t, lfc_t, padj_n)
    t_ours = time.time() - t0
    print(f"我们的 DE ({N_PERTS} 个扰动 x2 口径): {t_ours:.2f}s")

    from cell_eval2 import EvalConfig, compute_metrics          # noqa: E402
    from cell_eval2.de_compute import compute_de                # noqa: E402
    from dataclasses import replace                             # noqa: E402

    cfg = EvalConfig.from_preset("vcc2026")
    cfg = replace(cfg, pert_col="target_gene", device="cpu")
    cfg = replace(cfg, de=replace(cfg.de, backend="scanpy"))
    print(f"preset vcc2026: de.backend={cfg.de.backend} pert_col={cfg.pert_col}")
    if dry:
        print("\n[dry-run] 跳过官方 compute_de / compute_metrics (~10 min).")
        return 0

    # ---------- 官方 DE 表 ----------
    import polars as pl

    import anndata as ad

    t0 = time.time()
    off = compute_de(ad.read_h5ad(paths["pred"]), **DE_KW)
    t_off = time.time() - t0
    off.write_parquet(OUT / "de_pred_official.parquet")
    print(f"\n官方单侧 DE 表: {t_off:.2f}s  shape={off.shape}  "
          f"我们 {t_ours / 2:.2f}s  加速 {t_off / (t_ours / 2):.1f}x")

    pos = {g: i for i, g in enumerate(genes[ref.gidx])}
    print("\n扰动      官方  我们(含并列校正)  我们(未校正)  对称差")
    for pert in perts:
        sub = off.filter(pl.col("target") == pert)
        idx = np.array([pos[f] for f in sub["feature"].to_list()])
        assert len(idx) == ref.G and len(set(idx)) == ref.G, "gate 基因集不一致"
        o_lfc = sub["log2_fold_change"].to_numpy()
        o_padj = sub["p_adj"].to_numpy()
        padj_t, lfc_t, padj_n = ours[pert]
        s_off = set(np.array(sub["feature"].to_list())[o_padj < ALPHA])
        s_t = set(genes[ref.gidx][padj_t < ALPHA])
        s_n = set(genes[ref.gidx][padj_n < ALPHA])
        print(
            f"{pert:<8} {len(s_off):5d} {len(s_t):14d} {len(s_n):13d} "
            f"{len(s_off ^ s_t):9d}"
        )
        print(
            f"         lfc 最大绝对差 {np.abs(o_lfc - lfc_t[idx]).max():.3e}   "
            f"log10(p_adj) 中位绝对差 "
            f"{np.median(np.abs(np.log10(o_padj + 1e-300) - np.log10(padj_t[idx] + 1e-300))):.4f}"
            f"   未校正对称差 {len(s_off ^ s_n)}"
        )

    # ---------- 官方六指标 ----------
    t0 = time.time()
    df = compute_metrics(str(paths["pred"]), str(paths["real"]), config=cfg)
    print(f"\ncompute_metrics: {time.time() - t0:.1f}s")
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_width_chars(200)
    print(df)
    print(
        f"\n预判: jaccard = |R_p ∩ R_r| / |R_p ∪ R_r| = {N_SHARED}/"
        f"{2 * N_R - N_SHARED} = {N_SHARED / (2 * N_R - N_SHARED):.6f} "
        "(实际会因为解码器多判/少判 1-2 个基因而略偏);  "
        "direction_fidelity ≈ 0.5 (方向独立随机);  direction_reach = 0 (纯度到不了 0.9)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
