"""E31 前置校验（跑全量前 3 分钟就能证伪的三件事）：

1. `ref.m_full` 真的是全 n_ctrl 个对照细胞的全基因 CPM **总体**均值，且与 `tgt` 同尺度。
2. `design_cells(..., force_mean=True)`（新默认）与 **HEAD 版 decoder.py** 逐位相同
   —— 其它三个 agent 并发 import 这个函数，默认路径必须不变。
3. `force_mean=False` 真的把「抽样子集相对总体的偏差」留在了列均值里。
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "E10-h1-erp"))
from run import VCC_UMI, read_rows, thin  # noqa: E402

from vcclab.decoder import design_cells  # noqa: E402
from vcclab.scorer import TS_CELL, ControlRef  # noqa: E402

H5 = ROOT / "data" / "vcc2025" / "adata_Validation.h5ad"
N_CTRL, N_CELLS, SEED = 3000, 400, 0


def as_str(g):
    if isinstance(g, h5py.Group) and "categories" in g:
        c = [x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]
        return np.asarray(c)[g["codes"][:]]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g[:]])


def head_design_cells():
    """HEAD 提交里的 design_cells，用来证明默认路径逐位未变。"""
    src = subprocess.run(["git", "show", "HEAD:src/vcclab/decoder.py"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    d = Path(tempfile.mkdtemp())
    (d / "__init__.py").write_text("")
    (d / "decoder_head.py").write_text(src.replace("from .scorer import", "from vcclab.scorer import"))
    sys.path.insert(0, str(d))
    import decoder_head  # noqa: PLC0415
    return decoder_head.design_cells


def main() -> None:
    rng = np.random.default_rng(SEED)
    with h5py.File(H5, "r") as f:
        genes = as_str(f["var"]["_index"])
        tg = as_str(f["obs"]["target_gene"])
    rows = rng.choice(np.flatnonzero(tg == "non-targeting"), N_CTRL, replace=False)
    ctrl = thin(read_rows(rows), VCC_UMI, rng)
    tmp = Path(tempfile.mkdtemp()) / "_ctrl.h5ad"
    ad.AnnData(X=sp.csr_matrix(ctrl),
               var=pd.DataFrame(index=pd.Index(genes))).write_h5ad(tmp, compression="gzip")
    ref = ControlRef.load(tmp, list(genes))
    tmp.unlink()

    # --- 1. m_full 的定义与尺度 -------------------------------------------------
    pop = np.asarray(ref._cpm_csr.mean(0)).ravel()
    d1 = float(np.abs(ref.m_full - pop).max())
    print(f"[1] max|m_full - _cpm_csr.mean(0)| = {d1:.3e}   "
          f"m_full.sum() = {ref.m_full.sum():.1f}  (TS_CELL = {TS_CELL:.0f})")
    assert d1 < 1e-9, "m_full 不是 _cpm_csr 的总体均值"
    assert abs(ref.m_full.sum() - TS_CELL) / TS_CELL < 1e-6, "m_full 不在 CPM 尺度上"

    # --- 2. 默认路径与 HEAD 逐位相同 --------------------------------------------
    gsel = rng.choice(ref.G, 60, replace=False)
    lfc = rng.normal(0, 0.8, 60)
    old = head_design_cells()
    a = old(ref, gsel, lfc, n_cells=N_CELLS, seed=SEED)
    b = design_cells(ref, gsel, lfc, n_cells=N_CELLS, seed=SEED)
    same = bool(np.array_equal(a, b))
    print(f"[2] HEAD 版 vs 新版(force_mean 默认 True) 逐位相同 = {same}")
    assert same, "默认路径变了，其它 agent 的复现会被破坏"

    # --- 3. force_mean=False 保留了抽样波动 ------------------------------------
    c = design_cells(ref, gsel, lfc, n_cells=N_CELLS, seed=SEED, force_mean=False)
    keep = np.setdiff1d(np.arange(ref.n_genes), ref.gidx[gsel])   # 被逐列覆盖的响应基因除外
    hi = ref.gidx[ref.m_gate > 50]                                 # 只看高表达基因，量化噪声小
    hi = np.intersect1d(hi, keep)
    lf = np.zeros(ref.n_genes)
    lf[ref.gidx[gsel]] = lfc
    tgt = ref.m_full * 2.0 ** lf
    tgt *= TS_CELL / tgt.sum()
    rel_t = np.std(b[:, hi].mean(0) / tgt[hi] - 1.0)
    rel_f = np.std(c[:, hi].mean(0) / tgt[hi] - 1.0)
    print(f"[3] 列均值相对偏差 std（{len(hi)} 个高表达基因）: "
          f"force_mean=True {rel_t:.5f}   force_mean=False {rel_f:.5f}   "
          f"倍数 {rel_f / max(rel_t, 1e-12):.1f}x")
    assert rel_f > 5 * rel_t, "force_mean=False 没有恢复波动"
    print("\n三项全部通过")


if __name__ == "__main__":
    main()
