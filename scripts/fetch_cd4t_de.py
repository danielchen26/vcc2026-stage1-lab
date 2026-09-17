"""从 CZI 的 16.8 GB 远程 h5ad 里只抽出需要的 46 MB。

## 为什么能这么做

`GWCD4i.DE_stats.h5ad` 的 layers 是 **未压缩、连续存储** 的 float64 (33983, 10282)：

    chunks=None  compression=None

所以第 i 行就是从 `dataset.id.get_offset() + i * 10282 * 8` 开始的一段连续字节
（82,256 B）。S3 支持 Range（已验证 206 + Accept-Ranges: bytes），
于是可以只取需要的行，不必下载整个 16.8 GB —— 本机只有约 10 GB 磁盘。

## 为什么不用 fsspec 直接花式索引

`h5py` 的花式索引会对散落的行逐个发小读，而 fsspec 的读前缓存把每次小读放大成
一个 block（默认 4 MB+）。283 行 × 4 MB ≈ 1.1 GB 传输，实测超时。
改为：用 fsspec 只读元数据（obs/var，小），用 curl 精确取数据字节。

## 产物

`data/external/cd4t_rest_de.npz`：本届 300 靶基因在 CD4+T **Rest（无刺激）臂**的
逐基因 log_fc / lfcSE，外加每扰动的可信度元数据（guide/donor 复现性、细胞数）。

这是**第二个逐基因源**（第一个是 K562 GenomeWide）。E14–E26 已证明单源封顶，
多源并集是唯一有实测依据的突破路径。

跑法：  ~/vcc2026/.venv/bin/python scripts/fetch_cd4t_de.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import fsspec
import h5py
import numpy as np
import pandas as pd

URL = ("https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/"
       "marson2025_data/GWCD4i.DE_stats.h5ad")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external" / "cd4t_rest_de.npz"
LAYERS = ("log_fc", "lfcSE")
OBS_META = ("n_cells_target", "ontarget_effect_size", "guide_correlation_all",
            "donor_correlation_all_mean", "n_total_de_genes")


def as_str(g) -> np.ndarray:
    """anndata 的分类列或字符串列 → str 数组。"""
    if isinstance(g, h5py.Group) and "categories" in g:
        c = [x.decode() if isinstance(x, bytes) else str(x) for x in g["categories"][:]]
        return np.asarray(c)[g["codes"][:]]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in g[:]])


def ranges_of(rows: np.ndarray) -> list[tuple[int, int]]:
    """把行号压成连续区间，减少请求数。"""
    out, s = [], rows[0]
    for a, b in zip(rows, rows[1:]):
        if b != a + 1:
            out.append((s, a))
            s = b
    out.append((s, rows[-1]))
    return out


def curl_range(url: str, lo: int, hi: int, tries: int = 4) -> bytes:
    """取 [lo, hi] 字节。curl 是这台机器上唯一稳定的传输（urllib 有证书问题）。"""
    for k in range(tries):
        p = subprocess.run(["curl", "-sS", "--max-time", "300", "--retry", "2",
                            "-r", f"{lo}-{hi}", url], capture_output=True)
        if p.returncode == 0 and len(p.stdout) == hi - lo + 1:
            return p.stdout
        time.sleep(1.5 * (k + 1))
    raise RuntimeError(f"取字节 {lo}-{hi} 失败：得到 {len(p.stdout)} B，"
                       f"应为 {hi-lo+1} B；curl rc={p.returncode}")


def main() -> None:
    t0 = time.time()
    print("=== 从远程 16.8 GB h5ad 抽取 CD4+T Rest 臂的逐基因 DE ===\n")

    fo = fsspec.open(URL, mode="rb", block_size=1 << 20,
                     client_kwargs={"timeout": None}).open()
    h = h5py.File(fo, "r")

    cond = as_str(h["obs"]["culture_condition"])
    tg = as_str(h["obs"]["target_contrast_gene_name"])
    genes = as_str(h["var"]["gene_name"])
    n_obs, n_var = h["layers"][LAYERS[0]].shape
    print(f"n_obs={n_obs:,}  n_var={n_var:,}  条件={sorted(set(cond))}")

    targets = {str(t) for t in pd.read_csv(Path.home() / "vcc2026" /
                                          "pert_counts.csv")["target_gene"]}
    targets.discard("non-targeting")
    rows = np.flatnonzero((cond == "Rest") & np.isin(tg, list(targets)))
    if rows.size == 0:
        raise RuntimeError("Rest 臂与本届靶基因无交集 —— 检查列名是否为 "
                           "target_contrast_gene_name")
    rows = np.sort(rows)
    print(f"Rest ∩ 本届 {len(targets)} 个靶基因 = {rows.size} 行 "
          f"（{len(set(tg[rows]))} 个唯一基因，覆盖 {len(set(tg[rows]))/len(targets):.1%}）")

    meta = {k: h["obs"][k][:][rows] for k in OBS_META
            if k in h["obs"] and not isinstance(h["obs"][k], h5py.Group)}
    offsets = {}
    for L in LAYERS:
        d = h["layers"][L]
        off = d.id.get_offset()
        if off is None or d.chunks is not None or d.compression is not None:
            raise RuntimeError(f"{L} 不是连续未压缩存储，无法按偏移直读")
        offsets[L] = (off, d.dtype.itemsize)
    print(f"字节偏移: " + " · ".join(f"{L}@{offsets[L][0]:,}" for L in LAYERS))

    segs = ranges_of(rows)
    row_bytes = n_var * 8
    total = rows.size * row_bytes * len(LAYERS)
    print(f"需取 {rows.size} 行 × {len(LAYERS)} 层 = {total/1e6:.1f} MB，"
          f"压成 {len(segs)} 个连续区间\n")

    data = {}
    for L in LAYERS:
        off, isz = offsets[L]
        assert isz == 8, f"{L} 的 itemsize={isz}，本脚本假定 float64"
        buf = np.empty((rows.size, n_var), np.float32)
        pos, t = 0, time.time()
        for a, b in segs:
            nrow = b - a + 1
            raw = curl_range(URL, off + a * row_bytes,
                             off + (b + 1) * row_bytes - 1)
            blk = np.frombuffer(raw, np.float64).reshape(nrow, n_var)
            buf[pos:pos + nrow] = blk.astype(np.float32)
            pos += nrow
        data[L] = buf
        print(f"  {L}: {buf.shape}  ({time.time()-t:.0f}s)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, perts=tg[rows], genes=genes,
                        **data, **{f"obs_{k}": v for k, v in meta.items()})
    lfc, se = data["log_fc"], data["lfcSE"]
    z = np.abs(lfc) / np.maximum(se, 1e-9)
    print(f"\n已存 {OUT.relative_to(ROOT)}  {OUT.stat().st_size/1e6:.1f} MB")
    print(f"有限值 {np.isfinite(lfc).mean():.1%}   |lfc| 中位 {np.nanmedian(np.abs(lfc)):.4f}")
    print(f"lfcSE 中位 {np.nanmedian(se):.4f} → 隐含 MDE(z=3.184) "
          f"{3.184*np.nanmedian(se):.4f}")
    print(f"|z| 中位 {np.nanmedian(z):.3f}   |z|>3.184 占 {np.nanmean(z>3.184):.3%}")
    print(f"\n耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
