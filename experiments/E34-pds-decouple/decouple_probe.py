"""E34 · 能不能把 `pds` 的收益和 `reach` 的损害解耦？

背景（E28 design 探针 2026-09-22 重跑，见 out/design_probe_rerun.log）：

    A 仅召集 288（V8 现状）   pds 0.7321
    F top1000 单独            pds 0.7143   <- 单独是亏的
    C 全基因去面板均值        pds 0.7500
    G top1000 + 去面板均值    pds 0.7857   <- 最优

所以 pds 的收益**全部来自去面板均值**，top1000 只是在去均值之后才有增量。
而去面板均值恰好是摧毁 `de_direction_reach` 的那一步：reach 测的是我们自己
|lfc| 排序的方向纯净前缀深度 k*/N_conf，去均值让排序不再跟源侧置信度对齐
（V2 实测 reach raw 0.00796 vs V8 0.1482）。官方刻度上这笔交易是净亏：
pds +0.0139 avg，reach −0.0257 avg，净 −0.0118。

本探针问一个结构性问题：**如果只在召集集之外去均值，召集集内的 lfc 逐位不动，
pds 的收益还在吗？** 若在，则 reach 的纯前缀（由召集集内的排序决定）结构上不受影响，
两者就解耦了。

设计：
  A   召集 288，原值                      —— V8 现状，基准
  C   全基因去面板均值                    —— 参照
  G   top1000 去面板均值                  —— 参照（当前最优 pds）
  H   召集 288 原值 + 其余全部去面板均值
  I   召集 288 原值 + 其余 top1000 去面板均值
  J   召集 288 原值 + 其余去面板均值后再 x0.3（压幅，试探 mse/nmae 的代价）

⚠️ 本探针只算 `pds`。它是**相对**instrument，不是官方分数：
共同基因 7,481（run.py 的 _load_beta 用全部 K562GW 行），而 build 侧是 7,016
（build_v3.py:40-44 硬编码了 K_A 以避免这个差异）。绝对值不可与 agg_*.parquet 比，
只可用于同一次运行内的设计间比较。

跑法：/Users/chetianc/vcc2026/.venv/bin/python experiments/E34-pds-decouple/decouple_probe.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# 与 build_v3.py:37-38 同一手法：把 E28 的 run.py 当模块加载，复用它的 loader 与 pds。
_spec = importlib.util.spec_from_file_location(
    "e28run", ROOT / "experiments" / "E28-pds" / "run.py")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

K_CALL = 288   # V8 的扁平召集集大小（F30 的 V7 旋钮）


def main() -> None:
    perts, genes, real, gi_full, B, S = m._load_beta()
    m_full = m._ctrl_profile(real)
    bts = float(m.vcc_cfg().bulk_target_sum)
    real_bulk = m.pseudobulk_bulk_lognorm(real, m.PERT_COL, bulk_target_sum=bts)
    G = genes.size
    n_p = len(perts)
    print(f"扰动 {n_p}  基因 {G:,}  共同基因 {gi_full.size:,}  bts {bts:g}")

    Bc = np.where(np.isfinite(B), B, 0.0)          # (共同基因, 扰动)
    Bd = Bc - Bc.mean(1, keepdims=True)            # 去面板均值（逐基因跨扰动）

    def _topk_mask(v: np.ndarray, k: int | None) -> np.ndarray:
        """|v| 最大的 k 个位置为 True；k 为 None 则全 True。"""
        if k is None or k >= v.size:
            return np.ones(v.size, bool)
        keep = np.zeros(v.size, bool)
        keep[np.argsort(np.abs(v))[::-1][:k]] = True
        return keep

    def lf_plain(vals, topk=None, scale=1.0):
        """照 run.py:169-176 的 lf_from，用于复现 A / C / G。"""
        lf = np.zeros((n_p, G))
        for j in range(n_p):
            v = vals[:, j].copy()
            v[~_topk_mask(v, topk)] = 0.0
            lf[j, gi_full] = scale * v
        return lf

    def lf_split(topk_off=None, scale_off=1.0):
        """召集集内用原 Bc（逐位等于 A），召集集外用去均值的 Bd。

        召集集 = |Bc[:, j]| 的前 K_CALL 个位置（与 A 的 topk=288 同一集合）。
        """
        lf = np.zeros((n_p, G))
        for j in range(n_p):
            call = _topk_mask(Bc[:, j], K_CALL)
            v = np.where(call, Bc[:, j], 0.0)
            off = Bd[:, j].copy()
            off[call] = 0.0                                  # 召集集内不许被去均值污染
            off[~_topk_mask(off, topk_off)] = 0.0
            lf[j, gi_full] = v + scale_off * off
        return lf

    designs = [
        ("A 召集 288 原值（V8 现状）",           lf_plain(Bc, topk=K_CALL)),
        ("C 全基因去面板均值",                   lf_plain(Bd)),
        ("G top1000 去面板均值",                 lf_plain(Bd, topk=1000)),
        ("H 召集 288 + 其余全部去均值",          lf_split()),
        ("I 召集 288 + 其余 top1000 去均值",     lf_split(topk_off=1000)),
        ("J 召集 288 + 其余去均值 x0.3",         lf_split(scale_off=0.3)),
    ]
    # 幅度扫描：V8 的召集集用 LAMBDA=0.7，新通道该用多少？J(0.3) 已证明会亏，
    # 需要定位悬崖位置，避免用 0.7 build 完才发现收益没了（T10 的教训）。
    designs += [(f"I@scale={s:<4g} top1000", lf_split(topk_off=1000, scale_off=s))
                for s in (1.0, 0.85, 0.7, 0.5)]
    # top-K 扫描：1000 是从 V2 继承的，从未被单独验证过。
    designs += [(f"I@k={k:<5d} scale=1.0", lf_split(topk_off=k, scale_off=1.0))
                for k in (300, 500, 2000, 3000)]

    print(f"\n{'设计':36s} {'pds':>8s} {'from_base':>10s}  逐扰动")
    print("-" * 100)
    base = None
    for name, lf in designs:
        got = m.pds(m._pred_bulk_closed(m_full, lf, perts, bts), real_bulk, genes)
        mu = float(np.mean(list(got.values())))
        if base is None:
            base = mu
        print(f"{name:36s} {mu:8.4f} {2.0*(mu-0.5):10.4f}  "
              f"{'Δ vs A ' + format(mu - base, '+.4f'):>14s}  "
              + " ".join(f"{got[p]:.2f}" for p in perts))
    print("-" * 100)
    print("判读：H/I 若 ≥ G 的 pds，则去均值的收益不需要动召集集即可拿到，"
          "reach 的纯前缀结构上不受影响 → 解耦成立。\n"
          "      H/I 若 ≈ A，则 pds 的收益本质上来自召集集内那 288 个值被改写，"
          "与 reach 不可分 → 解耦失败，P1 关闭。")


if __name__ == "__main__":
    main()
