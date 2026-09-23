"""E34b · V14 候选：把集外通道的幅度压到召集集之下，救回 `direction_reach`。

V13 实测（`out/score_v13.json`）：
    pds   0.7500 -> 0.8750  官方 lo +0.2927   ← 赢，且超过榜首的 0.820
    reach 0.1482 -> 0.0250  官方 lo −0.1352   ← 塌 83%
    avg_lo 0.1215 -> 0.1508 (+0.0293)

SPEC §1 的机制论证被推翻：我以为「显著性只加在 r_set ⇒ reach 结构上不动」。
错在 `direction_reach` 的纯前缀深度 k*/N_conf 是在**我们自己全部非零 lfc 的 |lfc|
降序**上取的，不是在 `r_set` 上取的。集外那 550 个去均值 lfc 里有一部分幅度大于
召集集成员，于是它们挤进前缀；它们的符号来自跨系不可迁移的源（[F33](../../docs/02-findings.md#f33)：
符号一致率 51–57%），纯度立刻破，k* 崩。

修法（本文件要验的）：**保持召集集在 |lfc| 排序上整体领先集外**。
把集外向量整体等比缩放，使 max|off| < min|callset| —— 等比缩放保持集外内部的相对
次序与全部符号，只是把整块压到召集集下面。与 J（一律 ×0.3）不同：J 是任意幅度，
不保证跨越边界；本设计直接钉住边界。

两件事一起量：
  1. `pds`：缩放后还剩多少收益（cosine 由大坐标主导，压幅可能吃掉收益）。
  2. **排序不变量**：逐扰动检查 min|callset| > max|off| 是否成立 ——
     这是 reach 能不能恢复的结构性前提，可以在不跑打分器的情况下直接判。

跑法：/Users/chetianc/vcc2026/.venv/bin/python experiments/E34-pds-decouple/cap_probe.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "e28run", ROOT / "experiments" / "E28-pds" / "run.py")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

K_CALL, K_OFF = 288, 550
MARGIN = 0.99          # max|off| 压到 min|callset| 的这个倍数


def main() -> None:
    perts, genes, real, gi_full, B, S = m._load_beta()
    m_full = m._ctrl_profile(real)
    bts = float(m.vcc_cfg().bulk_target_sum)
    real_bulk = m.pseudobulk_bulk_lognorm(real, m.PERT_COL, bulk_target_sum=bts)
    G, n_p = genes.size, len(perts)
    print(f"扰动 {n_p}  共同基因 {gi_full.size:,}  K_CALL={K_CALL} K_OFF={K_OFF}")

    Bc = np.where(np.isfinite(B), B, 0.0)
    Bd = Bc - Bc.mean(1, keepdims=True)

    def _topk(v, k):
        keep = np.zeros(v.size, bool)
        keep[np.argsort(np.abs(v))[::-1][:min(k, v.size)]] = True
        return keep

    def build(mode: str, k_off: int = K_OFF):
        """返回 (lf 矩阵, 逐扰动的 (min|call|, max|off|))。"""
        lf = np.zeros((n_p, G))
        bounds = []
        for j in range(n_p):
            call = _topk(Bc[:, j], K_CALL)
            v = np.where(call, Bc[:, j], 0.0)
            off = Bd[:, j].copy()
            off[call] = 0.0
            off[~_topk(off, k_off)] = 0.0

            mn_call = np.abs(v[call]).min()
            mx_off = np.abs(off).max()
            if mode == "cap" and mx_off > 0:
                off *= (MARGIN * mn_call) / mx_off      # 等比压到召集集之下
            elif mode == "half" and mx_off > 0:
                off *= (0.5 * mn_call) / mx_off
            lf[j, gi_full] = v + off
            bounds.append((mn_call, np.abs(off).max()))
        return lf, bounds

    def lf_base():
        lf = np.zeros((n_p, G))
        for j in range(n_p):
            call = _topk(Bc[:, j], K_CALL)
            lf[j, gi_full] = np.where(call, Bc[:, j], 0.0)
        return lf

    def mean_pds(lf):
        got = m.pds(m._pred_bulk_closed(m_full, lf, perts, bts), real_bulk, genes)
        return float(np.mean(list(got.values()))), got

    base, _ = mean_pds(lf_base())
    rows = [("A 召集 288（V8）", base, None)]
    for name, mode in (("I  集外原幅（= V13）", "raw"),
                       ("K1 集外压到 0.99×min|call|", "cap"),
                       ("K2 集外压到 0.50×min|call|", "half")):
        lf, bounds = build(mode)
        mu, _ = mean_pds(lf)
        rows.append((name, mu, bounds))

    print(f"\n{'设计':32s} {'pds':>8s} {'Δ vs A':>9s} {'量子':>5s}")
    print("-" * 60)
    for name, mu, _ in rows:
        print(f"{name:32s} {mu:8.4f} {mu-base:+9.4f} {round((mu-base)*56):5d}")

    print("\n" + "=" * 78)
    print("排序不变量：min|callset| > max|off| 是否成立？（reach 能否恢复的结构前提）")
    print("=" * 78)
    print(f"{'扰动':10s}" + "".join(f"{n.split()[0]:>22s}" for n, _, b in rows if b))
    print("-" * 78)
    for j, p in enumerate(perts):
        cells = ""
        for _, _, b in rows:
            if b is None:
                continue
            mn, mx = b[j]
            ok = "OK " if mx < mn else "破 "
            cells += f"{ok}{mn:.3f}/{mx:.3f}".rjust(22)
        print(f"{p:10s}{cells}")
    print("-" * 78)
    print("判读：K1 若 8/8 个扰动 OK 且 pds 收益 ≥ 4 个量子，则它同时保住 reach 与 pds，"
          "\n      值得一次 build；若 pds 收益被压光，说明 pds 的收益本质来自"
          "\n      集外幅度**超过**召集集，与 reach 在同一个旋钮上不可分。")

    # ---- 压幅之后重扫 k_off ------------------------------------------------
    # 理由：k_off = 550 是在**未压幅**下选的（SPEC §0e）。压幅改变了每个集外基因的
    # 贡献，最优点会挪 —— 直接沿用 550 就是 T11 的「两个旋钮当一条曲线」再犯一次。
    print("\n" + "=" * 78)
    print("压幅（0.99×min|call|）之后重扫 k_off —— 不许沿用未压幅时的 550")
    print("=" * 78)
    print(f"{'k_off':>8s} {'pds':>8s} {'Δ vs A':>9s} {'量子':>5s} {'不变量':>8s}")
    print("-" * 46)
    best = None
    for k in (288, 550, 1000, 2000, 3000, 5000):
        lf, bounds = build("cap", k_off=k)
        mu, _ = mean_pds(lf)
        ok = sum(1 for mn, mx in bounds if mx < mn)
        print(f"{k:8d} {mu:8.4f} {mu-base:+9.4f} {round((mu-base)*56):5d} {f'{ok}/8':>8s}")
        if ok == n_p and (best is None or mu > best[1]):
            best = (k, mu)
    print("-" * 46)
    if best:
        print(f"压幅下最优且不变量全过：k_off = {best[0]}，pds {best[1]:.4f}"
              f"（+{round((best[1]-base)*56)} 个量子）")
        print(f"对比 V13（未压幅 k=550）：pds 0.8750，+8 量子，但不变量 0/8 ——"
              f" reach 因此塌到 0.0250")


if __name__ == "__main__":
    main()
