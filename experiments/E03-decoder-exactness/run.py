"""E03 — Stage 2 解码器的精确性冲烟测试.

给解码器一个「意图」: 250 个 gate 内基因 + 每个基因一个目标 lfc (随机方向,
|lfc| ~ U(0.4, 2.2)). 解码器必须交出 400 个**整数**细胞 (行和恰为 1e6),
使官方口径的 DE 在这 400 个细胞上:

  * 恰好判这 250 个基因显著 (召回 100%, 精确率 100%, 假阳性 0);
  * 每个基因的 lfc 与意图一致 (方向 100%, 幅度中位误差 ~6e-4).

行和恰为 1e6 => counts 等于 CPM, 打分器的归一化是恒等映射, 于是「意图」
被无损写进提交文件. 这是 Stage 1 (预测) 与 Stage 2 (构造) 解耦的前提.

跑法::

    ~/vcc2026/.venv/bin/python experiments/E03-decoder-exactness/run.py

耗时 ~20 s (加载 13 s + design 0.3 s + de_table 4 s). 固定种子, 完全确定.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from _common import header, load_ref

N_R = 250              # 意图响应基因数
N_CELLS = 400
ALPHA = 0.05
SEED_INTENT = 42       # 选基因 + 抽 lfc
SEED_DESIGN = 1        # 解码器内部自举与置换


def main() -> int:
    header("E03 解码器精确性")
    ref, t_load = load_ref("A")
    print(
        f"ControlRef 加载: {t_load:.1f}s   gate={ref.G}   对照细胞={ref.n_ctrl}"
    )

    rg = np.random.default_rng(SEED_INTENT)
    R = rg.choice(ref.G, N_R, replace=False)
    lfc = rg.choice([-1, 1], N_R) * rg.uniform(0.4, 2.2, N_R)

    t0 = time.time()
    C = ref.design(R, lfc, n_cells=N_CELLS, seed=SEED_DESIGN)
    t_des = time.time() - t0
    # 无损翻译的三个硬约束: 非负 / 整数 / 行和恰为 1e6
    assert np.all(C.sum(1) == 1_000_000), "行和不等于 1e6 -> counts != CPM"
    assert np.all(C >= 0) and np.all(C == np.floor(C)), "不是非负整数计数"

    t0 = time.time()
    padj, lf = ref.de_table(C)
    t_de = time.time() - t0

    R_hat = np.flatnonzero(padj < ALPHA)
    intended = np.zeros(ref.G, bool)
    intended[R] = True
    realized = np.zeros(ref.G, bool)
    realized[R_hat] = True
    tp = int((intended & realized).sum())
    fp = int((~intended & realized).sum())

    print(
        f"design: {t_des:.2f}s   de_table: {t_de:.2f}s   "
        f"nnz/cell={np.mean((C > 0).sum(1)):.0f}"
    )
    print(
        f"意图 |R|={len(R)}  实际 |R̂|={len(R_hat)}  命中={tp}  "
        f"召回={tp / len(R):.1%}  精确率={tp / max(len(R_hat), 1):.1%}"
    )
    print(f"方向一致率={np.mean(np.sign(lf[R]) == np.sign(lfc)):.1%}")

    null = ~intended
    print(
        f"假阳性={fp}   lfc 中位绝对误差={np.median(np.abs(lf[R] - lfc)):.5f}   "
        f"响应基因 padj 中位={np.median(padj[R]):.2e}   "
        f"null 基因 padj 最小={padj[null].min():.3f}"
    )
    print(
        f"外推 900 个 (pert,context): design {t_des * 900 / 60:.0f} min, "
        f"de_table {t_de * 900 / 60:.0f} min (单核)"
    )
    ok = tp == len(R) == len(R_hat) and fp == 0
    print(f"\n判定: {'PASS' if ok else 'FAIL'} (召回/精确率/假阳性 三项全中)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
