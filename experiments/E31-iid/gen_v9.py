"""V9 生成器：build_v8.py + 恰好一次 str.replace（F29 单旋钮纪律）。

唯一旋钮：`design_cells(..., force_mean=False)` —— 不再把抽到的 400 个对照细胞的
经验列均值钉死到目标 profile，改用对照**总体**均值 `ref.m_full` 做乘性平移，
让均值自身的抽样波动原样保留。lambda 仍 0.7，K 仍 288，排序仍 |beta|，
N_PERT 仍 8，seed 仍 0 —— 全部逐字继承 V8。
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[0] / "E28-pds" / "build_v8.py"
DST = HERE / "build_v9.py"

OLD = "n_cells=VCC_PERT_CELLS, seed=SEED)))"
NEW = ("n_cells=VCC_PERT_CELLS, seed=SEED,\n"
       "                                                 force_mean=False)))")


def generate() -> Path:
    src = SRC.read_text()
    assert src.count(OLD) == 1, f"锚点不唯一：{src.count(OLD)} 次"
    out = src.replace(OLD, NEW)
    # 替换确实落地：旧串消失、新串出现、且只差这一处
    assert OLD not in out, "旧串仍在，替换未落地"
    assert NEW in out, "新串不在，替换未落地"
    assert out != src, "文件未变"
    assert len(out) - len(src) == len(NEW) - len(OLD), "不止一处改动"
    DST.write_text(out)
    return DST


if __name__ == "__main__":
    p = generate()
    print(f"已生成 {p}（{len(p.read_text().splitlines())} 行）")
