"""从 E28-pds/build_v8.py 生成 build_v10.py / build_v11.py。

单旋钮纪律（F29）：唯一的**语义**改动是 LAMBDA 字面量，由恰好一次 str.replace
完成，并在替换后断言旧字面量消失、新字面量出现。其余替换全部是"记账性"的
（输出目录、提交文件标签、过时注释），不触碰任何计算路径，各自单独标注。

用法：/Users/chetianc/vcc2026/.venv/bin/python gen_build.py 10 0.85
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "experiments" / "E28-pds" / "build_v8.py"
HERE = Path(__file__).resolve().parent

# 旧字面量带一个尾随空格，这样 lam=0.775 时 "LAMBDA = 0.7 " 仍能被判定为"已消失"
# （若不带空格，"LAMBDA = 0.775" 会把 "LAMBDA = 0.7" 当作子串，断言就失效了）。
KNOB_OLD = "LAMBDA = 0.7 "
CMT_OLD = "# V8 唯一改动：lfc 收缩系数 0.5 -> 0.7"
OUTDIR_OLD = '/ "E28-pds" / "out"'
TAG_OLD = '("pred_v8", X_pred)'


def gen(ver: str, lam: str) -> Path:
    src = SRC.read_text()

    # ---- 替换 1/1（唯一语义旋钮）：lfc 收缩系数 --------------------------
    knob_new = f"LAMBDA = {lam} "
    txt = src.replace(KNOB_OLD, knob_new)
    assert txt != src, f"旋钮替换未命中：模板里找不到 {KNOB_OLD!r}"
    assert KNOB_OLD not in txt, f"旋钮替换未落地：旧字面量 {KNOB_OLD!r} 仍在"
    assert knob_new in txt, f"旋钮替换未落地：新字面量 {knob_new!r} 不在"

    # ---- 记账替换 A（仅注释，无计算影响）：LAMBDA 行的过时注释 ----------
    cmt_new = (f"# V{ver} 唯一改动：lfc 收缩系数 0.7 -> {lam}"
               f"（lambda 曲线 bisection 探点）")
    before = txt
    txt = txt.replace(CMT_OLD, cmt_new)
    assert txt != before, f"注释替换未命中：{CMT_OLD!r}"
    assert CMT_OLD not in txt and cmt_new in txt, "注释替换未落地"

    # ---- 记账替换 B（仅输出目录，无计算影响） ---------------------------
    outdir_new = '/ "E32-lambda" / "out"'
    before = txt
    txt = txt.replace(OUTDIR_OLD, outdir_new)
    assert txt != before, f"输出目录替换未命中：{OUTDIR_OLD!r}"
    assert OUTDIR_OLD not in txt and outdir_new in txt, "输出目录替换未落地"

    # ---- 记账替换 C（仅提交文件标签，无计算影响） -----------------------
    tag_new = f'("pred_v{ver}", X_pred)'
    before = txt
    txt = txt.replace(TAG_OLD, tag_new)
    assert txt != before, f"标签替换未命中：{TAG_OLD!r}"
    assert TAG_OLD not in txt and tag_new in txt, "标签替换未落地"

    # 注意：首行 docstring 仍写着"变体 V8"。那是纯文档串，再加一次替换只会增噪，
    # 故保留原样并在 RESULT.md 中说明。
    dst = HERE / f"build_v{ver}.py"
    dst.write_text(txt)

    # 事后自证：生成文件与模板的差异行数必须恰好等于被触碰的 3 行
    # （LAMBDA 行同时含旋钮与注释，OUT 行，tag 行）。
    a, b = src.splitlines(), txt.splitlines()
    assert len(a) == len(b), "行数变了，说明替换越界"
    diff = [i + 1 for i, (x, y) in enumerate(zip(a, b)) if x != y]
    assert len(diff) == 3, f"被改动的行数应为 3，实际 {len(diff)}：{diff}"
    print(f"已生成 {dst}  LAMBDA={lam}  改动行号 {diff}")
    return dst


if __name__ == "__main__":
    gen(sys.argv[1], sys.argv[2])
