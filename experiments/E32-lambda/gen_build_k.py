"""从 E28-pds/build_v8.py 生成 K 旋钮变体（build_v13.py = K25、build_v14.py = K10）。

与 gen_build.py 同样的纪律：唯一的**语义**改动是 K 字面量，恰好一次 str.replace，
替换后断言旧字面量消失、新字面量出现。LAMBDA 保持 0.7，ordering 保持 |beta|，
N_PERT=8，seed=0，quantizer=hamilton，一律不动。
其余替换是记账性的（输出目录、提交标签、过时注释），各自单独标注、无计算影响。

依据（SourceUnion 的已验证 reach 估计器，K562GW |beta| 排序对真实 lfc 的符号一致率）：
  k=1 87.5%   k=3 83.3% (p=0.0008)   k=10 65.0% (p=0.0048)   k=25 55.0%   k=288 52.4%（等于掷硬币）
即现行 K=288 把约 260 个掷硬币基因混进了排序前缀。

用法：/Users/chetianc/vcc2026/.venv/bin/python gen_build_k.py 13 25
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "experiments" / "E28-pds" / "build_v8.py"
HERE = Path(__file__).resolve().parent

# 唯一出现处是 build_v8.py:188。带尾随空格，保证 K=2880 之类不会被误判为"已消失"。
KNOB_OLD = "K = 288 "
CMT_OLD = "# V7 唯一改动：扁平 K=288，取代 E24 的 29/288/G 先验分区"
OUTDIR_OLD = '/ "E28-pds" / "out"'
TAG_OLD = '("pred_v8", X_pred)'


def gen(ver: str, k: str) -> Path:
    src = SRC.read_text()

    # ---- 替换 1/1（唯一语义旋钮）：召集规模 K ---------------------------
    knob_new = f"K = {k} "
    txt = src.replace(KNOB_OLD, knob_new)
    assert txt != src, f"旋钮替换未命中：模板里找不到 {KNOB_OLD!r}"
    assert KNOB_OLD not in txt, f"旋钮替换未落地：旧字面量 {KNOB_OLD!r} 仍在"
    assert knob_new in txt, f"旋钮替换未落地：新字面量 {knob_new!r} 不在"
    assert "LAMBDA = 0.7 " in txt, "LAMBDA 被意外改动，K 变体必须保持 0.7"

    # ---- 记账替换 A（仅注释，无计算影响） -------------------------------
    cmt_new = (f"# V{ver} 唯一改动：召集规模 288 -> {k}"
               f"（源排序符号一致率在 k~25-50 衰减到掷硬币）")
    before = txt
    txt = txt.replace(CMT_OLD, cmt_new)
    assert txt != before and CMT_OLD not in txt and cmt_new in txt, "注释替换未落地"

    # ---- 记账替换 B（仅输出目录，无计算影响） ---------------------------
    outdir_new = '/ "E32-lambda" / "out"'
    before = txt
    txt = txt.replace(OUTDIR_OLD, outdir_new)
    assert txt != before and OUTDIR_OLD not in txt and outdir_new in txt, "目录替换未落地"

    # ---- 记账替换 C（仅提交文件标签，无计算影响） -----------------------
    tag_new = f'("pred_v{ver}", X_pred)'
    before = txt
    txt = txt.replace(TAG_OLD, tag_new)
    assert txt != before and TAG_OLD not in txt and tag_new in txt, "标签替换未落地"

    dst = HERE / f"build_v{ver}.py"
    dst.write_text(txt)

    a, b = src.splitlines(), txt.splitlines()
    assert len(a) == len(b), "行数变了，说明替换越界"
    diff = [i + 1 for i, (x, y) in enumerate(zip(a, b)) if x != y]
    assert len(diff) == 3, f"被改动的行数应为 3，实际 {len(diff)}：{diff}"
    print(f"已生成 {dst}  K={k}  改动行号 {diff}")
    return dst


if __name__ == "__main__":
    gen(sys.argv[1], sys.argv[2])
