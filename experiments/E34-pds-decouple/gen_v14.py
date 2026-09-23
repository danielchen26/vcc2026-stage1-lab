"""V14 生成器：build_v13.py + 单旋钮「集外等比压幅」（F29 纪律）。

唯一语义旋钮：把集外向量等比缩放到 `CAP_MARGIN * min|lfc_t|`，使
`max|off| < min|lfc_t|` —— 召集集在 |lfc| 排序上整体领先集外，纯前缀只由召集集决定。

⚠️ 基准是 `min|lfc_t|` 即 `min|LAMBDA * b[sel]|`，也就是**提交出去的**召集集幅度，
不是探针里的原始 |b|。打分器看到的排序是提交值的排序（这是 V13 踩过的坑的镜像：
RESULT.md 错处 1 —— 排序不在 r_set 上，而在全部非零 lfc 的幅度上）。

逐字继承 V13 且不得改动：K_OFF = 550、LAMBDA_OFF = 1.0、LAMBDA = 0.7、K = 288、
排序统计量 |beta|、N_PERT = 8、SEED = 0、hamilton、force_mean = True。

参数出处：SPEC.md §8b —— 压幅后重扫 k_off，550 在两种 regime 下都最优；
不变量在全部 6 个 k 上都 8/8。CAP_MARGIN = 0.99 出自 cap_probe.py 的 K1 行。

跑法：/Users/chetianc/vcc2026/.venv/bin/python experiments/E34-pds-decouple/gen_v14.py
"""
from __future__ import annotations

import difflib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "build_v13.py"
DST = HERE / "build_v14.py"

# ---- 唯一语义替换：在 top-K 截断之后、组装 la_vec 之前插入等比压幅 ----------
OLD = (
    "        if K_OFF < off.size:\n"
    "            off[np.argsort(np.abs(off))[::-1][K_OFF:]] = 0.0\n"
    "        la_vec = np.zeros(ref.G)\n"
)
NEW = (
    "        if K_OFF < off.size:\n"
    "            off[np.argsort(np.abs(off))[::-1][K_OFF:]] = 0.0\n"
    "        # V14 唯一改动：等比压幅，使 max|off| < min|lfc_t| —— 恢复 |lfc| 排序，\n"
    "        # 让纯前缀重新只由召集集决定（SPEC §8a；V13 在 8/8 个扰动上破了这条）。\n"
    "        # 基准是**提交值** min|LAMBDA*b[sel]|，不是原始 |b|。\n"
    "        CAP_MARGIN = 0.99\n"
    "        mn_call = float(np.abs(lfc_t).min())\n"
    "        mx_off = float(np.abs(off).max())\n"
    "        if mx_off > 0.0:\n"
    "            off *= (CAP_MARGIN * mn_call) / mx_off\n"
    "        assert np.abs(off).max() < mn_call, (\n"
    "            f'压幅后仍越界: max|off|={np.abs(off).max():.6f} '\n"
    "            f'>= min|lfc_t|={mn_call:.6f}')\n"
    "        la_vec = np.zeros(ref.G)\n"
)

TAG_OLD = '("pred_v13", X_pred)'
TAG_NEW = '("pred_v14", X_pred)'
MSG_OLD = '不能同时放 _ctrl 与 pred_v13'
MSG_NEW = '不能同时放 _ctrl 与 pred_v14'


def _sub(txt: str, old: str, new: str, label: str) -> str:
    assert txt.count(old) == 1, f"{label}：锚点出现 {txt.count(old)} 次，须恰好 1 次"
    out = txt.replace(old, new)
    assert out.count(new) == 1, f"{label}：新串出现 {out.count(new)} 次，须恰好 1 次"
    if old in new:
        assert out.count(old) == 1, f"{label}：追加式替换后旧串应恰好 1 次（在新串内）"
    else:
        assert old not in out, f"{label}：旧串仍在"
    assert len(out) - len(txt) == len(new) - len(old), f"{label}：不止一处改动"
    return out


def generate() -> Path:
    assert SRC.exists(), f"缺 {SRC} —— 先跑 gen_v13.py"
    src = SRC.read_text()
    txt = _sub(src, OLD, NEW, "语义 等比压幅")
    txt = _sub(txt, TAG_OLD, TAG_NEW, "记账 提交标签")
    txt = _sub(txt, MSG_OLD, MSG_NEW, "记账 日志文字")

    # 冻结量逐字核验
    for frozen in ("K_OFF, LAMBDA_OFF = 550, 1.0", "LAMBDA = 0.7 ", "K = 288"):
        assert src.count(frozen) == txt.count(frozen) == 1, f"冻结量 {frozen!r} 被改动"
    assert "force_mean" not in txt, "不得触碰 force_mean（那是 E31 的旋钮）"
    # V13 的 5 条硬断言必须全部还在
    for a in ("召集集 lfc 被污染", "召集集本身变了", "集外通道落到了召集集上"):
        assert a in txt, f"V13 的断言 {a!r} 丢了"

    DST.write_text(txt)
    hunks = [l for l in difflib.unified_diff(
        src.splitlines(), txt.splitlines(), lineterm="", n=0) if l.startswith("@@")]
    a, b = src.splitlines(), txt.splitlines()
    print(f"已生成 {DST}")
    print(f"行数 {len(a)} -> {len(b)}（+{len(b)-len(a)}）  hunk 数 {len(hunks)}")
    for h in hunks:
        print("  " + h)
    return DST


if __name__ == "__main__":
    generate()
