"""V16 生成器：build_v13.py + 单旋钮 `force_mean=False`（F29 纪律）。

为什么这是现在最重要的一次 build（[F39](../../docs/02-findings.md#f39) ⑤.2）：
  V13（官方 lo 0.1508）与 V14（0.1158）都在 `force_mean=True` 下测的，
  而那条路径被打分器按 0.484× 打折并标记为「值得查看」。V9 证明关掉它要付 −0.0435，
  且代价**恰好落在** `jac` 与 `fid` —— 也就是 F35 指认的那批同号重归一伪影上。

  V13 的 +0.0293 全部来自 `pds`（+0.2927）与 `fid`（+0.0361）。其中 `fid` 那一份
  很可能与伪影同源，会在合法 regime 下消失；而 `pds` 的那一份来自 `lfc_all` 通道，
  机制上与 `force_mean` 无关，应当保留。**两者哪个对，决定最终提交用哪个配置。**

唯一语义旋钮：`design_cells(..., force_mean=False)`。
逐字继承 V13 且不得改动：K_OFF = 550、LAMBDA_OFF = 1.0、LAMBDA = 0.7、K = 288、
排序统计量 |beta|、N_PERT = 8、SEED = 0、quantizer = hamilton、DEPTH、ALPHA、K_A 表。

跑法：/Users/chetianc/vcc2026/.venv/bin/python experiments/E36-legal-regime/gen_v16.py
"""
from __future__ import annotations

import difflib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[0] / "E34-pds-decouple" / "build_v13.py"
DST = HERE / "build_v16.py"

# ---- 唯一语义替换 ----------------------------------------------------------
OLD = ("                                                 n_cells=VCC_PERT_CELLS, seed=SEED,\n"
       "                                                 lfc_all=la_vec)))\n")
NEW = ("                                                 n_cells=VCC_PERT_CELLS, seed=SEED,\n"
       "                                                 lfc_all=la_vec,\n"
       "                                                 force_mean=False)))\n")

# ---- 记账替换 --------------------------------------------------------------
OUT_OLD = 'OUT = Path(__file__).resolve().parents[1] / "E34-pds-decouple" / "out"'
OUT_NEW = 'OUT = Path(__file__).resolve().parents[1] / "E36-legal-regime" / "out"'
TAG_OLD = '("pred_v13", X_pred)'
TAG_NEW = '("pred_v16", X_pred)'
MSG_OLD = '不能同时放 _ctrl 与 pred_v13'
MSG_NEW = '不能同时放 _ctrl 与 pred_v16'


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
    assert SRC.exists(), f"缺 {SRC} —— 先跑 E34 的 gen_v13.py"
    src = SRC.read_text()
    assert "force_mean" not in src, "模板里已出现 force_mean，旋钮不唯一"
    txt = _sub(src, OLD, NEW, "语义 force_mean=False")
    txt = _sub(txt, OUT_OLD, OUT_NEW, "记账 输出目录")
    txt = _sub(txt, TAG_OLD, TAG_NEW, "记账 提交标签")
    txt = _sub(txt, MSG_OLD, MSG_NEW, "记账 日志文字")

    for frozen in ("K_OFF, LAMBDA_OFF = 550, 1.0", "LAMBDA = 0.7 ", "K = 288"):
        assert src.count(frozen) == txt.count(frozen) == 1, f"冻结量 {frozen!r} 被改动"
    for a in ("召集集 lfc 被污染", "召集集本身变了", "集外通道落到了召集集上"):
        assert a in txt, f"V13 的断言 {a!r} 丢了"
    assert txt.count("force_mean=False") == 1, "force_mean 旋钮不唯一"

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
