"""V13 生成器：build_v8.py + 单旋钮「集外去面板均值通道」（F29 纪律）。

唯一语义旋钮 = 给 `design_cells(...)` 传 `lfc_all=`，即打开 `decoder.py:119-128`
那条一阶矩通道。它由两处替换共同构成（常量/面板均值 + 调用点），两处各自断言：

  1/2  常量与面板均值：K_OFF = 550、LAMBDA_OFF = 1.0、Bd_panel = Bc - 逐基因跨扰动均值
  2/2  调用点：算出集外 top-K_OFF 的去均值 lfc 向量，传给 lfc_all

逐字继承 V8 且**不得改动**：LAMBDA = 0.7（召集集）、K = 288 扁平、排序统计量 |beta|、
N_PERT = 8、SEED = 0、quantizer = hamilton、force_mean = True、DEPTH、ALPHA、K_A 表。

参数出处：experiments/E34-pds-decouple/SPEC.md §0e（密网格平台 500/550/600 取中心 550，
留一法 8/8；λ_off 平台取 1.0）。**这两个数字在 SPEC 里是 build 前定死的。**

关键不变量：`decoder.py:128` 的 `lf[ref.gidx[r_set]] = lfc` 在 `lfc_all` **之后**执行，
所以召集集那 288 个 lfc 逐位等于 V8 —— 这是 `sig_jaccard` / `direction_reach` 不该变的
结构性理由（SPEC §1）。

跑法：/Users/chetianc/vcc2026/.venv/bin/python experiments/E34-pds-decouple/gen_v13.py
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[0] / "E28-pds" / "build_v8.py"
DST = HERE / "build_v13.py"

# ---- 替换 1/2（旋钮的常量部分）----------------------------------------------
SETUP_OLD = '    print(f"共同基因 {len(common):,}")\n'
SETUP_NEW = (
    '    print(f"共同基因 {len(common):,}")\n'
    '    # V13 唯一改动（1/2）：集外去面板均值通道的常量与面板均值。\n'
    '    # 取值出处 E34-pds-decouple/SPEC.md §0e（build 前定死，不许事后改）。\n'
    '    K_OFF, LAMBDA_OFF = 550, 1.0\n'
    '    Bc_panel = np.where(np.isfinite(B), B, 0.0)\n'
    '    Bd_panel = Bc_panel - Bc_panel.mean(1, keepdims=True)\n'
    '    print(f"集外通道 K_OFF={K_OFF} LAMBDA_OFF={LAMBDA_OFF} '
    'Bd 非零 {np.count_nonzero(Bd_panel):,}")\n'
)

# ---- 替换 2/2（旋钮的调用点）------------------------------------------------
CALL_OLD = (
    "        lfc_t = LAMBDA * b[sel]\n"
    "        X_pred.append(sp.csr_matrix(design_cells(ref, r_set, lfc_t,\n"
    "                                                 n_cells=VCC_PERT_CELLS, seed=SEED)))\n"
)
CALL_NEW = (
    "        lfc_t = LAMBDA * b[sel]\n"
    "        # V13 唯一改动（2/2）：集外 top-K_OFF 的去面板均值 lfc -> lfc_all 通道。\n"
    "        off = Bd_panel[:, j].copy()\n"
    "        off[sel] = 0.0                     # 召集集不进这条通道\n"
    "        if K_OFF < off.size:\n"
    "            off[np.argsort(np.abs(off))[::-1][K_OFF:]] = 0.0\n"
    "        la_vec = np.zeros(ref.G)\n"
    "        la_vec[gi_g] = LAMBDA_OFF * off\n"
    "        # 硬断言：不通过就不许出 .h5ad（SPEC §2）\n"
    "        assert np.array_equal(lfc_t, LAMBDA * b[sel]), '召集集 lfc 被污染'\n"
    "        assert np.array_equal(r_set, gi_g[sel]), '召集集本身变了'\n"
    "        assert la_vec.shape == (ref.G,), f'lfc_all 形状 {la_vec.shape} != {(ref.G,)}'\n"
    "        assert np.count_nonzero(la_vec) <= K_OFF, '集外通道开得比 K_OFF 多'\n"
    "        assert not np.any(la_vec[gi_g[sel]]), '集外通道落到了召集集上'\n"
    "        X_pred.append(sp.csr_matrix(design_cells(ref, r_set, lfc_t,\n"
    "                                                 n_cells=VCC_PERT_CELLS, seed=SEED,\n"
    "                                                 lfc_all=la_vec)))\n"
)

# ---- 记账替换（无计算影响，各自断言）----------------------------------------
OUT_OLD = 'OUT = Path(__file__).resolve().parents[1] / "E28-pds" / "out"'
OUT_NEW = 'OUT = Path(__file__).resolve().parents[1] / "E34-pds-decouple" / "out"'
TAG_OLD = '("pred_v8", X_pred)'
TAG_NEW = '("pred_v13", X_pred)'
MSG_OLD = '不能同时放 _ctrl 与 pred_v8'
MSG_NEW = '不能同时放 _ctrl 与 pred_v13'


def _sub(txt: str, old: str, new: str, label: str) -> str:
    """恰好一次替换，且替换确实落地。

    ⚠️ 追加式替换（`new` 以 `old` 开头，例如在某行后面插入几行）会让朴素的
    `old not in out` 永假 —— E32 的 gen_build.py 靠给旧串加尾随空格绕过，
    这里改成结构化判定：`old` 是 `new` 的子串时，只要求出现次数不增。
    """
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
    src = SRC.read_text()
    txt = _sub(src, SETUP_OLD, SETUP_NEW, "语义 1/2 常量")
    txt = _sub(txt, CALL_OLD, CALL_NEW, "语义 2/2 调用点")
    txt = _sub(txt, OUT_OLD, OUT_NEW, "记账 输出目录")
    txt = _sub(txt, TAG_OLD, TAG_NEW, "记账 提交标签")
    txt = _sub(txt, MSG_OLD, MSG_NEW, "记账 日志文字")

    # 冻结量的逐字核验：这些行必须与模板完全一致，否则不是单旋钮
    for frozen in ('LAMBDA = 0.7 ', 'K = 288', 'np.abs(b)', 'quantizer'):
        if frozen in src:
            assert src.count(frozen) == txt.count(frozen), f"冻结量 {frozen!r} 被改动"
    assert 'force_mean' not in txt, "不得触碰 force_mean（那是 E31 的旋钮）"

    DST.write_text(txt)
    print(f"已生成 {DST}")
    # 审计轨迹必须是真的：朴素的逐行 zip 在插入之后整体错位，会把「被触碰的行」
    # 报成几十行（第一次就这样报了 80 行）。改用 difflib 报真实 hunk。
    import difflib
    hunks = [l for l in difflib.unified_diff(
        src.splitlines(), txt.splitlines(), lineterm="", n=0) if l.startswith("@@")]
    a, b = src.splitlines(), txt.splitlines()
    print(f"行数 {len(a)} -> {len(b)}（+{len(b)-len(a)}）  hunk 数 {len(hunks)}")
    for h in hunks:
        print("  " + h)
    return DST


if __name__ == "__main__":
    generate()
