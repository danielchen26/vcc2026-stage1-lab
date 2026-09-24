"""E35 第二部分 · 82% 的跨系符号一致率里，有多少是**扰动特异**的？

第一部分（`sign_ceiling.py`）实测跨系符号一致率 0.67–0.82，远高于 F33 报的 0.51–0.57。
差别的来源已定位：F33 在**全部重叠基因**上算（绝大多数是零效应基因，符号本就是掷硬币），
而指标的门是**逐扰动的 real 侧显著集** —— 这一条是 F33 自己确立的，但它的符号表没用这个门。

⚠️ 但 0.67–0.82 **不能**直接读成「符号可迁移」。[T12](../../docs/06-traps.md#t12) 的教训是：
不要拿实测值和**假设**的无技巧点比。这里 0.5 几乎肯定不是无技巧点 —— 若多数显著基因在
两个细胞系里都朝同一个方向（生长/适应度的共同效应），那一致率会很高而**完全不含扰动特异
信息**，与 [F35](../../docs/02-findings.md#f35) 在 MAT2A 上定性的 marginal-sign exploit 同形，
也正是 `competition.py:327-330` 记录的那个 exploit。

三个对照，同一个门、同一批基因，只换「用哪个扰动的源侧 lfc」：

  actual    用 a 侧**同一个**扰动的 lfc 符号                      <- 含扰动特异 + 共同方向
  permuted  用 a 侧**另一个随机**扰动的 lfc 符号（R 次取均值）      <- 只含共同方向
  habitual  用 a 侧该基因**跨全部扰动的中位符号**                  <- 只含共同方向，最强形式

**扰动特异信号 = actual − permuted。** 若它 ≈ 0，则 F33/F35 的 NO-GO 结论成立
（只是当初的理由和数字都错了）；若它显著为正，则 `nmae`/`reach` 的前提被重开。

判据（执行前写下）：
  - actual − permuted ≥ 0.05（5 个百分点）且逐扰动 t > 5 ⇒ 存在扰动特异符号信号，重开前提。
  - actual − permuted < 0.02 ⇒ 82% 全是共同方向，NO-GO 成立，且**共同方向本身是可利用的**
    （但那是 exploit，`competition.py` 明确在防，且 F35 的 oracle 上界已把 reach 那条堵死）。
  - 中间：报告，不下结论。

跑法：/Users/chetianc/vcc2026/.venv/bin/python experiments/E35-sign-ceiling/marginal_control.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sign_ceiling import ALPHA, PAIRS, REF, bh, cols_of, load  # noqa: E402

OUT = HERE / "out"
N_PERM, SEED = 20, 20260924


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    ref_cols = set(cols_of(REF))
    report = {}

    for a, b, label in PAIRS:
        shared = sorted(ref_cols.intersection(cols_of(b)))
        if not shared:
            continue
        ga, la = load(a, shared, "lfc")
        gb, lb = load(b, shared, "lfc")
        _, pb = load(b, shared, "p")

        def _dedup(g):
            _, first = np.unique(g, return_index=True)
            return np.sort(first)
        ia, ib = _dedup(ga), _dedup(gb)
        ga, la = ga[ia], la[ia]
        gb, lb, pb = gb[ib], lb[ib], pb[ib]
        common, ja, jb = np.intersect1d(ga, gb, return_indices=True)
        ga, la, lb, pb = common, la[ja], lb[jb], pb[jb]

        sa, sb = np.sign(la), np.sign(lb)
        # habitual：a 侧每个基因跨全部共有扰动的中位符号（不含任何扰动特异信息）
        with np.errstate(invalid="ignore"):
            hab = np.sign(np.nanmedian(la, axis=1))
        both = np.isfinite(la) & np.isfinite(lb)
        qb = bh(pb)
        n_p = len(shared)

        rows = {"actual": [], "permuted": [], "habitual": []}
        for j, pert in enumerate(shared):
            gate = both[:, j] & np.isfinite(qb[:, j]) & (qb[:, j] < ALPHA) & (ga != pert)
            n = int(gate.sum())
            if n < 10:
                continue
            tb = sb[gate, j]
            rows["actual"].append(float((sa[gate, j] == tb).mean()))
            rows["habitual"].append(float((hab[gate] == tb).mean()))
            # permuted：a 侧换成别的扰动列，重复 N_PERM 次取均值（同门、同基因）
            acc = []
            for _ in range(N_PERM):
                k = int(rng.integers(n_p - 1))
                k = k + 1 if k >= j else k                      # k != j
                col = sa[gate, k]
                ok = np.isfinite(la[gate, k])
                if ok.sum() >= 10:
                    acc.append(float((col[ok] == tb[ok]).mean()))
            if acc:
                rows["permuted"].append(float(np.mean(acc)))

        if not rows["actual"]:
            continue
        A_ = np.asarray(rows["actual"])
        P_ = np.asarray(rows["permuted"][:A_.size])
        H_ = np.asarray(rows["habitual"])
        m = min(A_.size, P_.size)
        d = A_[:m] - P_[:m]
        t = float(d.mean() / (d.std(ddof=1) / np.sqrt(d.size))) if d.size > 1 else float("nan")
        report[label] = {"pair": f"{a}->{b}", "n_scored": int(A_.size),
                         "actual": float(A_.mean()), "permuted": float(P_.mean()),
                         "habitual": float(H_.mean()),
                         "specific": float(d.mean()), "specific_sd": float(d.std(ddof=1)),
                         "t_specific": t,
                         "frac_specific_positive": float((d > 0).mean())}
        r = report[label]
        print(f"\n=== {label}  n={r['n_scored']:,} ===")
        print(f"  actual   （同一扰动）        {r['actual']:.4f}")
        print(f"  permuted （随机别的扰动×{N_PERM}） {r['permuted']:.4f}")
        print(f"  habitual （基因跨扰动中位符号） {r['habitual']:.4f}")
        print(f"  ⇒ 扰动特异 = actual − permuted = **{r['specific']:+.4f}**"
              f"  (sd {r['specific_sd']:.4f}, t = {r['t_specific']:.1f},"
              f" 正的占比 {r['frac_specific_positive']:.3f})")

    print("\n" + "=" * 96)
    print("判读（门在执行前写下）")
    print("=" * 96)
    for label, r in report.items():
        verdict = ("存在扰动特异符号信号 ⇒ 重开前提" if r["specific"] >= 0.05 and r["t_specific"] > 5
                   else "共同方向解释了全部 ⇒ NO-GO 成立" if r["specific"] < 0.02
                   else "中间，不下结论")
        print(f"{label:28s} specific {r['specific']:+.4f}  t {r['t_specific']:6.1f}  → {verdict}")
    json.dump(report, open(OUT / "marginal_control.json", "w"), indent=2, ensure_ascii=False)
    print(f"\n已存 {OUT/'marginal_control.json'}   耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
