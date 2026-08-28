# E04 — RESULT

**结果获得于 2026-08-27**. 机器: Apple M1 Pro / 16 GB / **无 CUDA** (arm64 Darwin;
官方 DE 因此走 `device="cpu"` + `backend="scanpy"`).
版本: **cell-eval2 0.16.0** (preset `vcc2026`), numpy 2.5.2, scipy 1.18.1,
pandas 3.0.5, h5py 3.16.0, anndata 0.13.3, Python 3.12.0.
对象: context_A, 3 个扰动 (ABCD1 / ACLY / ADNP), 每侧 19,600 个细胞
(18,400 对照 + 3 × 400 构造).

## 1. gate 与 DE 表

```
gate 基因集      : 9,929, 与 ControlRef.gidx 完全一致
官方 DE 表       : 29,787 行 (= 3 × 9,929)
log2_fold_change : 最大绝对差 1.007e-5 / 1.008e-5 / 1.019e-5   (三个扰动)
log10(p_adj)     : 中位绝对差 0.0000
```

显著集 (p_adj < 0.05) 大小与对称差:

```
扰动      官方  我们(含并列校正)  我们(未校正)  对称差
ABCD1     251        251            249         0
ACLY      250        250            249         0
ADNP      251        251            250         0
```

-> 含并列校正: **3/3 对称差 = 0**. 未校正: 分别少判 2 / 1 / 1 个基因, 且不报错.

## 2. 官方六指标 (`compute_metrics`, preset `vcc2026`)

```
de_wilcoxon_sig_jaccard = 0.249377   (三组相同; = 100/401, 与设计的 100 个重叠吻合)
direction_fidelity      = 0.49004 / 0.466135 / 0.517928   (方向独立随机 -> ~0.5)
direction_reach         = 0                               (纯度到不了 0.9, 行为正确)
```

## 3. 速度

```
官方单侧 DE 表 : 296.85 s
我们           :   7.66 s
比值           : 38.8x        (op 计数的理论比 47.1x; 全 panel 摊销后 44.8x)
外推全 panel   : 官方 52 小时  vs  我们 60 min 单核 / 6 min 十核
```

## 4. 2026-08-28 脚本固化与验证状况

`run.py` 是对 2026-08-27 那次跑法 (`parity.py` + `dump_de.py` + 手工构造)
的整理与固化, 并把 pred/real 的构造做成固定种子可复现.

**按验收要求, 官方打分器那 ~10 分钟没有重跑**; 本次实际验证到的部分:

```
=== E04 官方打分器逐基因对齐  [dry-run] ===
日期=2026-08-28  机器=arm64 Darwin 3.12.0  numpy=2.5.2 scipy=1.18.1
数据=/Users/chetianc/vcc2026  基线=vcc_local.py

ControlRef 加载 21.7s   gate=9929   对照=18400
解码器构造 6 个扰动块: 2.21s
写出 .../experiments/E04-official-parity/out/parity_pred.h5ad  (19600 个细胞)
写出 .../experiments/E04-official-parity/out/parity_real.h5ad  (19600 个细胞)
我们的 DE (3 个扰动 x2 口径): 28.19s
preset vcc2026: de.backend=scanpy pert_col=target_gene

[dry-run] 跳过官方 compute_de / compute_metrics (~10 min).
```

即: 两个 h5ad 构造成功 (19,600 × 18,533, 布局与 2026-08-27 用的
`parity_pred.h5ad` 一致 —— 对照原始计数在前、行和 1e6 的构造细胞在后),
我们两个口径的 DE 跑通, 且 `EvalConfig.from_preset("vcc2026")` +
`compute_de` / `compute_metrics` 的导入与字段名全部有效.

余下那段「官方 DE 表 -> 逐基因对齐」的 plumbing, 用 2026-08-27 存档的官方
parquet (`~/vcc2026/de_pred_official.parquet`) 单独验证过:

```
ABCD1 rows 9929 gate 9929 |sig| 251 lfc range -2.238..2.170
ACLY  rows 9929 gate 9929 |sig| 250 lfc range -2.222..2.159
ADNP  rows 9929 gate 9929 |sig| 251 lfc range -2.232..2.180
plumbing OK: gate 基因集与官方 feature 集完全一致 (9929, 无缺无重)
```

这独立复核了上表「官方 251 / 250 / 251」与 gate = 9,929 两个数字.

产物 `out/` (两个 h5ad 共约 1.8 GB) 跑完已删除; `.gitignore` 里 `*.h5ad`
与 `*.parquet` 保证它们不会入库.
