# E04 — 与官方 cell-eval2 逐基因对齐

**状态**: 已完成 (2026-08-27), 三项全对: gate 一致 / DE 一致 / 指标可预判
**耗时**: **完整跑约 10 分钟** —— 官方单侧 DE 表 296.85 s + `compute_metrics`
约 5 min, 我们自己那部分 < 30 s. 日常迭代请用 `--dry-run` (~60 s, 跳过两个
官方调用, 只验证构造 + 我们的 DE + 官方 API 签名).

## 目的

E01–E03 全部建立在一个前提上: `vcc_local.ControlRef.de_table` **就是**官方
`cell-eval2` 0.16.0 (preset `vcc2026`) 的 DE. 如果这个前提有偏差, 前面三个
实验的结论全部作废. 本实验用官方打分器本体逐基因验证它, 并顺手量出加速比.

## 方法

因为目标细胞系没有任何扰动答案, "real" 一侧也由 Stage 2 解码器构造:

* 3 个扰动 (`pert_counts.csv` 前三个: ABCD1 / ACLY / ADNP), 每侧 250 个意图
  响应基因, **故意让 pred 与 real 共享 100 个**, 方向各自独立随机;
* 于是两个可预判的量: `jaccard = 100 / (250 + 250 - 100) = 0.25`,
  `direction_fidelity ≈ 0.5`;
* 写出两个 19,600 细胞的 h5ad (18,400 个对照原始计数在前 + 3×400 个构造细胞),
  产物落在 `out/` (h5ad / parquet 都被 `.gitignore` 拦掉, 绝不入库).

三层比对:

1. **gate**: 官方 `filter_gene_min_cpm_cell=5.0` 的 feature 集 vs `ControlRef.gidx`;
2. **DE**: 官方 `compute_de(backend="scanpy", ...)` 的 `log2_fold_change` /
   `p_adj` vs `de_table(tie_correct=True)`, 以及显著集 R̂ 的对称差;
   同时跑一遍 `tie_correct=False` 作为对照, 证明并列校正不是可选项;
3. **指标**: `compute_metrics` 的六个指标 vs 我们的预判.

`DE_KW` 里的 12 个参数与 preset `vcc2026` 的 DE 段逐字段抄齐 (`mean_calc`
= arithmetic, `epsilon` = 1e-9, `input_type` = counts, `target_sum` = 1e6,
`fdr_scope` = per_pert, `device` = cpu …), 任何一个不同都会让对齐失败.

## 实测结果

见 `RESULT.md`. 摘要:

* **gate 完全一致** (9,929); 官方 DE 表 29,787 行 = 3 × 9,929;
* `log2_fold_change` 最大绝对差 **1.007e-5 / 1.008e-5 / 1.019e-5**
  (float32 存储噪声量级), `log10(p_adj)` 中位绝对差 **0.0000**;
* 显著集 **对称差 = 0** (3/3), 但**必须**含并列校正: 未校正时 3 个扰动分别
  少判 2 / 1 / 1 个基因;
* `de_wilcoxon_sig_jaccard` = **0.249377** (三组相同, = 100/401, 与设计的
  100 个重叠吻合); `direction_fidelity` = 0.49004 / 0.466135 / 0.517928
  (方向独立随机 -> 0.5); `direction_reach` = 0 (方向随机, 纯度到不了 0.9,
  行为正确);
* 速度: 官方单侧 DE 表 **296.85 s** vs 我们 **7.66 s** = **38.8x**
  (op 计数给出的理论比 47.1x; 全 panel 摊销后 44.8x).
  外推: 官方全 panel **52 小时** vs 我们 60 min 单核 / 6 min 十核.

## 结论

1. `de_table` 是官方 DE 的**精确复刻**, E01–E03 的前提成立.
2. **并列校正是必需项**, 不做会静默少判基因 (每组 1–2 个), 且不报错.
   (另一个同类陷阱: BH 必须是 `np.minimum.accumulate(q[::-1])[::-1]`,
   写成 maximum 会让发现数从 250 掉到 0 —— 也不报错.)
3. 38.8x 的加速把「提交前用官方口径自评全 panel」从 52 小时压到 6 分钟,
   于是 Stage 1 的模型选择可以直接以官方指标为目标函数迭代.
4. `direction_reach = 0` 说明该指标是个**阈值型**指标 (纯度 ≥ 0.9 才给分),
   随机方向拿不到任何分 —— Stage 1 必须把方向做到高纯度才值得报, 这直接
   驱动了 E06 (共表达定符号) 与 E08 (报数 K 的取舍).
