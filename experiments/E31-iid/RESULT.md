# E31 · i.i.d. 伪影：修复已实现并本地验证，官方六指标分数未测

**头条（`measured`）**：`design_cells` 现在有 `force_mean` 开关。默认 `True` 与 HEAD **逐位相同**
（`np.array_equal` 对 `git show HEAD:src/vcclab/decoder.py` 的输出，见 §2），
`False` 把列均值的相对偏差 std 从 **0.00020 抬到 0.03487（173.2×）**，即恢复了打分器要找的抽样波动。

**未完成（`attempted, not measured`）**：V9 的官方六指标分数。build 成功（292 MB，358 s），
`compute_metrics` 跑了 27 分钟只累计 **13 秒 CPU**（page-fault bound，见 §4），被资源耗尽杀掉。

**尺度声明**：本文件引用的 `avg_score` 若为 0.2025 一类，均为**仓库内部刻度**
（`from_baseline`，分母是对照均值 tile 400 遍的退化基线）。官方两端刻度
$(u-b)/(r-b)$ 的数值一律标注 lo/hi 两端，无中点。

---

## 0 · 三级证据表

| 级别 | 内容 |
|---|---|
| ① `measured` | §2 的三项 sanity（默认路径逐位相同、`m_full` 恒等、173.2× 波动恢复）；§1 的 build 产物与耗时 |
| ② `inferred from measured` | §3：honest arm 在官方刻度上的 `expr_mse` 代价**可证为 0**，因为该成员地板在 0.0000 且让均值波动只能把 raw 推**高** |
| ③ 需要机器 | V9 的六指标 raw 与 `avg_score`；`pds_cosine` 是唯一可能被真正损害的成员 |

---

## 1 · 复现命令

```bash
cd experiments/E31-iid
PY=/Users/chetianc/vcc2026/.venv/bin/python
$PY sanity.py                 # 19.9 s，三项验证，输出存 out/sanity.txt
$PY gen_v9.py                 # 由 build_v8.py 单处 str.replace 生成 build_v9.py
$PY build_v9.py build         # 292 MB gzip 提交
$PY score_e31.py v9           # ⚠️ 本机跑不完，见 §4
```

单旋钮生成（F29 纪律）：`gen_v9.py` 读 `build_v8.py` 原文，一处 `str.replace` 把
`design_cells(ref, r_set, lfc_t, n_cells=VCC_PERT_CELLS, seed=SEED)` 换成同一调用加
`force_mean=False`，并 `assert` 旧串已消失、新串存在。
λ 仍 0.7、K 仍扁平 288、排序仍 $\lvert\beta\rvert$、N_PERT 仍 8、seed 仍 0、量化器仍 `hamilton`。

build 实测（`out/chain.log`）：gate 10,779、对照 18,400、MDE 中位 0.0149、共同基因 7,016、
8 个扰动各 $\lvert\hat R\rvert$ = 288、`pred (21600, 18080)` = 292 MB、耗时 358 s。

---

## 2 · 三项 sanity（`measured`，`out/sanity.txt`，19.9 s）

```
[1] max|m_full - _cpm_csr.mean(0)| = 0.000e+00   m_full.sum() = 1000000.1  (TS_CELL = 1000000)
[2] HEAD 版 vs 新版(force_mean 默认 True) 逐位相同 = True
[3] 列均值相对偏差 std（5198 个高表达基因）:
    force_mean=True 0.00020   force_mean=False 0.03487   倍数 173.2x
```

**[2] 是本条最重要的一项**，它把「改了共享文件」这个风险彻底消掉：断言对象是
`git show HEAD:src/vcclab/decoder.py` 取出的 blob，不是散文描述，比较的是 400×18,080 的实际输出。
所以并发跑的其它三个 slice 一行都没被扰动。**纪律：改共享文件时，对 git blob 断言，不要对注释断言。**

**[1]** 证明 honest 构造的分母是对的：`ref.m_full` 精确等于 `_cpm_csr.mean(0)`（差 0），
且在 1e-7 内合到 1e6，与 `tgt` 同标度。

**[3]** 证明修复真的起作用。`force_mean=True` 那 0.0002 是**量化残差**，不是抽样波动 ——
均值被钉死之后剩下的只有 `hamilton` 取整的余量。0.03487 才是真实的均值抽样散布。

---

## 3 · 改动本身

`src/vcclab/decoder.py`，`design_cells` 新增关键字 `force_mean: bool = True`：

```python
if force_mean:
    V *= tgt / np.maximum(V.mean(0), 1e-12)      # 钉死经验均值 -> 打分器告警
else:
    V *= tgt / np.maximum(ref.m_full, 1e-12)     # 只平移总体均值，保留波动
```

关键点：**天真地删掉那一行是错的** —— 那会让细胞均值停在**对照** profile 而不是目标 profile，
预测直接失效。`False` 分支仍把均值移到 `tgt`，但用的是对照**总体**均值 `ref.m_full`
而不是抽到的那 400 个细胞自己的均值，于是样本相对总体的偏差被保留下来 ——
那正是打分器在找的波动。后续的 per-gene $\bar\psi$ 二分与 `V *= TS_CELL / V.sum(1)` 未动。

被告警的量（`delta.py`）：`PRED_CORRECTION_BUDGET_FLAG_RATIO = 0.7`，
V6/V7/V8 实测 spread 0.003167 对声称校正 0.006544 = **比值 0.484 < 0.7**，
打分器原话：*per-cell scatter 在 pseudobulk 里基本抵消，这不像一个 i.i.d. 抽样的预测细胞群*。

### 代价的上界（`inferred from measured`，但结论是硬的）

`expr_mse_unbiased_capped_norm` 的 policy 是 `clamp_low = 0.0`，冻结无技巧点是
**1.0 =「原样贴对照」**。八个变体的 raw 全在 1.03–1.58，**全部在无技巧点的坏侧**，
所以官方刻度上一律读 **0.0000**。而让均值波动只会**增加**均值误差，把 raw 推得更高，
不可能掉到 $b$ = 0.986–0.992 以下。

> **所以 honest arm 在 `expr_mse` 上的官方刻度代价可证为 0。**
> 真正可能被损害的只有 `pds_cosine` —— 而它是我们唯一实质得分的成员。

这是③级未测项里唯一值得再花一次机器的：E33 已证明 pds 的
**闭式→build 偏移恰好为 +0.0000**，所以 V9 的 pds 可以不跑 scorer、用闭式路径定价。
**这是本 slice 的下一步**，不需要 `compute_metrics`。

---

## 4 · 为什么没测到：资源，不是代码

| 尝试 | 结果 |
|---|---|
| V8 重建 + 打分（作为 V9 的对照臂） | build 成功 358 s；`compute_metrics` **27 分钟累计 13 秒 CPU**，被杀 |

诊断（本机实测）：进程状态是 `R` 不是 `D`，所以不是锁死，是 **page-fault bound**。
系统 pagein 累计 20,103,026 次 × 16 KB ≈ **320 GB 换页**，而数据卷当时 460 GB 里只剩 **8 GB**。
anndata 读 1.11 GB 的 `real.h5ad` 与内核写 swap 抢同一个快满的设备。
`out/chain.log` 最后一次增长停在 scanpy `rank_genes_groups` 里，静默 15 分钟。

**这不是 i.i.d. 修复的问题，也不是本 slice 设计的问题。**
同一台机器上同一条代码路径在磁盘宽松时跑完过（V5–V8 各约 1,289 s）。
需要的是 ~15 GB 以上的卷余量。

---

## 5 · 结论

1. **修复已就位且安全**（`measured`）：默认路径逐位不变，`False` 分支恢复 173.2× 的均值波动。
2. **合法性的代价在官方刻度上可证为 0**（`inferred`），因为 `expr_mse` 已在地板上。
   所以「要不要交一个诚实的提交」不是取舍题 —— 除非 `pds` 被损害。
3. **下一步不需要 scorer**：用 E33 验证过的闭式 pds 路径给 V9 的 `pds_cosine` 定价
   （闭式→build 偏移实测 +0.0000）。若 pds 不掉，honest arm 应当直接采用。
4. ⚠️ 不要把「`expr_mse` 现在是 0 所以它不重要」当成永久结论。
   该成员的官方 $r$ = 0.028–0.045，跨度极大；一旦某个变体把 raw 压到 1.0 以下它就活了。
   届时 `force_mean` 的取舍要重测。
