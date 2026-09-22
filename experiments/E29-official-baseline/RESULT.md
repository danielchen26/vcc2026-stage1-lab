# E29 · 官方基线与官方刻度：我们的 0.2025 是在什么分母上量出来的

一句话结论（**measured**）：**0.2025 不能和排行榜的 0.1899 并排放**。把同样的原始值放到
官方的两端刻度上，V8 是 **0.1215（乐观端）／0.0848（悲观端）**，离第一名 0.1899 还差
0.068–0.105；而 A→V8 的三步改进链**方向上活着**（每一步都没变坏），只是幅度和绝对水平
全部缩水。0.2025 里最大的一项 `expr_mse_unbiased_capped_norm`（0.5836，占总分 48%）
在官方刻度上是 **0.0000 —— 被 policy 的 clamp_low 地板夹住的零分**，因为我们的原始值
1.0819 已经**差于「原样贴对照」**这个无技巧点。

**第二个结论，同样 measured，而且它推翻了「anchor 才是全部问题」这个假设（§4.2）**：
把两个改动分开施加后，**只换 anchor 让 V8 从 0.2025 升到 0.2178／0.2066（帮我们）**，
**只换 baseline 让它掉到 0.1013／0.0844（全部损失都在这里）**。两者符号相反。所以真正
要修的是**比较器**，不是刻度形状 —— 也就是本切片最初被分派的那个靶子。

## 三个尺度错配，现在的状态

| 错配 | 状态 |
|---|---|
| 基线定义（我们发明的「对照均值 tile 400 遍」vs 官方 generic-response） | **已用官方 b 列定量归因**（§5.2）；本地重建被内核 OOM 回收两次，见 §5.3 |
| 刻度形状（anchor 固定 0/1 vs 实测 replicate anchor r） | **已关闭并独立验证**（§2 代码路径、§4.1 用榜面自证、§4.2 分解） |
| 8 扰动 / 1 context vs 300 扰动 / 3 context | **仍然存在**，§4 的两张表都带着它 |

§4 的表是唯一能和 0.1899 并排的表，**它仍然不是同类比较**：分子（我们的 raw）来自 8 个
扰动、1 个 context；分母（官方 b、r）来自 300 扰动 × 3 context 的官方 val bundle。
§5.2 的归因表**不带**这个错配的第三条以外的成分：它比的是两个基线的原始值本身。

---

## 1 · 复现命令（全部 measured）

解释器一律 `/Users/chetianc/vcc2026/.venv/bin/python`。

```bash
cd experiments/E29-official-baseline
# 全家重算（快，~17 s）：官方两端刻度 / 冻结刻度 / from_baseline 参照 / mse 专项审计
python anchor_local.py rescore
# 二因素分解（快，~13 s）：只换 anchor vs 只换 baseline，见 §4.2
#   —— 代码内联在 out/decomposition.json 的生成命令里，逻辑同 anchor_local._two_ended
# 官方 generic-response 基线（内存里建预测并评分，零 .h5ad 落盘，~25 min）
#   本机 swap 紧张，用自带闸门脚本等到安全窗口再启动：
./gate_and_build.sh                    # 等 swap>2G & disk>8G & 重进程<2，然后自动跑
python -u run_official_baseline.py baseline   # 机器空闲时可直接跑
# 本地 split-half replicate anchor（~1.5-2 h，可断点续跑；当前状态见 §8）
python anchor_local.py splits 5
python anchor_local.py assemble
```

评分配置逐字照 `experiments/E28-pds/score_v8.py`：

```python
cfg = EvalConfig.from_preset("vcc2026")
cfg = replace(cfg, pert_col="target_gene", device="cpu")
cfg = replace(cfg, de=replace(cfg.de, backend="scanpy"))
```

**F29 单旋钮生成器纪律在本实验不适用**：E29 不产生任何新的 variant build，V8 的 `.h5ad`
一行都没重跑。唯一变化的是 `score_metrics` 的**分母**和**刻度形状**，两侧的分子全部逐字
复用已落盘的 `agg_*.parquet`。可归因性由「分子字节不变」保证，比单旋钮更强。

磁盘占用 **0 字节的 .h5ad**：`build_generic_baseline(save_pred=None)` 在内存里建预测并直接
评分（`baseline.py:1464-1465` 只在 `save_pred` 非 None 时写盘）。产物只有几 KB 的
parquet/json，无需 AUTO_CLEAN。

---

## 2 · 官方刻度是什么：从代码确认，不是从算术反推

**结论：`(u - b) / (r - b)` 是 `cell_eval2` 的原生能力，我们只是从来没传那两个参数。**

`score_metrics` 的签名（`score.py:511-531`）有 `anchor=` / `anchor_cache=` / `anchor_expect=`
/ `real_bundle=` 四个入口，本仓库每一处调用都只传 `results_user, results_base,
comparison_statistic`，所以我们一直读的是 `from_baseline` 列。

两端刻度的实现路径，逐行（**measured**，全部读自安装树 `cell_eval2 0.16.0`）：

1. `score.py:874` —— 只有给了 `anchor=` 或 `anchor_cache=` 才会加 `from_replicate` 列。
2. `score.py:898` → `_replicate_entries(base_by_name, frame)`（`score.py:377`）。docstring
   第一行：*"The replicate scale as (base, Scoring) pairs: 0 = baseline, 1 = replicate."*
3. `score.py:431` —— **关键一行**：
   `policy = replace(spec.scoring, anchor=float(rep), allow_negative_baseline=False)`。
   也就是把 catalog policy 的 anchor **换成实测的 r**。
4. `score.py:270` —— `score_one(by_canon[canon], entry.base, entry.scoring)`。
   `scoring.score_one`（`scoring.py:461`）的未夹持核心就是 `(u - b) / (anchor - b)`。

所以 Main 的算术在**结构上被证实**：不是巧合的数值吻合，而是同一个表达式。
`_replicate_entries` 的 docstring（`score.py:381-382`）自己把这句话写出来了：
*"`score_one` with `policy.anchor` set to the measured replicate has `(u - b) / (r - b)` as
its unclamped linear core for both directions."*

`anchor_frame` 需要什么：`metric` + `replicate` 两列（schema 见 `anchor.py:_ANCHOR_SCHEMA`，
共 8 列，`_replicate_entries` 只读前两列）。三个供给口：

| 入口 | 来源 | 我们能否本地驱动 |
|---|---|---|
| `real_bundle=` | 官方冻结 bundle（`read_real_bundle`） | **否** —— 我们没有 bundle |
| `anchor_cache=` | `_cached_bundle`（`score.py:294`），docstring：*"NEVER computes"* | 否（缓存里没东西） |
| `anchor=` | 一个写出来的 anchor artifact，过 `validate_anchor` + `AnchorExpect` 门 | 是，但需要 run_meta |
| **`anchor.compute_replicate_anchor`（`anchor.py:178`）** | **公开生产者，只吃 real 侧** | **是 —— 不需要任何 bundle** |

`compute_replicate_anchor` 就是关键：它是 public API，签名
`(real, *, config=None, base_seed=0, n_splits=5, **overrides) -> (splits, anchor)`，
只读真值、把真值劈成不相交的两半（`ceiling._disjoint_halves`）、用 half_a 给 half_b 打分，
内部强制 `control_source="pred"`（`anchor.py:_inner_config`，模块 docstring 解释为什么：
共享对照会让被测的两个量共享采样噪声，实测 lfc_nmae 偏乐观 0.5–2.3%）。

**`score_metrics(agg_pred, agg_base)` 缺的守卫（额外发现）**：官方 CLI 在打分前会跑
`cli._check_baseline_config`（`cli.py:120`），比对 baseline 侧 `baseline_meta.json` 与提交侧
`run_meta.json` 的 `cell_eval2_version` / `config_digest` / `source_fingerprint` /
`resolved_device` / `resolved_de_backend` / `input_type_real_effective` /
`de_real_fingerprint` / `comparator`，任一不合就 `SystemExit("baseline/user mismatch -- the
margins would be meaningless")`。**这个守卫只在 CLI 上，库函数 `score_metrics` 完全不做这件
事。** 我们的调用因此没有任何配对校验；本实验里两侧是同一进程同一配置同一 real.h5ad，所以
实际安全，但这是一个真实的缺口，值得记进 findings。

---

## 3 · 冻结刻度 `low-random_high-1_v10`（免费诊断，不需要 bundle）

`cell_eval2.scales.SCALES` 只有**一个**成员（**measured**，`list(SCALES) ==
['low-random_high-1_v10']`）。它**不是** replicate anchor 刻度 —— 1 端是「真实计数矩阵
本身」而不是实测复本；但它的 0 端是每个指标的**无技巧点常数**，覆盖 vcc2026 的全部六个
计分成员，且不需要任何官方产物（`scales.py:310-452`）：

| 指标 | 冻结 base（=0 端，无技巧点） | anchor | clamp_low |
|---|---|---|---|
| `expr_mse_unbiased_capped_norm` | **1.0 = 原样贴对照** | 0.0 | −6.0 |
| `de_wilcoxon_lfc_nmae` | 1.0 = `lfc_hat = 0`（预测零变化） | 0.0 | −1.0 |
| `pds_cosine` | 0.5 = 均匀秩 | 1.0 | −1.0 |
| `de_wilcoxon_direction_fidelity_yield_raw` | 0.5 = 掷硬币 | 1.0 | −1.0 |
| `de_wilcoxon_direction_reach_raw` | 0.0 = 指标最小值 | 1.0 | 0.0 |
| `de_wilcoxon_sig_jaccard` | 0.0 = 理论下界 | 1.0 | 0.0 |

`scale=` 只会**加一列**，绝不动 `from_baseline`（`score.py:985-988`）。全家重算
（**measured**）：

| 指标 | A | V2 | V3 | V4 | V5 | V6 | V7 | V8 |
|---|---|---|---|---|---|---|---|---|
| `de_wilcoxon_sig_jaccard` | 0.0485 | 0.0476 | 0.0231 | 0.0484 | 0.0483 | 0.0484 | 0.0484 | 0.0482 |
| `de_wilcoxon_lfc_nmae` | −0.0168 | −0.0274 | −0.0206 | −0.0170 | −0.0141 | 0.0118 | 0.0115 | −0.0004 |
| `de_wilcoxon_direction_fidelity_yield_raw` | −0.0092 | 0.0035 | −0.3294 | −0.0095 | −0.0205 | −0.0043 | −0.0075 | −0.0163 |
| `de_wilcoxon_direction_reach_raw` | 0.1348 | 0.0080 | 0.0101 | 0.1343 | 0.1487 | 0.0204 | 0.0219 | 0.1482 |
| `pds_cosine` | 0.0357 | 0.5714 | 0.0000 | 0.0357 | 0.4286 | 0.4286 | 0.5000 | 0.5000 |
| `expr_mse_unbiased_capped_norm` | **−0.2422** | **−2.4226** | −0.8353 | −0.2422 | **−0.5772** | −0.0347 | −0.0425 | **−0.0819** |
| **avg_score** | **−0.0082** | −0.3032 | −0.1920 | −0.0084 | 0.0023 | 0.0784 | 0.0886 | **0.0996** |

读法：**在「0 = 无技巧」的绝对刻度上，variant A 是负的（−0.0082），V8 是 +0.0996，而
+0.0996 里 0.5000/6 = 0.0833 全部来自 `pds_cosine` 一项。** 改进链依然单调
（V5 0.0023 → V6 0.0784 → V7 0.0886 → V8 0.0996）。

---

## 4 · 表 2（跨尺度）：官方 b 和官方 r，只有这张表能和 0.1899 并排

b、r 取自 `docs/01-scoring.md:167-174`（实时榜反算，三个官方 context 的区间）。
**⚠️ 这张表不是同类比较**：分子是我们 8 扰动 / 1 context 的 raw，分母是官方 300 扰动 /
3 context 的 b 与 r。区间两端各算一次，让不确定度可见。打分不是我手算的 —— 走
`scoring.score_one` + catalog policy，anchor 换成实测 r（复刻 `score.py:431` 的同一次
`replace`）。

**乐观端 b_lo / r_lo**（**measured**）

| 指标 | A | V2 | V3 | V4 | V5 | V6 | V7 | V8 | 第一名 |
|---|---|---|---|---|---|---|---|---|---|
| `pds_cosine` | 0.0418 | 0.6691 | 0.0000 | 0.0418 | 0.5018 | 0.5018 | 0.5855 | **0.5855** | 0.708 |
| `expr_mse_unbiased_capped_norm` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** | 0.041 |
| `de_wilcoxon_sig_jaccard` | 0.0776 | 0.0750 | 0.0060 | 0.0773 | 0.0770 | 0.0775 | 0.0774 | **0.0769** | −0.004 |
| `de_wilcoxon_lfc_nmae` | −0.0252 | −0.0419 | −0.0311 | −0.0255 | −0.0209 | 0.0201 | 0.0196 | **0.0009** | 0.178 |
| `de_wilcoxon_direction_fidelity_yield_raw` | −0.0331 | −0.0111 | −0.5851 | −0.0336 | −0.0526 | −0.0247 | −0.0302 | **−0.0453** | 0.003 |
| `de_wilcoxon_direction_reach_raw` | 0.0964 | −0.0429 | −0.0405 | 0.0958 | 0.1116 | −0.0291 | −0.0276 | **0.1110** | 0.213 |
| **avg_score** | **0.0263** | 0.1080 | −0.1085 | 0.0260 | 0.1028 | 0.0909 | 0.1041 | **0.1215** | **0.1899** |

**悲观端 b_hi / r_hi**（**measured**）

| 指标 | A | V2 | V3 | V4 | V5 | V6 | V7 | V8 |
|---|---|---|---|---|---|---|---|---|
| `pds_cosine` | 0.0369 | 0.5903 | 0.0000 | 0.0369 | 0.4427 | 0.4427 | 0.5165 | **0.5165** |
| `expr_mse_unbiased_capped_norm` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** |
| `de_wilcoxon_sig_jaccard` | 0.0297 | 0.0273 | −0.0359 | 0.0294 | 0.0292 | 0.0296 | 0.0295 | **0.0291** |
| `de_wilcoxon_lfc_nmae` | −0.0265 | −0.0450 | −0.0331 | −0.0268 | −0.0217 | 0.0237 | 0.0231 | **0.0023** |
| `de_wilcoxon_direction_fidelity_yield_raw` | −0.0858 | −0.0653 | −0.6022 | −0.0862 | −0.1040 | −0.0780 | −0.0830 | **−0.0972** |
| `de_wilcoxon_direction_reach_raw` | 0.0429 | −0.1011 | −0.0986 | 0.0423 | 0.0586 | −0.0869 | −0.0853 | **0.0581** |
| **avg_score** | **−0.0005** | 0.0677 | −0.1283 | −0.0007 | 0.0675 | 0.0552 | 0.0668 | **0.0848** |

### 三步链活下来了吗

| 刻度 | A | V5 | V6 | V7 | V8 | 相对 A |
|---|---|---|---|---|---|---|
| `from_baseline`（仓库一直在读的，退化基线 + anchor 0/1） | 0.1110 | 0.1559 | 0.1762 | 0.1873 | **0.2025** | **+82%** |
| 冻结刻度（0 = 无技巧） | −0.0082 | 0.0023 | 0.0784 | 0.0886 | **0.0996** | 从负到正 |
| 官方两端 b_lo/r_lo | 0.0263 | 0.1028 | 0.0909 | 0.1041 | **0.1215** | +362% |
| 官方两端 b_hi/r_hi | −0.0005 | 0.0675 | 0.0552 | 0.0668 | **0.0848** | 从负到正 |

**判决：链条方向活着，但「+82% 超过 0.1899」这个说法死了。** 三个刻度都同意 V8 是八个
variant 里最好的，也都同意 V8 > V7 > A。但：

- **绝对水平**：V8 在官方刻度上是 0.0848–0.1215，**不是** 0.1899，差 0.068–0.105。
- **排序有一处真实分歧，而且必须归因到「比较器」而不是「两端刻度」**：`from_baseline` 说
  V5(0.1559) < V6(0.1762) < V7(0.1873)，单调；**官方刻度说 V5(0.1028) > V6(0.0909)**。
  §4.2 的分解把责任钉死了（**measured**）：
  - **只换 baseline**：V5 **0.0865** > V6 **0.0712** —— 分歧**已经出现**。
  - **只换 anchor**：V5 0.1671 < V6 0.1912 —— 依然单调，**分歧不出现**。

  所以「V6 的 λ=0.5 是净损失」是**比较器换成官方基线**造成的，不是两端刻度造成的。
  机制：V6 的收益全在 `expr_mse`（from_baseline 0.3930 → 0.6018），而那一项的收益完全
  来自退化基线自己烂在 2.5985；换成官方 b≈0.989 后我们的 1.0347 落在无技巧点的错误一侧，
  再被 `clamp_low=0.0` 夹成 0.0000，整项蒸发；而它在 `reach` 上付的代价
  （0.1424 → 0.0132）是真的。**F30 式的错误重演了一次：被优化的那一项在真实分母下
  没有梯度。** 这点尤其重要，因为**比较器是我们能修的**，刻度形状不是。
- V7 → V8（λ 0.5 → 0.7）在全部四个刻度上都是正的，这一步是真的。

### 4.1 这把尺子对不对：拿第一名自己的 raw 反推他自己的 scaled（独立校验，measured）

把 `docs/01-scoring.md:169-174` 里**第一名自己的六个 raw** 喂进同一个 helper，看能不能
复现他自己**公布的 scaled**。这是对「刻度形状 + 抓取表」的联合检验，不用我们自己的任何数据。

| 指标 | 公布 scaled | b_lo/r_lo | b_hi/r_hi | 公布值落在区间内？ |
|---|---|---|---|---|
| `pds_cosine` | 0.708 | 0.7494 | 0.6612 | ✅ |
| `expr_mse_unbiased_capped_norm` | 0.041 | 0.0282 | 0.0348 | ❌ **两端都低** |
| `de_wilcoxon_sig_jaccard` | −0.004 | 0.0226 | −0.0207 | ✅ |
| `de_wilcoxon_lfc_nmae` | 0.178 | 0.1723 | 0.1922 | ✅ |
| `de_wilcoxon_direction_fidelity_yield_raw` | 0.003 | 0.0310 | −0.0258 | ✅ |
| `de_wilcoxon_direction_reach_raw` | 0.213 | 0.2415 | 0.1930 | ✅ |
| **avg_score** | **0.18983** | **0.20751** | **0.17244** | ✅ 公布总分落在区间内 |

**6 项里 5 项、以及总分 0.1899 都落在区间内 —— 刻度形状 `(u−b)/(r−b)` 与 catalog policy
（含每项的 clamp）被独立验证。** 这不是用我们的数据自证：分子分母全来自排行榜。

唯一的失配是 `expr_mse_unbiased_capped_norm`：公布 0.041，四个角点算出来 0.0282–0.0348，
**两端都偏低约 0.006–0.013**。解方程反推：取 r = 0.036（区间中点）时
`(0.959 − b)/(0.036 − b) = 0.041` 给出 **b ≈ 0.9985**，比抓到的 0.986–0.992 高约 0.007。
所以**该项的 b 端点需要重抓**（或者公布的 raw 0.959 本身是四舍五入的）。
对总分的影响 ≤ 0.002，不改变本文件任何结论；标注为 `inferred`，要 measure 它需要官方
frozen real bundle 的 `baseline_agg`，我们没有。

交叉印证：SourceUnion 用自己的手算公式 `(b−u)/(b−r)` 从 raw 0.892 独立得到 nmae
0.173–0.191，与本 helper 的 0.1723/0.1922 吻合（**measured**，两条独立实现）。

### 4.2 究竟是 anchor 还是 baseline 杀死了 0.2025？（二因素分解，measured）

这是本切片最重要的一个数字，而且它**推翻了「anchor 是全部问题」这个假设**。把两个改动
分开施加：`avg_score`，八个 variant（`out/decomposition.json`）。

| 组合 | A | V2 | V3 | V4 | V5 | V6 | V7 | V8 |
|---|---|---|---|---|---|---|---|---|
| b = 我们的退化基线，r = catalog 0/1（**仓库一直在报的**） | 0.1110 | 0.0922 | −0.0119 | 0.1109 | 0.1559 | 0.1762 | 0.1873 | **0.2025** |
| b 不变，**只换 anchor** → 官方 r 乐观端 | 0.1122 | 0.1069 | −0.0583 | 0.1120 | 0.1671 | 0.1912 | 0.2039 | **0.2178** |
| b 不变，**只换 anchor** → 官方 r 悲观端 | 0.1110 | 0.0925 | −0.0471 | 0.1107 | 0.1570 | 0.1821 | 0.1933 | **0.2066** |
| **只换 baseline** → 官方 b 乐观端，r = catalog 0/1 | 0.0201 | 0.0874 | −0.0665 | 0.0199 | 0.0865 | 0.0712 | 0.0827 | **0.1013** |
| **只换 baseline** → 官方 b 悲观端，r = catalog 0/1 | 0.0031 | 0.0693 | −0.0867 | 0.0029 | 0.0696 | 0.0531 | 0.0647 | **0.0844** |
| 两个都换（= §4 的表 2） | 0.0263 | 0.1080 | −0.1085 | 0.0260 | 0.1028 | 0.0909 | 0.1041 | **0.1215** |
| 两个都换，悲观端 | −0.0005 | 0.0677 | −0.1283 | −0.0007 | 0.0675 | 0.0552 | 0.0668 | **0.0848** |

**V8 的分解：**

| 改动 | 0.2025 变成 | 贡献 |
|---|---|---|
| 只换 anchor（r: 1 或 0 → 实测 r） | 0.2178 / 0.2066 | **+0.004 … +0.015（帮我们）** |
| 只换 baseline（b: 退化 → 官方） | 0.1013 / 0.0844 | **−0.101 … −0.118（杀手）** |
| 两个都换 | 0.1215 / 0.0848 | −0.081 … −0.118 |

**结论：分母 b 是全部的损失来源，anchor r 反而略微抬高我们的分数。**
机制：`score_one` 的核心是 `(u − b)/(r − b)`，实测 r 比 catalog 的 0/1 更靠近 b，**跨度
`(r − b)` 变小**，于是任何已有的余量 `(u − b)` 被**放大**；我们六项加总的净余量是正的，
所以缩跨度让我们赚。而把 b 换成强得多的官方基线，直接缩小的是 `(u − b)` 本身，很多项
还会翻成负的。

**所以「anchor 才是全部问题」这个假设被测量推翻了，而本切片最初被分派的靶子 —— 基线定义
—— 才是真正的那一个。** 两个改动的符号相反，这也是为什么只看「两个都换」的 0.1215 会
同时低估 baseline 的破坏力（−0.101…−0.118）并掩盖 anchor 的方向。

---

## 5 · 官方 generic-response 基线（本地构建，作为校验）

### 5.1 调用路径与前置条件（全部 measured，读自 `baseline.py` 1535 行全文）

`build_generic_baseline(real, *, config, exclude_target_gene=True, emit="dispersed",
seed=0, de_real=None, save_pred=None, allow_degenerate=False) -> BaselineResult`
（`baseline.py:1354`）。

`BaselineResult`（`baseline.py:671-679`）四个字段：
`results`（tidy 逐扰动帧）、`agg`（`aggregate_metrics_wide` 的宽帧，`score_metrics` 直接吃）、
`profile`（`GenericProfile`）、`meta`（provenance 戳）。

`GenericProfile`（`baseline.py:86-101`）：`values, genes, n_perturbations,
exclude_target_gene, n_excluded`。`n_excluded` 被显式报出来而不是只 log，因为当扰动标签是
guide ID 而不是基因符号时自剔除是**静默 no-op**。

`allow_fractional_counts=True` **不需要调用方设**：`baseline_config`
（`baseline.py:1202-1242`）自己加，并把 `cache_pred` 清成 None（两个 emit 臂、两个
`exclude_target_gene` 臂共享非严格 fingerprint，暖的 pred cache 会串味）。
`cache_real` 保留。`de.backend="deseq2"` 被 `_reject_unsupported` 直接拒（profile 是均值，
分数计数喂负二项 GLM 无意义）。

本机实测的前置条件检查（**measured**，`out/` 下 smoke 记录）：

| 前置条件 | 强制位置 | 我们的值 | 通过 |
|---|---|---|---|
| `pert_col` 在 obs | `baseline.py:150` | `target_gene` | ✅ |
| `control` 在 obs | `baseline.py:159` | `non-targeting`（18,400 cells） | ✅ |
| 非对照扰动 ≥ 2 | `baseline.py:184` | 8 | ✅ |
| var index 唯一 | `baseline.py:192` | 18,080 / 18,080 唯一 | ✅ |
| 至少一个 target 能解析 | `distances.resolve_exclusion_columns` 零解析即 raise | `n_excluded = 8`（8/8 全中） | ✅ |
| `emit="dispersed"` 要求 real 解析为 counts | `baseline.py:1398-1404` | v2 + `autodetect=False` + 声明 counts → `counts` | ✅ |
| 矩阵空间锁 | `_lock_from_adata` | 两侧都 `counts`，`allow_discrete` 保持 False | ✅ |
| `max_counts_per_cell` | `run.py` 的 v2 scale gate | 上限 1e6，实际最大行和 **20,431** | ✅ |

emission 诊断（**measured**）：`r_max = 23.0`、`r_median_nonzero = 1.00250`、
`n_genes_control_zero = 970`、`profile_mass_unreachable = 1.44e-06`、
`max_scaled_noncontrol_row_total = 20,418.93`、`n_rows = 3200`。
profile 总和 20,005.5（我们把细胞 thin 到 20,000 UMI，吻合）。

### 5.2b 代码路径的端到端 dry-run 与**逐字**的 scorer 警告（measured）

在把 25 分钟押上去之前，我用一个**合成 panel**（4 扰动 × 30 细胞 + 120 对照，300 基因，
Poisson(3)）跑通了 `stage_baseline` 的**每一行**。结果：代码路径全程正常，
在 `build_generic_baseline` 内部 `aggregate_metrics_wide` 处按**合成数据自身的性质**报错
而停 —— 这不是脚本 bug，见下。收获是两条**逐字**的 scorer 警告，以及一条逐字的拒绝理由。

警告 1（`DeprecationWarning` 之外的实质警告，**逐字**，出现两次 —— real 侧与 pred 侧各一次）：

> `allow_fractional_counts=True was LOAD-BEARING here: the pred side has fractional values under declared input_type='counts', so it passed the input-type check ONLY because of the flag -- without it validation would have refused the run ("declared input_type='counts' but values are fractional"). If this is a BASELINE arm that is expected: a mean profile is fractional in any space. If this is a SUBMISSION and processing continues (later gates, including the scale limit, may still refuse it), these values will be INTERPRETED as counts. Note that config_digest cannot see this flag (baseline.DIGEST_EXEMPT_FIELDS), so `score --baseline-agg` pairing will NOT report a baseline/submission mismatch on it; the --real-bundle path IS protected (anchor._SEMANTIC_FIELDS compares it, via score.expect_from_run_meta).`

这条警告顺手确认了 §2 末尾那个缺口的一个**具体后果**：`allow_fractional_counts` 是
digest-exempt 的，所以 `score --baseline-agg` 的配对检查**看不到它** —— 只有 `--real-bundle`
路径受保护。我们的库函数调用连前者都没有。

警告 2（**逐字**）：

> `de_lfc_nmae: omitted 4 perturbation(s) for an empty gate (e.g. AAA). The gate is real-side only, so this set is the same for every submission.`

拒绝理由（**逐字**，合成数据触发，真实 panel 不会触发）：

> `ValueError: expr_mse_unbiased_capped_norm: the sum of expr_distance_unbiased over 4 perturbation(s) is -0.0030600605563094723, which is not positive, so the ratio of sums is undefined or sign-flipped. That means this reference panel carries no measurable aggregate effect at this cell depth -- a property of the REFERENCE, not of the submission. Individual negative values are normal and expected; only the SUM going non-positive is a failure. Fix or replace the reference panel.`

**为什么这条在真实 panel 上不会触发**：它要求 `expr_distance_unbiased` 的**和**非正。
我的合成数据是纯 Poisson 噪声、扰动与对照同分布，所以真实效应为零、去偏后和为负；
而我们真实 panel 上该指标实测 **0.001208 > 0**（三份 agg parquet 都是这个值，
`agg_base` / `agg_pred` / `agg_v8` 完全一致）。所以 dry-run 验证了代码路径，
而它停下的那个位置正是一个**只有假数据才会撞上的守卫**。


### 5.2 我们发明的退化基线 vs 官方 b 列：分母膨胀究竟在哪几项（measured）

官方 b 列取自 `docs/01-scoring.md:167-174`；我们的退化基线原始值取自
`experiments/E27-six-metrics/out/agg_base.parquet`。这张表不需要本地重建基线就能给出结论。

| 指标 | 方向 | 我们的退化 b | 官方 b | 强弱 | 占 V8 0.2025 之和的份额 |
|---|---|---|---|---|---|
| `pds_cosine` | higher | **0.50000** | **0.500** | **完全相同** | **41.2%** |
| `expr_mse_unbiased_capped_norm` | lower | **2.59845** | 0.986–0.992 | **我们弱 2.6×** | **48.0%** |
| `de_wilcoxon_direction_reach_raw` | higher | **0.00732** | 0.047–0.097 | **我们弱 6–13×** | **11.7%** |
| `de_wilcoxon_direction_fidelity_yield_raw` | higher | 0.49624 | 0.505–0.522 | 我们略弱 | −0.7% |
| `de_wilcoxon_sig_jaccard` | higher | 0.04752 | 0.021–0.037 | 我们**更强** | 0.1% |
| `de_wilcoxon_lfc_nmae` | lower | 0.99774 | 1.0009–1.0017 | 我们**更强** | −0.2% |

**这是整个 0.2025 的机制，一行说完：分母膨胀精确地集中在我们的基线最弱的两项上 ——
`expr_mse`（48.0%）和 `reach`（11.7%），合计 59.7%。** 我们的对照粘贴基线在这两项上
分别比官方 generic-response 基线弱 2.6 倍和 6–13 倍，于是「超过基线」这件事变得廉价。

而唯一一项我们的基线**可证明与官方完全相同**的指标（`pds_cosine`，两边都恰好 0.500），
正好也是 V8 在官方刻度上**唯一真正得分**的指标（0.5855/6 = 0.0976，占 0.1215 的 80%）。
这不是巧合：官方 `competition.py:289` 明确写着，在 `exclusion_scope="panel"`（vcc2026
preset 的取值）下，一个贴对照的臂测出来就是 **0.5000 —— "the control-paste floor"**。
所以我们的 pds 分母一开始就是对的，而其余五项不是。

**推论：V8 的 0.2025 里，真正在官方尺子上站得住的只有 pds 那一项；F25 的「78% 不是
生物学」低估了 —— 按官方刻度算是 80% 来自单一指标，另外五项合计 0.0239。**

### 5.3 本地重建官方基线（`build_generic_baseline`）—— 未完成，原因与代价

本地构建**两次被内核 OOM 回收**，不是代码失败：进程消失、日志 0 字节、无 traceback
（SIGKILL 不 flush）。机制已定量（**measured**）：`sysctl vm.swapusage` 报
`total = 34,816M  used = 33,662.94M  free = 1,153.06M`，而 swap 就住在只剩 3.5 GiB 的
同一个卷上，内核无法扩 swap，于是回收 RSS 最大的进程。同一时刻项目里六个 python 进程的
RSS 全部只有 22–171 MB —— 它们不是小，是**被整体换出**。

`build_generic_baseline` 的峰值内存无法压缩：`_materialize_reference` 必须把 real 全量读进
内存（`baseline.py:62-83` 说明了为什么不能 backed —— `prep._grouped_means` 用
`issparse(X)` 一次性决定稀疏分支，backed CSR 会让 `pseudobulk` 抛
`ValueError: setting an array element with a sequence`），real 本身 138,532,377 个非零
× (4B data + 4B indices) ≈ **1.11 GB**，预测是同尺寸的第二份，再加 `cache_strict=True`
的严格内容哈希与 `compute_metrics` 的工作集，下限约 3 GB。

**未完成的只有「六个原始值的本地实测」这一项，而它不是本文件任何结论的前提**：§5.2 用
官方 b 列已经给出了分母膨胀的定量归因，§4.1 已经独立验证了刻度，§6 已经用算术恒等式
证明了 0.5836 是分母伪影。本地重建的价值是把官方 b 列从 `inferred`（榜面反算）升级为
`measured`（我们自己的 panel 上实测），以及给出 V8/A 在官方基线上的 `from_baseline`。

复现它需要的全部条件已经验证通过（§5.1 的八项前置条件全绿，profile 与 emission 诊断都已
实测），命令是 `python -u run_official_baseline.py baseline`，在 swap 空闲 > 2 GB 且磁盘
> 8 GB 时单独跑，约 25 分钟，**零 .h5ad 落盘**。

---

## 6 · 共享 helper（本仓库其余 agent 的官方刻度入口）

Main 已要求所有报官方刻度数字的 agent 统一走这里，避免五份结果各自漂移。入口在
`experiments/E29-official-baseline/anchor_local.py`，只读、无副作用、不需要 bundle：

```python
import sys; sys.path.insert(0, "<repo>/experiments/E29-official-baseline")
import anchor_local as A

A.official_two_ended(agg_parquet)            # 吃 aggregate_metrics 落盘的 parquet
A.official_two_ended_from_raw({m: raw, ...}) # 吃六个原始均值
# -> {"b_lo/r_lo": {metric: {"score", "degenerate"}, "avg_score": {"score", "n"}},
#     "b_hi/r_hi": {...}}
A.SCORED        # 六个官方指标名
A.OFFICIAL_BR   # metric -> (b_lo, b_hi, r_lo, r_hi)
```

它**不是手算**：`dataclasses.replace(spec.scoring, anchor=float(r))` 复刻 `score.py:431`，
然后 `scoring.score_one(u, b, policy)` 复刻 `score.py:270`，所以 catalog 的 clamp 与 penalty
形状全部生效。**两端都要报，不要取中点。** 六个 `SCORED` 键缺一个就 `KeyError` —— 故意的，
静默少算一项会改变 `avg_score` 的分母。

---

## 7 · `expr_mse_unbiased_capped_norm` 专项审计（Main 的两个点问）

catalog policy（**measured**）：`direction=lower, anchor=0.0, penalty=boxcox,
clamp_low=0.0, clamp_high=1.0, decisive=True`。冻结刻度的 base = **1.0 = 「原样把对照贴
出来」**（`scales.py:366`，#257 起这是任何 panel 上的**性质**而非声明）。

| 变体 | raw | 过了无技巧点 1.0？ | `1 − raw/退化base` | 冻结刻度分 | 官方刻度未夹持核心 (b_lo/r_lo) |
|---|---|---|---|---|---|
| A | 1.2422 | **是** | 0.5219 | −0.2422 | −0.2674 |
| V2 | 3.4226 | **是** | −0.3172 | −2.4226 | — |
| V3 | 1.8353 | **是** | 0.2937 | −0.8353 | — |
| V4 | 1.2422 | **是** | 0.5220 | −0.2422 | — |
| V5 | 1.5772 | **是** | 0.3930 | −0.5772 | −0.6172 |
| V6 | 1.0347 | **是** | 0.6018 | −0.0347 | −0.0509 |
| V7 | 1.0425 | **是** | 0.5988 | −0.0425 | −0.0590 |
| V8 | **1.0819** | **是** | **0.5836** | **−0.0819** | **−0.1001** |
| 退化基线 | 2.5985 | **是** | 0.0000 | — | — |

**(a) 是 —— 八个 variant 无一例外，raw 全部 > 1.0，即全部已经差于「原样贴对照」。**

**(b) 是 —— 0.5836 完完全全是分母伪影。** 算术恒等式（**measured**）：
`1 − 1.08188 / 2.59845 = 0.58365`，与我们一直在引用的 0.5836 逐位吻合。也就是说这一项的
全部「收益」来自退化基线自己烂到 2.5985（是「贴对照」的 2.6 倍差），而不是我们做对了
什么 —— 我们自己也在 1.0819，也在无技巧点的错误一侧。

**一句直白话：0.2025 里 0.5836/6 = 0.0973（48%）由一个我们其实没有得分的指标贡献，
它只是比一个被 unbiased MSE 重罚的退化基线不那么烂。F25 说「78% 不是生物学」是**低估**了
—— 在官方刻度上 V8 六项里只有 `pds_cosine` 一项实质得分（0.5855/6 = 0.0976，占
0.1215 的 80%），其余五项合计 0.0239。**

一个次级观察（**measured**）：该 policy 的 `clamp_low = 0.0`，所以这一项**永远不会为负**
—— 官方刻度上我们读到 0.0000 而不是 −0.10，是地板夹的。这也交叉验证了抓下来的排行榜
表：`jac` 的 policy 是 `clamp_low=None, metric_min=0.0`（无地板），所以第一名的 `jac`
能印成 **−0.004**；而 `mse` 有 0.0 地板，所以第一名的 `mse` 只能是 ≥ 0 的 0.041。
两张表在**哪些指标允许负数**这一点上完全一致，这是抓取正确性的独立证据。

---

## 8 · 如果未来只能做一件事：修**比较器**，不是修 anchor

直说（依据是 §4.2 的分解，**measured**）：

- **只换 anchor：V8 0.2025 → 0.2178 / 0.2066。anchor 让我们看起来更好，约 +1 个点。**
- **只换 baseline：V8 0.2025 → 0.1013 / 0.0844。全部的 0.1899 差距都在这里。**

**所以：本仓库该修的是比较器（用 `build_generic_baseline` 造官方 generic-response 基线），
不是 anchor。anchor 是一个二阶修饰，而且方向对我们有利。未来的 session 如果只能做一件事，
就去建 generic-response 基线。**

anchor 仍然值得要，理由只有两条，都与「提分」无关：(1) 它是排行榜真实的刻度形状，
(2) 它只依赖 `real.h5ad`，一次算好被所有未来 variant 复用（§9）。它是**基础设施，不是
发现**，可以等一台空闲的机器。

**anchor 本地计算的当前状态：已主动终止，0 / 5 个 split 落盘。** 它当时占着约 1 GB RSS、
预计还要 1.5–2 小时，而 generic-response 基线正因为它占用的 swap 而无法启动 —— 按上面
自己的数字，那是把资源投在了二阶项上。驱动是**可断点续跑**的（逐 split 落盘，seed 由
`numpy.random.SeedSequence(0).generate_state(n)` 生成，前缀天然嵌套，所以已完成的 k 个
split 恰好就是官方 anchor 的前 k 个），因此这是**暂停而不是损失**。恢复命令：

```bash
python anchor_local.py splits 5   # 跳过已有 split，接着算
python anchor_local.py assemble   # 用已有的任意 k 个 split 组装 anchor，RESULT 里标明 k
```

`from_replicate` 的管线已用**合成 anchor 帧**端到端验证通过（**measured**），所以真 anchor
一落地就能直接接上、不会在最后一步挂掉：以官方 r 中点作 1 端、我们的退化基线作 0 端，
八个 variant 全部算出（V8 `from_baseline` 0.2025 → `from_replicate` 0.2119）。

---

## 9 · 让整个仓库默认走两端刻度，要做什么

**anchor 只依赖 real 侧，一次计算服务所有未来 variant（measured，从代码确认）。**
`cached_anchor`（`anchor.py:914-955`）的缓存键是
`fingerprint_adata(real_ad, pert_col, strict=True)` +
`anchor_cache_params`（= `anchor_semantic_params(cfg, real_ad, names)` +
`base_seed` + `n_splits` + `seed_derivation` + `metric_names` + `cell_eval2_version`）。
**里面没有任何一项与预测有关。** 所以同一个 `real.h5ad` + 同一个评分配置下，anchor 是常量，
A / V2…V8 以及未来所有 variant 共用同一份。

存放位置：`anchor_store(cfg)`（`anchor.py:847-854`）返回 `CacheStore(cfg.cache_real)` ——
即**real 侧缓存目录**，与该数据集的 pseudobulk / DE 产物同居；`cfg.cache_real is None` 就
没有 anchor 缓存（是「没缓存」，不是错误）。本实验的产物落在
`experiments/E29-official-baseline/out/anchor_agg.parquet`（+ `anchor_splits.parquet`
+ 逐 split 的 `anchor_split_<i>.json`）。

要让默认打分变成两端刻度，最小改动是三步：

1. 给评分配置设 `cache_real=<repo>/.cache/cell_eval2_real`，跑一次
   `cached_anchor(real, cfg, store=anchor_store(cfg))`，anchor 从此常驻。
2. 打分时改成
   `score_metrics(user_wide, base_wide, comparison_statistic="mean", anchor=<dir>,
   anchor_expect=expect_from_run_meta(run_meta))`，读 `from_replicate` 列而不是
   `from_baseline`。**注意 `score.py:641-649`：带 anchor 时
   `comparison_statistic` 必须是 `"mean"`**（anchor 的 replicate 是五个 split 聚合的均值）。
3. `anchor_expect` 需要 `run_meta.json` 且 **`source_fingerprint_strict` 必须为真**
   （`score.expect_from_run_meta`，`score.py:364-370`：anchor 的门是严格内容哈希）。
   vcc2026 preset 已经 `cache_strict: true`，所以 `build_run_meta(cfg, real, pred)` 直接
   可用 —— 但它需要 **pred 侧对象/路径**，而我们把 `.h5ad` 删了。
   两条出路：(i) 打分时顺手写 `run_meta.json`，或 (ii) 像本实验一样直接用
   `_replicate_entries` + `_reference_column` 加列，绕过 `AnchorExpect` 门 —— 门校验的是
   「这个 anchor artifact 配不配这次 run」，在同一进程内自己刚算出来的 anchor 上它是冗余的。
   本实验走的是 (ii)，并且**没有重写任何打分算术**。

---

## 10 · 诚实的边界

- §4 的数字带着 8 扰动 / 1 context 的错配，**不能**当成「我们在官方 val 上能拿多少」。
- 官方 b、r 是从实时榜**反算**的（`docs/01-scoring.md:162-163` 记录了反算方法），区间跨
  三个 context；本文件两端都算了，没有取中点。这是 `inferred` 的部分：要 measure 它，
  需要官方的 frozen real bundle，我们没有。
- §3 的冻结刻度 1 端是「真实矩阵本身」而非实测复本，所以它**不是**排行榜用的那把尺；它的
  价值在于 0 端是无技巧点常数，读起来绝对。`scales.py:157-161` 自己警告：scale 名字是
  标签不是认证，`score` 施加 scale 时不校验 run identity。
