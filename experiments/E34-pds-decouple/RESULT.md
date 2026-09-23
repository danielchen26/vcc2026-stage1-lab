# E34 · RESULT —— V13：`pds` 赢了并超过榜首，但 SPEC §1 的机制论证被推翻

**判定：分数 GO（`avg_lo` 0.1215 → 0.1508，+0.0293），机制 NO。**

`score_v13.py` 打印的 `GO` 只看了 `avg_lo`，**这是我的判定代码写漏了**：
SPEC §5 明写停止条件「优先级高于本实验本身」，而两条停止条件**都触发了**。
正确的读法是：**赢了，但赢的原因不是我说的那个**，且赢法是一笔**交易**，不是免费收益。

## 尺度声明（SPEC §3 强制）

三列都报。官方两端由 `E29-official-baseline/anchor_local.official_two_ended_from_raw`
算出（$b$/$r$ 取 `docs/01-scoring.md:167-174` 从实时榜反算的表），未自写算术。
仓库内部列 = `score_metrics(pred, E27/agg_base.parquet, comparison_statistic="mean")`。

⚠️ **分子是本 8 扰动 / 1 context 的 raw，分母是官方 300 扰动 / 3 context 的 $b$/$r$。**
可以并排看榜首 0.1899，**但不是同一回事**。本文件不宣称追平或超过总分。

## 实测（`out/score_v13.json`，`compute_metrics` 1422 s）

| 指标 | V8 raw | V13 raw | V8 lo | V13 lo | Δlo | V8 hi | V13 hi | V13 内部 |
|---|---|---|---|---|---|---|---|---|
| `pds_cosine` | 0.7500 | **0.8750** | 0.5855 | **0.8782** | **+0.2927** | 0.5165 | 0.7748 | 0.7500 |
| `expr_mse_capped_norm` | 1.0819 | 2.2208 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 | 0.1453 |
| `sig_jaccard` | 0.048234 | 0.048247 | 0.0769 | 0.0770 | +0.0000 | 0.0291 | 0.0291 | 0.0008 |
| `lfc_nmae` | 1.0004 | 1.0115 | 0.0009 | −0.0168 | −0.0177 | 0.0023 | −0.0172 | −0.0138 |
| `direction_fidelity` | 0.4919 | 0.5023 | −0.0453 | −0.0092 | +0.0361 | −0.0972 | −0.0634 | 0.0121 |
| `direction_reach` | 0.1482 | **0.0250** | 0.1110 | **−0.0241** | **−0.1352** | 0.0581 | −0.0817 | 0.0178 |
| **avg_score** | | | **0.1215** | **0.1508** | **+0.0293** | **0.0848** | **0.1069** | 0.1520 |

**`pds_cosine` raw 0.8750 超过榜首的 0.820**（scaled 0.8782 vs 榜首 0.708）。
连同 `sig_jaccard`，我们现在在**两个**成员上高过榜首。

## 预测 vs 实测（SPEC §4 在 build 前写死，git 可证）

| 量 | 预测 | 实测 | 判 |
|---|---|---|---|
| `pds` raw | 0.8929（+8 量子） | 0.8750（+7 量子） | 差 1 个量子，方向与量级对 |
| `sig_jaccard` | **逐位不变** | 变了 $1.3\times10^{-5}$ | ❌ |
| `direction_reach` | **逐位不变** | 0.1482 → 0.0250（塌 83%） | ❌ |
| `mse` | 不可能变差（已在地板） | raw 翻倍到 2.2208，官方仍 0.0000 | ✅ |
| `nmae` | 悲观界 −0.0059 | −0.0177 | ❌ 悲观界不够悲观，差 3 倍 |
| `fid` | ±未知 | **+0.0361** | 未预测到的收益 |
| `avg_lo` | 0.1714 | 0.1508 | 低 0.0206 |

**6 个成员里预测对 1 个。** `avg` 的方向对了，但逐成员的理由几乎全错 ——
这正是 SPEC 存在的理由：如果没有预先写下逐成员预测，我会把 +0.0293 当成
「机制验证成功」收下，而实际上机制是错的。

## 错处 1 · `direction_reach` 的排序不在 `r_set` 上

SPEC §1 写：「`reach` 测 $k^\ast/N_{\text{conf}}$，由 `r_set` 的排序决定。`r_set` 不动 ⇒ reach 不动。」

**错。** 纯前缀是在**我们自己全部非零 lfc 的 $\lvert\text{lfc}\rvert$ 降序**上取的，
不是在 `r_set` 上。集外那 550 个去均值 lfc 中有一部分幅度**大于**召集集成员，
于是它们挤进前缀；而它们的符号来自跨系不可迁移的源
（[F33](../../docs/02-findings.md#f33)：符号一致率 51–57%），纯度立刻破，$k^\ast$ 崩。

`build_v3.py` 的 docstring 三周前就写过同一句话 ——
「去面板均值让排序不再跟源侧置信度对齐，纯前缀立刻崩掉」。
**我读过它，还是把它当成只适用于「改写召集集」的情形。** 它适用于任何进入
$\lvert\text{lfc}\rvert$ 排序的东西。

→ 修法是结构性的、可判的：**把集外幅度整体压到召集集最小值之下**，排序即恢复。
由 `cap_probe.py` 验证（同时量 pds 收益与「$\min\lvert\text{call}\rvert > \max\lvert\text{off}\rvert$」不变量）。

## 错处 2 · 我们不直接控制 $\hat R$，打分器自己重算

`sig_jaccard` 动了 $1.3\times10^{-5}$。我以为显著集就是我们声明的 `r_set`。
**不是。** 打分器对**预测出来的计数**自己跑 Wilcoxon，判定线薄到
$d_{\text{crit}} = 0.0278$（[01-scoring.md](../../docs/01-scoring.md)）——
给 550 个基因加一阶矩，其中少数几个的百分位就越过了线。

**声明的召集集 ≠ 打分器认定的显著集。** 这一条影响所有把 `sig_jaccard` 当作
「我们完全控制、不会被别的改动碰到」的推理。量级极小（$10^{-5}$，折 avg $10^{-5}$ 量级），
但方向上说明 `jac` 是**有敞口**的，不是锁死的。
既然 `jac` 是我们唯一的独有优势，这个敞口必须在每次改动后逐位核对，不能假定。

## 净账：这是一笔交易，不是免费收益

| 成员 | Δlo |
|---|---|
| `pds_cosine` | **+0.2927** |
| `direction_fidelity` | +0.0361 |
| `sig_jaccard` | +0.0000 |
| `expr_mse_capped_norm` | +0.0000 |
| `lfc_nmae` | −0.0177 |
| `direction_reach` | **−0.1352** |
| 合计 / 6 | **+0.0293** |

买 `pds`，卖 `reach`。交易本身有利（两端刻度都赢：lo +0.0293、hi +0.0221），
但它**不是** SPEC §1 声称的「拿到 pds 收益且 reach 不受影响」。

### 一个被隐藏的代价：`mse` 的期权被关掉了

`mse` raw 从 1.0819 翻到 **2.2208**。官方刻度上不花钱（`clamp_low=0`，V8 已在地板），
所以「免费赌注」这一条预测是对的。但 [07-roadmap](../../docs/07-roadmap.md) 的 P4 是
「raw 须跌破官方 $b\approx0.9985$ 才开始得分，V6 的 1.0347 最近，差 4.7%」——
V13 把这个距离从 4.7% 拉到 **122%**。
**免费只在「永远不打算拿 mse」的前提下成立。** 这个代价在 SPEC §4 里没写。

## 采用与否

| 刻度 | V8 | V13 | 结论 |
|---|---|---|---|
| 官方 $b_{lo}/r_{lo}$ | 0.1215 | **0.1508** | V13 赢 +0.0293 |
| 官方 $b_{hi}/r_{hi}$ | 0.0848 | **0.1069** | V13 赢 +0.0221 |
| 仓库内部（已被 [T9](../../docs/06-traps.md#t9) 判为不可外推） | 0.2025 | 0.1520 | 不作为依据 |

**两端都赢 ⇒ V13 取代 V8 成为当前最优配置**，但标注为「交易型收益，机制待改写」。
不推进到线上提交：V14（capped）可能同时保住 `reach`，差值 0.1352 scaled
（折 avg +0.0225）远大于一次 build 的成本。

## 旋钮声明

唯一语义改动：`design_cells(..., lfc_all=la_vec)`。
`la_vec` = 集外 top-550 的去面板均值 lfc × 1.0，召集集位置强制为 0。
逐字未动：$\lambda = 0.7$（召集集）、$K = 288$ 扁平、排序统计量 $\lvert\beta\rvert$、
`N_PERT = 8`、`SEED = 0`、`hamilton`、`force_mean = True`、`DEPTH`、`ALPHA`、`K_A` 表。

生成器 `gen_v13.py` 的 6 个 hunk 已核（`diff` 干净），build 内 5 条硬断言全过 ——
其中 `召集集 lfc 被污染` 与 `集外通道落到了召集集上` 证明那 288 个 lfc 逐位等于 V8。
参数 $k_{\text{off}} = 550$、$\lambda_{\text{off}} = 1.0$ 出自 SPEC §0e，
由 `stability.py` 的密网格（500/550/600 三点平台）与留一法（8/8）定，**在 build 之前**。

⚠️ 本次日志未出现 V6/V7/V8 那条 i.i.d. 告警，但 `score_v13.py` 只在算内部刻度时调用了
一次 `score_metrics`，调用路径与 `E28-pds/score_v8.py` 不同，**不能据此说告警消失**。
i.i.d. 伪影仍是 E31 的未结事项。

## 复现

```bash
cd /Users/chetianc/code/vcc2026-stage1-lab
P=/Users/chetianc/vcc2026/.venv/bin/python
$P experiments/E34-pds-decouple/decouple_probe.py   # 设计筛选，~200 s
$P experiments/E34-pds-decouple/stability.py        # k_off 密网格 + 留一法，~35 s
$P experiments/E34-pds-decouple/gen_v13.py          # 单旋钮生成，<1 s
$P experiments/E34-pds-decouple/build_v13.py        # 437 s，+293 MB
$P experiments/E34-pds-decouple/score_v13.py        # 1422 s，成功后自动删 .h5ad
$P experiments/E34-pds-decouple/cap_probe.py        # V14 的前置判定
```
