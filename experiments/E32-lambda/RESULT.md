# E32 · lambda 曲线：F31 的表混了 K 规则，两种刻度对 argmax 意见相反

**头条三行。** (1) F31/T10 当作"同一条 lambda 曲线"引用的三行里，两行跑在
`K = 29/288/G`、一行跑在 `K = 288 flat`，所以那条曲线是混淆的，"reach 对 lambda
饱和"这一更正**未被测量**；匹配 K 下只有一道 (0.5, 0.7) 的悬崖。(2) 内部刻度与
官方刻度对 lambda 的 argmax **意见相反**（mixed K 上内部选 λ=0.5、官方选 λ=1.0），
翻转 100% 由官方刻度上恒为 0.0000 的 `expr_mse` 造成。(3) 两个新 lambda 点
（0.85、1.0）**尝试过、未测量**，均在机器 I/O 抖动下被协调者终止；已测 flat-K
网格 {0.5, 0.7} 上 λ=0.7 仍是两种刻度共同的最优。

以下三个结果不需要新算力，全部 `measured`，其中两个推翻 findings 文件现存结论：

1. **F31 的 lambda 表混了 K 规则。** 被当作"同一条 lambda 曲线"引用的
   `lambda 1.0 / 0.7 / 0.5` 三行里，两行跑在 `K = 29/288/G`，一行跑在
   `K = 288 flat`。`pds_cosine` 原始值对 lambda **不变**、只随 K 变
   （mixed → 0.7143，flat → 0.7500），这正是混淆的指纹。
2. **F30 的"更正"本身也建立在混淆证据上。** 那条更正断言
   `direction_reach` 对 lambda **饱和**，依据是 0.1424 / 0.1419 / 0.0132 的三元组。
   在**匹配的** `K = 288 flat` 下只存在两个点，reach 原始值 0.0219 (λ=0.5) 与
   0.1482 (λ=0.7)：这是 (0.5, 0.7) 区间内的一道**悬崖**，对 λ>0.7 的行为
   **一个字都没说**。饱和是未测量的，不是证据薄弱。
3. **两种刻度对 lambda 的 argmax 意见相反**，而且这个分歧本身是已测量的，
   见下文 §4。这是本切片的头条。

---

## 0 · 刻度声明（必读，否则下面每个数都会被误读）

- `内部 avg` = `score_metrics(agg_x, agg_base)` 返回的 `from_baseline` 列，
  锚点 0/1，分母是 E27 的 **control-mean-tiled 退化基线**
  （`experiments/E27-six-metrics/out/agg_base.parquet`）。这是仓库内部刻度，
  V8 的 0.2025 就在这个刻度上，本文所有内部数字与它同刻度、可直接比较。
- `官方 lo / hi` = 官方两端刻度 $(u-b)/(r-b)$，用 OfficialBaseline 的共享 helper
  `experiments/E29-official-baseline/anchor_local.py` 的 `official_two_ended()`
  算出，它复现 `cell_eval2` 自己的 replace-anchor 路径（`score.py:431` 的
  `replace(..., anchor=float(r))` 加 `score.py:270` 的 `scoring.score_one`），
  不是手工算术。**两端全报，从不取中点**；`b`/`r` 区间转录自
  `docs/01-scoring.md:167-174`。接线已校验：本文复现 V8 = 0.1215 (lo) / 0.0848 (hi)，
  与 OfficialBaseline 给的参考值逐位一致。
- **尺度不匹配，必须随数字一起读：** 分子是我们的 **8 扰动 / 1 context** 原始均值，
  分母 `b`/`r` 是官方的 **300 扰动 / 3 context** 区间。官方列可以摆在
  leader 的 0.1899 旁边，但**不是** apples-to-apples。同 panel 的臂间比较有效，
  绝对水平与任何隐含的榜上差距都是 `inferred`。
- 全文每个数字标注 `measured` 或 `inferred`。没有标注的行是叙述，不是数据。

---

## 1 · 跑了什么命令（逐字）

生成器（单旋钮纪律，F29）：

```
cd experiments/E32-lambda
/Users/chetianc/vcc2026/.venv/bin/python gen_build.py 10 0.85     # -> build_v10.py
/Users/chetianc/vcc2026/.venv/bin/python gen_build.py 12 1.0      # -> build_v12.py
/Users/chetianc/vcc2026/.venv/bin/python gen_build_k.py 13 25     # -> build_v13.py（K 旋钮，未跑）
```

构建与打分：

```
/Users/chetianc/vcc2026/.venv/bin/python build_v10.py build       # 完成，122 s
/Users/chetianc/vcc2026/.venv/bin/python build_v12.py build       # 完成，233 s
/Users/chetianc/vcc2026/.venv/bin/python score_lambda.py v10      # 中途被杀，未产出
/Users/chetianc/vcc2026/.venv/bin/python score_lambda.py v12      # 中途被杀，未产出
```

汇总（只读 parquet，无重算）：

```
/Users/chetianc/vcc2026/.venv/bin/python report.py | tee out/report.txt
```

`gen_build.py` / `gen_build_k.py` 的纪律是可执行的、不是口头的：每次替换后断言
旧字面量消失且新字面量出现，最后再断言"生成文件与模板的差异行数**恰好** 3 行"
（旋钮行、OUT 目录行、提交标签行）。两个生成器实测都报
`改动行号 [90, 93, 214]`（lambda）与 `[90, 188, 214]`（K），`measured`。
旧字面量刻意带一个尾随空格（`"LAMBDA = 0.7 "`、`"K = 288 "`），否则
`LAMBDA = 0.775` 会把 `LAMBDA = 0.7` 当子串、断言静默失效 —— 这是本切片
唯一一处"差点犯的" F29 类错误。

### 1.1 单旋钮之外的三处替换，逐条交代

任务要求"恰好一次 `str.replace`"。语义上确实只有一次：**只有旋钮字面量**
（`LAMBDA` 或 `K`）被改。另外三处替换是记账性的、不碰任何计算路径，各自单独标注：

| 替换 | 内容 | 是否影响计算 |
|---|---|---|
| 1（旋钮） | `LAMBDA = 0.7 ` → `LAMBDA = 0.85 ` / `1.0 `；或 `K = 288 ` → `K = 25 ` | **是，唯一语义改动** |
| A（注释） | 该行过时的 `# V8 唯一改动…` 注释 | 否 |
| B（目录） | `/ "E28-pds" / "out"` → `/ "E32-lambda" / "out"` | 否，只换落盘位置 |
| C（标签） | `("pred_v8", X_pred)` → `("pred_v10"/"v12"/"v13", X_pred)` | 否，只换文件名 |

首行 docstring 仍写着"变体 V8"，`measured`：这是纯文档串，再加一次替换只增噪，
故保留原样并在此声明。

### 1.2 打分器的警告文本（逐字）

每次 `score_metrics` 调用都打印，`measured`：

```
metric 'expr_distance_unbiased' not scored (scored=False)
metric 'expr_mse_unbiased' not scored (scored=False)
metric 'expr_mse_unbiased_capped' not scored (scored=False)
metric 'expr_real_mass_ratio' not scored (scored=False)
```

构建侧无警告。两个构建的扰动选择、gate、共同基因数逐位相同，`measured`：
`gate 10,779  对照 18,400  MDE 中位 0.0149`，`共同基因 7,016`，
`[TCF7L2, GNG12, VCL, COX4I1, MAT2A, PAXIP1, SLIRP, ZNF581]`，
提交文件 `(21600, 18080)`、292 MB gzip。即两个变体之间除 LAMBDA 外确实没有别的差别。

---

## 2 · 三旋钮标注表：混淆一目了然

这是本切片要求的核心交付物。**同一张表里必须同时印 ordering、K 规则、lambda**，
否则"lambda 曲线"这个说法就是错的。全部 `measured`，源自
`experiments/E28-pds/out/agg_v{5,6,7,8}.parquet`。

| 变体 | ordering | K 规则 | lambda | 内部 avg | 官方 lo | 官方 hi |
|---|---|---|---|---|---|---|
| V5 | `\|beta\|` | 29/288/G | 1.00 | 0.1559 | 0.1028 | 0.0675 |
| V6 | `\|beta\|` | 29/288/G | 0.50 | 0.1762 | 0.0909 | 0.0552 |
| V7 | `\|beta\|` | 288 flat | 0.50 | 0.1873 | 0.1041 | 0.0668 |
| V8 | `\|beta\|` | 288 flat | 0.70 | **0.2025** | **0.1215** | **0.0848** |

原始均值（`measured`）：

| 变体 | lambda | K 规则 | reach | expr_mse_capped_norm | lfc_nmae | pds_cosine | sig_jaccard | fid_yield |
|---|---|---|---|---|---|---|---|---|
| V5 | 1.00 | 29/288/G | 0.1487 | 1.5772 | 1.0141 | **0.7143** | 0.0483 | 0.4898 |
| V6 | 0.50 | 29/288/G | 0.0204 | 1.0347 | 0.9882 | **0.7143** | 0.0484 | 0.4978 |
| V7 | 0.50 | 288 flat | 0.0219 | 1.0425 | 0.9885 | **0.7500** | 0.0484 | 0.4963 |
| V8 | 0.70 | 288 flat | 0.1482 | 1.0819 | 1.0004 | **0.7500** | 0.0482 | 0.4919 |

内部 `from_baseline` 逐指标（`measured`）：

| 变体 | reach | expr_mse | lfc_nmae | pds | sig | fid |
|---|---|---|---|---|---|---|
| V5 | 0.1424 | 0.3930 | −0.0164 | 0.4286 | 0.0008 | −0.0129 |
| V6 | 0.0132 | 0.6018 | 0.0096 | 0.4286 | 0.0010 | 0.0032 |
| V7 | 0.0147 | 0.5988 | 0.0092 | 0.5000 | 0.0009 | 0.0000 |
| V8 | 0.1419 | 0.5836 | −0.0026 | 0.5000 | 0.0007 | −0.0087 |

官方两端逐指标（`lo / hi`，`measured`）：

| 变体 | reach | expr_mse | lfc_nmae | pds | sig | fid |
|---|---|---|---|---|---|---|
| V5 | +0.1116/+0.0586 | 0.0000/0.0000 | −0.0209/−0.0217 | +0.5018/+0.4427 | +0.0770/+0.0292 | −0.0526/−0.1040 |
| V6 | −0.0291/−0.0869 | 0.0000/0.0000 | +0.0201/+0.0237 | +0.5018/+0.4427 | +0.0775/+0.0296 | −0.0247/−0.0780 |
| V7 | −0.0276/−0.0853 | 0.0000/0.0000 | +0.0196/+0.0231 | +0.5855/+0.5165 | +0.0774/+0.0295 | −0.0302/−0.0830 |
| V8 | +0.1110/+0.0581 | 0.0000/0.0000 | +0.0009/+0.0023 | +0.5855/+0.5165 | +0.0769/+0.0291 | −0.0453/−0.0972 |

### 2.1 永远不要再引用的两行

> **`lambda = 1.0` / `K = 29/288/G` 这一行（V5，内部 0.1559）与
> `lambda = 0.5` / `K = 29/288/G` 这一行（V6，内部 0.1762），
> 不得再与 V7/V8 并列当作同一条 lambda 曲线上的点。**

它们与 V7/V8 相差**两个**旋钮（lambda 与 K 规则），不是一个。判别证据是
`pds_cosine` 原始值：mixed K 下恒为 0.7143，flat K 下恒为 0.7500，
与 lambda 无关（`measured`，四个变体全部一致）。任何把 0.7143 与 0.7500
混在一列里的"lambda 曲线"都已经混淆了。

仓库里**确实**存在两条各含两点的合法 lambda 曲线，它们必须分开读：

- **mixed K = 29/288/G**：λ=0.5 → V6，λ=1.0 → V5。
- **flat K = 288**：λ=0.5 → V7，λ=0.7 → V8。

`K = 288 flat` 下 **不存在** `lambda = 1.0` 的测量点。这正是我尝试补的那个角，
也正是被资源耗尽打断的那次运行。

---

## 3 · 五点 lambda 表（按验收标准，含未测到的点）

`K = 288 flat`、ordering `|beta|`、N_PERT=8、seed=0、quantizer `hamilton`。

| lambda | 来源 | reach | expr_mse_capped_norm | lfc_nmae | pds_cosine | 内部 avg | 官方 lo / hi | 状态 |
|---|---|---|---|---|---|---|---|---|
| 0.50 | V7 | 0.0219 | 1.0425 | 0.9885 | 0.7500 | 0.1873 | 0.1041 / 0.0668 | `measured`（引用） |
| 0.60 | — | — | — | — | — | — | — | **取消**，从未构建 |
| 0.70 | V8 | 0.1482 | 1.0819 | 1.0004 | 0.7500 | **0.2025** | **0.1215 / 0.0848** | `measured`（引用） |
| 0.85 | V10 | — | — | — | — | — | — | **构建完成，打分中途被杀，未测量** |
| 1.00 | V12 | — | — | — | — | — | — | **构建完成，打分中途被杀，未测量** |

（`lambda = 1.0` 在 **mixed** K 下有测量点 V5：reach 0.1487、expr_mse 1.5772、
lfc_nmae 1.0141、pds 0.7143、内部 0.1559、官方 0.1028 / 0.0675 —— 但它**不在**
这张表的设计上，只能作为另一条曲线的点引用。）

### 3.1 三个点为什么没有数字

全部为协调者在测量资源耗尽下的明确指令，**不是代码失败**：

- **λ = 0.85（V10）** —— 构建成功（`pred_v10.h5ad` 292 MB，122 s）。打分进行约
  10 分钟后被协调者取消并降级为"纯好奇心的内点"。`agg_v10.parquet`
  **从未落盘**，因此没有任何数字可抢救。标注：**cancelled mid-scoring, not measured**。
- **λ = 1.0（V12）** —— 构建成功（292 MB，233 s）。打分跑到 wall 29:20 时被
  协调者下令终止，理由是实测 CPU 时间仅 00:14（利用率 0.8%），进程
  page-fault bound：数据卷 99% 满、系统级 pagein 约 320 GB，swap 在同一个满卷上
  无法增长。**29 分钟里几乎没有发生计算**，所以终止不是丢弃进度、而是回收抖动。
  标注：**attempted, killed for resource exhaustion, not measured**。
- **λ = 0.6** —— 由协调者在 `expr_mse` 发现之后取消（见 §4：官方刻度下收缩方向
  只可能亏），随后随整个 lambda 方向一起降级为最低优先级。从未生成、从未构建。
  标注：**cancelled before build**。

两个提交文件在被杀的同时删除；我的 `out/` 现在只剩 `perts.csv` 与 `report.txt`，
不占任何提交空间（`measured`：删除后 `df` 从 6.0 GiB 回到 8.3 GiB）。

**恢复命令（逐字，任一单进程时段即可跑）：**

```
cd experiments/E32-lambda
# λ = 1.0（最高价值：匹配 K 下缺的那个角，决定 reach 是否饱和）
/Users/chetianc/vcc2026/.venv/bin/python build_v12.py build
/Users/chetianc/vcc2026/.venv/bin/python score_lambda.py v12
# λ = 0.85（(0.7, 1.0) 的内点，仅在 1.0 有意思时才值得）
/Users/chetianc/vcc2026/.venv/bin/python gen_build.py 10 0.85
/Users/chetianc/vcc2026/.venv/bin/python build_v10.py build
/Users/chetianc/vcc2026/.venv/bin/python score_lambda.py v10
# 两刻度汇总（report.py 的 POINTS 已含 V12；加 V10 需在该列表里补一行）
/Users/chetianc/vcc2026/.venv/bin/python report.py
```

`build_v12.py` 已在树上、已构建成功过一次，直接跑即可；`build_v10.py` 同样在树上，
上面重列 `gen_build.py` 只为让"它从哪来"可复现。预算：构建约 122–233 s，
`compute_metrics + score_metrics` 在**不抖动**的机器上约 1,289 s；本次两次失败
都不是时长问题，而是 0.8% CPU 利用率的 I/O 抖动。

---

## 4 · 头条：两种刻度对 lambda 的 argmax 意见相反

这个分歧不需要新算力，它已经在 parquet 里，`measured`。

**mixed K = 29/288/G 上的 lambda 对（V6 λ=0.5 vs V5 λ=1.0）：**

| 刻度 | λ=0.5 | λ=1.0 | 谁赢 |
|---|---|---|---|
| 内部 `from_baseline` | 0.1762 | 0.1559 | **λ=0.5**，差 −0.0203 |
| 官方 lo | 0.0909 | 0.1028 | **λ=1.0**，差 +0.0119 |
| 官方 hi | 0.0552 | 0.0675 | **λ=1.0**，差 +0.0123 |

**排序翻转了，而且原因是单一成员。** 逐指标分解（λ=1.0 减 λ=0.5，`measured`，
六项之和除以 6 与 avg 差值逐位吻合）：

| 成员 | 内部 Δ | 官方 lo Δ |
|---|---|---|
| `direction_reach` | **+0.1292** | **+0.1407** |
| `expr_mse_capped_norm` | **−0.2088** | **0.0000** |
| `lfc_nmae` | −0.0260 | −0.0410 |
| `pds_cosine` | 0.0000 | 0.0000 |
| `sig_jaccard` | −0.0002 | −0.0005 |
| `fid_yield` | −0.0161 | −0.0279 |
| 合计 / 6 | **−0.0203** ✓ | **+0.0119** ✓ |

内部刻度把 −0.2088 判给了"提高 lambda"，官方刻度判 0.0000。翻转**完全**由这一项
造成。机制：`expr_mse_unbiased_capped_norm` 的 catalog 策略 `clamp_low = 0.0`，
其 no-skill 基准是 1.0（"原样输出对照"），而我们八个变体的原始值全部 > 1.0，
于是官方刻度上一律读作 0.0000。内部刻度看到的 0.3930 / 0.6018 只是
我们那个退化基线比"粘贴对照"差 2.6 倍的产物 —— 对 V8 而言
$1 - 1.08188/2.59845 = 0.58365$，与记录的 0.5836 逐位一致（`measured`）。

**flat K = 288 上的 lambda 对（V7 λ=0.5 vs V8 λ=0.7）：** 两种刻度**一致**。

| 刻度 | λ=0.5 | λ=0.7 | 差 |
|---|---|---|---|
| 内部 | 0.1873 | 0.2025 | +0.0152 |
| 官方 lo | 0.1041 | 0.1215 | +0.0174 |
| 官方 hi | 0.0668 | 0.0848 | +0.0180 |

这里一致，是因为 λ 0.5→0.7 时 `expr_mse` 内部只动了 −0.0152（不像
0.5→1.0 时的 −0.2088），不足以翻转 reach 的 +0.1272。

**推论（`measured` 前提 + `inferred` 结论）：** 官方刻度上收缩（降低 lambda）
的整个回报是假的 —— 它换来的 `expr_mse` 增益恒为 0.0000。因此官方刻度下
lambda 应当被**推高**，而不是压低。这与内部刻度过去两轮给出的方向相反。

---

## 5 · `pds_cosine` 不变性：确认了什么、没确认什么

- **确认（`measured`）：** `pds_cosine` 原始值对 lambda **完全不变**，在两条曲线上
  各自恒定 —— mixed K 下 λ=0.5 与 λ=1.0 都是 **0.7143**；flat K 下 λ=0.5 与
  λ=0.7 都是 **0.7500**。余弦对尺度不变，收缩整个 lfc 向量不动其方向。
- **修正（`measured`）：** 任务书与 F31 都说"lambda 1.0 / 0.7 / 0.5 三点 pds 恒为
  0.7143"。**这是错的。** 0.7143 属于 mixed K，0.7500 属于 flat K。
  pds 不随 lambda 变，但**随 K 变**（+0.0357 原始，官方刻度 +0.0837 lo / +0.0738 hi）。
- **未确认：** λ=0.85 与 λ=1.0 在 **flat K** 下的 pds 没有测到，因为两次运行都被杀。
  依据尺度不变性，预期仍是 0.7500（`inferred`）。**会测到它的**动作：
  `python build_v12.py build && python score_lambda.py v12`，然后读
  `out/agg_v12.parquet` 的 `pds_cosine` 行 —— 生成器与打分器都已就位、已验证，
  只缺一个单进程时段。

---

## 6 · 形状裁决

**不是单肩，也不是平台，是一道悬崖加一段未测区间。**

- **匹配 K = 288 flat 下已测量的形状：** reach 原始 0.0219 (λ=0.5) → 0.1482 (λ=0.7)。
  6.8 倍跃升，跨度只有 0.2 的 lambda。这是 **(0.5, 0.7) 区间内的一道悬崖**，
  与"reach 的 k\* 受符号纯度上限约束、只要实现的 lfc 高于 bootstrap 噪声就保持"
  这一机制一致：λ 降到某处时实现的 lfc 掉到噪声之下，符号随机化，k\* 崩塌。
- **同一道悬崖在 mixed K 下也在：** 0.0204 (λ=0.5) → 0.1487 (λ=1.0)。
- **饱和是未测量的。** 产生"饱和"故事的那个近似相等 —— 0.1487 (λ=1.0) 对
  0.1482 (λ=0.7) —— **跨了两个 K 规则**，一个在 mixed、一个在 flat。
  在任何单一 K 规则下，λ > 0.7 的 reach 行为**没有任何测量**。
- **仍未测量的 bracket：**
  - **flat K = 288 上的 (0.70, 1.00]** —— 全空。argmax 可能是 0.7，也可能在
    0.85 或 1.0；本切片两次尝试正是要填这里。
  - **flat K = 288 上的 (0.50, 0.70)** —— 悬崖的确切位置未定位。
  - **mixed K 上的 (0.50, 1.00)** —— 整段开区间未采样，只有两个端点。
- **argmax 裁决：** 在**实际测到的** flat-K 网格 {0.50, 0.70} 上，
  **λ = 0.70 是 argmax，两种刻度都是**（内部 0.2025 > 0.1873；
  官方 0.1215 > 0.1041 lo，0.0848 > 0.0668 hi）。但这个网格只有两个点，
  而 §4 的 mixed-K 证据显示官方刻度偏好**更高**的 lambda，
  所以 **λ=0.7 是"已测最优"，不是"已证 argmax"**。
  两个新点都没有打败 0.2025 —— 因为两个新点都没有数字。**这就是结果，如实报告。**

---

## 7 · 为什么 lambda 方向本身已经不该再投算力

协调者随后基于 SourceUnion 的位精确分解给出的判断，本切片接受并记录
（数字为 SourceUnion `measured`，本切片未独立复现）：V8 的 reach = 0.1482 中
**84.3% 来自单个扰动 MAT2A**（`N_conf=5`、`k*=5`、reach 恰好 1.0000），
而 MAT2A 的 confident pool 里**没有一个**我们召集的 288 个基因；
其满分来自解码器近零预测在质量重归一化后系统性略偏负，恰好匹配了
那 5 个参考显著基因的负向真实 lfc。panel 的 `k*` 中位数是 2。

因此：**lambda 唯一真正推动的成员是 reach，而 reach 约五分之四是质量守恒伪影。**
优化 lambda 等于优化一个空心成员，与此前在 `expr_mse` 上花掉两轮是同一个错误。
`expr_mse` 官方刻度恒 0.0000、`nmae`/`fid`/`jac` 是噪声（悲观端两项为负），
官方刻度上我们**只**在 `pds_cosine` 上得分（V8：0.5855 lo / 0.5165 hi，
占 0.1215 的约 80%）。这是 lambda 曲线被降到最低优先级的正当理由，
也是本切片同意停手的理由。

---

## 8 · 已交付但未运行：K = 25 单旋钮变体

协调者把 K 旋钮改派给本切片，并列为单槽队列的第一优先。生成器已写好并验证：

```
/Users/chetianc/vcc2026/.venv/bin/python gen_build_k.py 13 25
→ 已生成 build_v13.py  K=25  改动行号 [90, 188, 214]     （measured）
```

唯一语义替换是 `K = 288 ` → `K = 25 `（模板里该字面量只出现在 `build_v8.py:188`，
`measured`）。生成器额外断言 `"LAMBDA = 0.7 " in txt`，即 K 变体不得动 lambda。
`build_v14.py`（K=10）用 `gen_build_k.py 14 10` 即可产出。

运行它需要一个单进程时段与 `df > 8 GB`；协调者已宣布单槽串行且 token 在
OfficialBaseline 手上，故**未启动**。命令就是：

```
cd experiments/E32-lambda
/Users/chetianc/vcc2026/.venv/bin/python build_v13.py build
/Users/chetianc/vcc2026/.venv/bin/python score_lambda.py v13
/Users/chetianc/vcc2026/.venv/bin/python report.py
```

唯一可用的解释器是 `/Users/chetianc/vcc2026/.venv/bin/python`；`python` / `python3`
都没有 `cell_eval2`、`anndata`、`scanpy`、`polars`。

**事先写下预期以便被推翻，而不是事后附会：** `pds_cosine` 应当**下降**，
因为余弦被大坐标支配、非零坐标变少意味着方向向量更稀疏更噪
（V7 的全部 +0.0714 就来自 29/288/G → flat 288 的坐标增多）。
若 pds 的跌幅超过 DE 成员的增益，则小 K 亏，"稀释"就是 pds 为坐标付的价；
若 pds 稳住而 `lfc_nmae` 与 `fid` 改善，则 K=288 一直是错的。
依据是 SourceUnion 已验证 reach 估计器测得的符号一致率随深度衰减
（k=1 87.5%、k=3 83.3% p=0.0008、k=10 65.0% p=0.0048、k=25 55.0%、
k=288 52.4% 即掷硬币，`measured` by SourceUnion）。

**报告 K 臂时必须同时声明（`inferred` 警告）：** 8 扰动 panel 上测得的 pds 原始值
**不能**外推到 300 —— `discrimination.py:215` 令 $D = n-1$ 取自 panel 内非对照扰动数，
`:248` 在 `exclusion_scope="panel"` 下把每个 panel 目标基因从两侧操作数里剔除。
同 panel 的 K 臂之间比较有效；绝对水平与任何隐含的榜上差距是 `inferred`。

---

## 9 · 文件清单

| 文件 | 作用 | 状态 |
|---|---|---|
| `gen_build.py` | lambda 旋钮生成器，单语义替换 + 断言 + 3 行差异自证 | 已跑 |
| `gen_build_k.py` | K 旋钮生成器，同纪律，额外断言 LAMBDA 未动 | 已跑 |
| `build_v10.py` | λ=0.85，由 `gen_build.py 10 0.85` 生成 | 构建过，未测量 |
| `build_v12.py` | λ=1.0，由 `gen_build.py 12 1.0` 生成 | 构建过，未测量 |
| `build_v13.py` | K=25，由 `gen_build_k.py 13 25` 生成 | 未运行，等 token |
| `score_lambda.py` | 打分器，逐字照 `E28-pds/score_v8.py` 的 EvalConfig 三步与 AUTO_CLEAN | 两次被杀 |
| `report.py` | 两刻度并列汇总，只读 parquet | 已跑 |
| `out/report.txt` | `report.py` 的原始输出 | 已落盘 |
| `out/perts.csv` | 两次构建选中的 8 个扰动 | 已落盘 |

`out/` 内无任何 `.h5ad`。本切片未执行任何 `git` 写操作，未编辑 `docs/**`，
未编辑 `src/vcclab/**`，未触碰其他 agent 的目录，未设置任何线程/BLAS 环境变量。
