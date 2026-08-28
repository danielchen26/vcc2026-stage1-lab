# VCC 2026 · Stage 1 Lab

Arc Institute [Virtual Cell Challenge 2026](https://virtualcellchallenge.org/) 的**工作仓库**。
不是展示页 —— 这里放可复现的实验、可复用的代码、和把讨论整理成有逻辑结构的文档，
后续的建模与理论验证都在这里做。

**核心论断：** 官方打分器对每个基因只读两个标量。因此问题可以精确拆成两级——
**预测**（Stage 1，待做）⊕ **构造**（Stage 2，已完成且无损）。
Stage 2 已与官方打分器逐基因对齐，并比它快 41 倍，全部在一台没有 GPU 的笔记本上。

---

## 当前状态

| 部分 | 状态 | 证据 |
|---|---|---|
| 打分器闭式约化（ψ 算子） | ✅ 已验证 | 与 `scipy.mannwhitneyu` 3/3 精确相等 · [E01](experiments/E01-psi-identity/) |
| 显著性与方向解耦 | ✅ 已验证 | 固定均值下 p 从 2.8e-29 荡到 0.77 · [E02](experiments/E02-dispersion-dial/) |
| Stage 2 构造器 | ✅ 已完成 | 意图 250 → 实现 250，假阳性 0 · [E03](experiments/E03-decoder-exactness/) |
| 与官方打分器对齐 | ✅ 逐基因一致 | 显著集对称差 **0**，log₁₀(p_adj) 差 **0.0000** · [E04](experiments/E04-official-parity/) |
| 可检出性（MDE）刻画 | ✅ 已实测 | 用 BH 有效阈值，范围 **6.8×**（p95/p25）· [docs/02](docs/02-findings.md#f6) |
| **理论确认：一切走 $h$** | ✅ 已完成 | 追平榜首需 $h\ge0.134$；零生物学上限 0.130 · [docs/02](docs/02-findings.md#f15) |
| **Stage 1 估计器** | ⬜ **待做** | 设计已定 · [docs/05](docs/05-stage1-design.md) |
| 自有框架评估 | ✅ 已完成 | 四套，一套是真金 · [docs/04](docs/04-framework-eval.md) |

---

## 文档索引（建议按顺序读）

| # | 文档 | 内容 |
|---|---|---|
| 00 | [问题定义](docs/00-problem.md) | 比赛在问什么、input/output 的精确规格、维度账、为什么今年难 |
| 01 | [打分规则](docs/01-scoring.md) | 六个指标、0/1 锚点怎么量出来的、闭式约化、d_crit |
| 02 | [实测发现](docs/02-findings.md) | 编号的事实清单，每条带数字与来源脚本 |
| 03 | [Stage 2 构造器](docs/03-stage2-decoder.md) | 数学形式、三个推论、耦合约束、验证结果 |
| 04 | [自有框架评估](docs/04-framework-eval.md) | ANM / bnode / AdaptiveEROP / alphagenome 逐项判决 |
| 05 | [Stage 1 设计](docs/05-stage1-design.md) | 装配好的估计器，逐步骤 + 算力预算 |
| 06 | [会静默扣分的坑](docs/06-traps.md) | 八条，四条是我们真踩过的 |
| 07 | [路线图](docs/07-roadmap.md) | 待做实验、优先级、决策点 |

交互式版本（含可拖动的演示）：见 [`app/`](app/)。

---

## 仓库结构

```
docs/               有逻辑组织的讨论（上表）
src/vcclab/         可复用 Python 包
  scorer.py         ControlRef：ψ 表构建 + de_table() 官方 DE 精确复刻
  decoder.py        Stage 2 构造器（design_cells + hamilton 取整）
  detectability.py  最小可检出效应 MDE
  shrinkage.py      James–Stein + Ledoit–Wolf→ν₀（移植自 AdaptiveEROP）
  probit.py         可检出性概率 + 非参门控
experiments/        编号实验，每个含 README / run.py / RESULT.md
scripts/            数据导出与一次性工具
app/                交互式 app（Vite + React + TS，双语，部署到 Vercel）
data/               挑战数据落地处，**不入库**
tests/              单元测试 + 集成测试
```

---

## 快速开始

```bash
# 1. 环境（复用已装好的）
export PY=~/vcc2026/.venv/bin/python
$PY -m pip install -e . --no-deps

# 2. 官方数据（不入库，自己下）
#    在 virtualcellchallenge.org 注册后：vcc datasets download controls -d data/
#    解压出 context_{A,B,C}.h5ad + gene_names.csv + pert_counts.csv

# 3. 跑测试
$PY -m pytest tests/ -q

# 4. 复现任一实验
$PY experiments/E01-psi-identity/run.py
$PY experiments/E03-decoder-exactness/run.py

# 5. 交互式 app
cd app && npm install && npm run dev
```

---

## 三十秒版本

1. **要交的不是模型，是一张表** —— 360,000 × 18,533 的整数计数矩阵。官方原话
   *only those results form your entry*。

2. **打分器每个基因只读两个数** —— 400 个细胞的平均值，和它们在 18,400 个对照细胞里的
   平均中位秩。细胞之间怎么搭配、看起来像不像真数据，**一分不加**。

3. **因为对照组从不变化**，Wilcoxon 秩和有闭式
   $U_g=\sum_i \psi_g(v_i)$，预排序一次之后每基因只需 400 次二分查找 → **快 41 倍**。

4. **显著性与方向是两个独立旋钮** —— 固定平均值、只改组内分布形状，可以让同一个基因
   从「铁定变了」变成「没变」，甚至让平均值向上而检验读作向下。

5. **于是 Stage 2 是解析可解的无损解码器**。剩下的全部工作量在 Stage 1。

6. **可检出性完全由输入决定** —— $R_p = (\text{生物学}) \cap (\text{可检出性})$，后者用 BH 有效阈值
   （$z=3.184$，不是 1.96）算出来，跨基因范围 **6.8×**。
   ⚠️ **但它赢不了比赛**：理论确认只按它排序的上限是 $h=0.130$，而追平榜首需要 $h\ge0.134$。
   它是边际修正项，不是策略。

8. **一切走一个数** —— $h$ = 能恢复的参考响应基因比例。`fid` 被结构性耦合成
   $0.5+(a_{\text{true}}-0.5)h$，没有绕过 $h$ 的捷径。物理上限 $h_{\text{replicate}}=0.570$
   （同系重做实验也只重叠 57%）。**唯一的 go/no-go：跨细胞系迁移能否达到 replicate 的 23%？**
   这个问题用公开数据就能答 —— [E07](experiments/E07-source-coverage/)。

7. **占总分三分之二的四个 DE 指标，全场 315 队现在是空的** —— 不是模型不够大，
   是一个「预测什么都没变」的提交会被判 **99.3%** 的基因「变了」（解码器故障）。

---

## 声明

本仓库不隶属于 Arc Institute。所有官方数值引自
[virtualcellchallenge.org](https://virtualcellchallenge.org/) 与
[`ArcInstitute/cell-eval2`](https://github.com/ArcInstitute/cell-eval2) 的 vcc2026 metric specification。
**本仓库不包含任何挑战数据集文件。** 标注「本机实测」的数值均可用 `experiments/` 下的脚本复现
（Apple M1 Pro / 16 GB / 无 CUDA，2026-08-27）。
