# 04 · 自有框架评估

问题：手上四套自有框架，对 Stage 1 有没有可借鉴应用的地方？

四路并行只读审计（2026-08-27）。结论：**三套基本不适用于「预测什么」，
但能借的部分全部落在同一个地方 —— 「你的预测该有多确信、该排在第几位」。**

而这不是边角料：`reach` 这个指标的定义**就是**「按你自己的置信度排序，方向能保持可靠到多深」，
而 `fid`/`jac` 的最优阈值本身就是校准决策。

---

## 判决表

| 框架 | 路径 | 判决 | 理由 |
|---|---|---|---|
| **AdaptiveEROP / BRACE** | `dssi-decsci-assay-prediction` | ✅ **值得投入** | 三处**精确同构**，公式可直接搬 |
| bnode（Bayesian NODE） | `dssi-decsci-bnode` | 🟡 部分可借 | 只借 UQ/校准公式，不借代码 |
| ANM core | `ANM/` | 🟡 部分可借（窄而真） | 三样具体东西 |
| ANM domain adaptation template | `ANM/scripts/` | ❌ 不适用 | **撞词** |
| ANM_application | `ANM_application/` | ❌ 不适用 | 22 维手设权重，差 3–4 个数量级 |
| SciML_Modeling | `dssi-dyve-SciML_Modeling` | ❌ 不适用 | 7 状态生物反应器 ODE |
| alphagenome | `alphagenome/` | ❌ 不适用 | cis 顺式 1 Mb 窗口，给不出 trans 响应 |

---

## ✅ AdaptiveEROP / BRACE —— 真金

ADMET 试验优先级排序框架：$p = 36$ 个 assay 的残差协方差 + 分层贝叶斯 + 阈值 probit。
与 Stage 1 有**三处精确同构**。

### 同构 ① 阈值 probit ↔ 可检出性

`src/Core/p_success.jl` 的 `compute_p_success_cdf`：

$$p = \Phi\!\left(\frac{d\,(\tau - \tilde\mu)}{\tilde\sigma}\right)$$

把 $\tau$ 换成 MDE，双侧化：

$$P(h \in R_p) = \Phi\!\left(\frac{\hat\mu_h - \text{MDE}_{h,c}}{\tilde\sigma_h}\right)
+ \Phi\!\left(\frac{-\text{MDE}_{h,c} - \hat\mu_h}{\tilde\sigma_h}\right)$$

**这个公式在框架里已经写好了。** 我们独立推导 MDE 时是从零推的。约 40 行，重写成 numpy 比跨 Julia 调用便宜。
→ [`src/vcclab/probit.py`](../src/vcclab/probit.py)

### 同构 ② Ledoit–Wolf → ν₀ 恒等式 ↔ 跨细胞系收缩

`src/HierarchicalBayes/inverse_wishart.jl`：

$$\nu_0 = (p+1) + \frac{\alpha^\ast n_c}{1-\alpha^\ast},
\qquad \Psi_0 = (\nu_0 - p - 1)\Sigma \;\Rightarrow\; \mathbb{E}[\Sigma_c] = \Sigma$$

**零超参、维度无关。** 为什么关键：**Stage 1 没有验证集** —— 目标细胞系一条答案都没有，
任何需要调超参的方法都无处可调。$\alpha^\ast$ 由数据自己定，而且自适应：
源细胞系之间差异小 → $\alpha^\ast\to 1$ → 重度池化；差异大 → $\alpha^\ast\to 0$ → 各自为政。

### 同构 ③ mean_shift 退化成 per-gene James–Stein

`src/HierarchicalBayes/mean_shift.jl`：

$$\hat b = (\Sigma_b^{-1} + n_c\Sigma^{-1})^{-1} n_c \Sigma^{-1}\bar r
\;\xrightarrow{\;\Sigma,\Sigma_b\text{ 取对角}\;}\;
\hat b_h = \bar r_h \cdot \frac{n_c\sigma_b^2}{\sigma_h^2 + n_c\sigma_b^2}$$

18,533 个基因 $O(p)$，M1 上毫秒级。这是把「$g$ 在别的细胞系里的实测反应」收缩成先验均值的
**正确**经验贝叶斯形式，严格优于简单平均。
→ [`src/vcclab/shrinkage.py`](../src/vcclab/shrinkage.py)

### 关键的转置技巧

原代码是 $p \lesssim 40$ 的稠密 $O(p^2N)$ EM + 显式 $p\times p$ Cholesky
（注释自己写了「fine for p≲40」）。$p=18{,}533$ 直接不可行 —— $3.4\times10^8$ 项，16 GB 装不下。

**解法：把「维度轴」从基因转成细胞系。** $p = 3\sim8$（源细胞系数），行 = (扰动, 基因) 对。
这样 `conditional_gaussian.jl` 的 $\Sigma_{MO}\Sigma_{OO}^{-1}$ 变成 8×8 分解 —— 免费。

**而且这个转置在语义上更对**：我们要建模的相关性是「同一个效应在不同细胞系之间」，
不是「不同基因之间」。

### 另外两个可借

- `src/Flow/empirical_cdf.jl` + `src/Pipeline/predict.jl` 的**门控规则**：
  只有 $|\hat\mu - \text{MDE}|/\tilde\sigma < 1$ 的基因走非参 MC，其余用解析 $\Phi$，误差 < 2%。
  → 约 2k 个基因需要非参处理，其余 16k 免费。
- `benchmark/realdata/verify_null_fdr.py` —— 置换零分布 + 要求同时超过 null p95 **并**通过 BH-FDR。
  **这正好是官方测量算子（Wilcoxon + BH）的镜像**，可直接搬来自检 MDE 复刻。
- `benchmark/realdata/verify_skipkill_matched_pr.py` —— matched operating point 下的
  recall@matched-precision / PR-AUC / block bootstrap CI。
  **Stage 1 的候选集选择就是这个形状**：从 18,533 里选 ~288，base rate 1–2%。

### ⚠️ 诚实的警告

它自己的 ablation 显示：**$n_c$ 小 / 无组效应时，分层层反而变差** —— 而这正是 Stage 1 的 regime。
必须真的测池化 vs 不池化。好消息是 $\alpha^\ast$ 会自动处理。

---

## 🟡 bnode —— 救我们一个会发出去的 bug

核心文件 `docs/MATH_CHANGES.md`（1037 行数学变更文档）。**只借公式，不借代码** ——
那些数学对应的代码在未 checkout 的 `cleanup/DSSI-1693-multiIC` 分支；
而且分层贝叶斯 / partial pooling 在仓库里**一处都没实现**（grep 零命中）。

### 最重要一条：$n_{\text{eff}}$

> **D4 Wilson 区间：$n_{\text{eff}}$ = 独立轨迹数，不是 $d\cdot n_t$，因为时间点强自相关。**

映射：$n_{\text{eff}}$ = **独立源细胞系数**（3~8），**不是 18,533 个基因**。
基因之间强相关，按基因数算会让置信区间假窄约 $\sqrt{18533/5} \approx 61$ 倍。**我们本来会踩这个。**

### 其余可借

| 机制 | 公式 | 映射到 Stage 1 |
|---|---|---|
| **C1 lag-1 噪声** | $\hat\sigma^2_{\text{lag1}} = \text{median}((\Delta y)^2)/(2\times0.4549364)$ | 沿「同一基因在多个源细胞系」取差分 → 不依赖模型误配的噪声下界 |
| **C1 misfit_ratio** | $\sqrt{\hat\sigma^2_{\text{SSE}}/\hat\sigma^2_{\text{lag1}}}$ | 区分「区间宽因为跨系真差异大（诚实）」vs「因为均值预测错了（假覆盖）」 |
| **D3 shape_quality** | $(\text{corr},\ \text{nrmse},\ \text{band\_rel})$ | `corr` 尺度无关 —— 正是只关心分类+排序时该用的指标族 |
| **D6 OOB envelope** | held-out 落在 train 区间外的比例，>25% 降级 | 该基因在目标系的基础表达是否落在源系表达包络内？包络外 → **置信度降级**。**`reach` 排序键最强的可借项** |
| **A1 边缘化 $\sigma_j^2$** | $U(\theta)=\sum_j \frac{N_j}{2}\log(\text{SSE}_j + N_j\sigma^2_{\text{floor}}) + \frac{\|\theta\|^2}{2\sigma^2_{\text{prior}}}$ | 每基因一个自己的 $\sigma^2_j$ 被积掉 → 对应 MDE 跨基因一个数量级的动态范围，大 MDE 基因不该被小 MDE 基因的似然压倒 |
| **B4 tail-aware scale** | $\text{tail\_index}=\text{std}/(1.4826\,\text{MAD})$；>3 改用 $(p_{98}-p_{02})/(2\times2.0537489)$ | 单细胞计数重尾，这个开关比裸 MAD 稳健 |

**不适用**：所有绑 ODE / 时间轴的部分（collocation 需要 $d\hat u/dt$，multiple-shooting 需要时间窗口，
`detect_regime` 的 stiffness_ratio / spike_recur / n_periods 全定义在时间差分上）。Stage 1 无时间轴。

---

## 🟡 ANM core —— 窄而真

ANM 是「有限窗口响应诊断理论」：$\delta u_t \to \delta z_T \to \delta O_T$。
**前提是能对系统施加扰动、重跑、并读到 endpoint —— Stage 1 恰好禁止这一点。**
而且整个仓库**没有任何学习/估计机制**（无 loss、无拟合、无 OT、无经验贝叶斯、无低秩）。

⚠️ **一个需要纠正的猜测**：「Active」指 **active matter**（Vicsek 自驱粒子 + 反应扩散趋化），
**不是主动实验设计**。所以每天 2 次的提交额度**不能**被 ANM 有意义地建模成 query budget。

### 可借 ① 嵌套观测量 —— 这个映射非常准

ANM 的两级读出满足 $0 \le Q_f \le P_f \le 1$（由集合包含推出）：

| ANM | Stage 1 | 指标 |
|---|---|---|
| 外层 $P_f$ = workability | 这个基因**是否响应** | `jac`（集合重叠） |
| 内层 $Q_f$ = exactness | 响应的**方向是否对** | `fid`（集合内的方向正确率） |

而 $Q \subseteq P$ 的嵌套关系正好是评分结构（`fid` 的分子只在你报的集合内数）。

**结论：应该用两个独立校准的阈值，而不是一个分数。**

### 可借 ② half-support 归一化 → 让 K 自适应

`benchmarks/cross_family/compare_ai_physical_local_response.py`，纯 Python < 60 行：

$$S = Q/Q_0 \quad\text{vs}\quad \lambda = \frac{b_{1/2}-b}{\ell},\qquad \ell = \frac{Q(b_0)}{|dQ/db|}$$

把不同 family 的响应曲线塌缩到同一条归一形式。映射：把报数 $K$ 从常数（288）变成
由可计算的 MDE 分布推出的、**随细胞系变化**的量。
`src/universality/response_scaling.jl` 的 `_response_scaling_exclusion_reasons`
（「什么时候不许做跨域塌缩」的硬性准入清单）可以直接照抄。

### 可借 ③ 防泄漏审计纪律 —— 必须搬

先封印 source hash 再读 endpoint；失败留在分母；leave-one-family-out + delete-one-point
jackknife + bootstrap pass-fraction。

→ **leave-one-cell-line-out** 交叉验证 + **delete-one-gene** jackknife。
给定我们只有 2 次/天提交、且决赛轮是**不同细胞系 + 不同基因 panel**，
这套纪律是防止「validation 轮过拟合、决赛轮归零」的唯一保障。

---

## ❌ 明确不适用的，及证据

**ANM domain adaptation template** —— **撞词**。它的 "domain adaptation" 指
「把 ANM 框架适配到一个新问题领域（写一份 JSON spec）」。两组正交 grep 确认：
`optimal_transport|sinkhorn|wasserstein|MMD|CORAL|importance_weight|covariate_shift|reweight`
全仓库**零算法命中**（所有 `adversarial` 命中都是「造谣智能体」和「红队评审」）。
真实算法是「时间折扣有符号证据求和 + argmax」，动作集 2–4 个离散选项。

**ANM_application** —— Julia 免疫肿瘤边界引擎：输入 ≤22 个手工命名的免疫学坐标
（clamp 到 [0,1]），输出 1 个标量 + 3 类标签，权重阈值全硬编码、零拟合。
与我们的 $(18533\times18400) \to 18533$ 维有符号稀疏向量差 3–4 个数量级，
且坐标是免疫学名词不是基因。唯一可借：`Uncertainty.jl` 的
「不确定度 → 决策翻转概率 → 排序键」骨架，同构于我们的「每基因不确定度 → $P(\text{跨过 MDE})$ → top-K」
—— 但它的 sd 公式是凭空启发式，而我们有闭式解，$18533\times900\times1000$ 次 MC 纯浪费。

**SciML_Modeling** —— `GARNET/example1.jl` 是 7 状态 MTK 生物反应器 ODE（Monod 动力学）；
`Bayesian_UDE/BUDE_LV.jl` 是标准 SciML UDE 教程（$u' = \text{known}(u) + NN(u)$）。
唯一可借的是**残差分解的思想**。`src/core/hmc_sampler.jl` 是 AdvancedHMC NUTS 的 60 行包装，
对 900 组 × M1 Pro 16 GB 完全不可行。

**alphagenome** —— DeepMind 官方客户端库的**未修改克隆**（Apache-2.0，无权重，纯 gRPC 远程 API）。
它预测的是 **cis 顺式、1 Mb 窗口内**的调控效应（DNA 序列 → RNA-seq/ATAC track）。
CRISPRi 敲低的下游响应基因是 **trans 反式/通路级**的 —— **结构上给不出**。

---

## ⚠️ 一个重要的空白

**四个 repo 里没有任何共表达网络、STRING、Reactome、GO、基因模块划分**
可以立刻当「候选响应基因集」先验。唯一现成的生物学资源是 AlphaGenome 教程引用的一个远程
gencode 注释文件（`gencode.v46.annotation.gtf.gz.feather`），只能做基因名↔Ensembl ID 映射、
TSS 坐标、protein_coding 过滤 —— 是数据清洗工具，不是先验。

**所以候选响应基因先验得自己造。** 好消息是材料现成：
**目标细胞系那 18,400 个未扰动细胞的基因×基因相关矩阵** ——
细胞系特异的、免费的，而且正是迁移最难补的那块。见 [E06](../experiments/E06-coexpression-sign/)。

---

## 一句话

**AdaptiveEROP 是真金**（阈值 probit + 零超参收缩 = Stage 1 的核心估计器，只需转置维度轴）。
**bnode 和 ANM 贡献的是校准与审计** —— 在别的比赛里是边角料，但这里 `reach` 直接给它打分，
而且是防止 validation 过拟合的唯一手段。**ANM 的域适配模板和 alphagenome 是死路，不要投入。**
