import type { L, LN } from "../i18n";

export type Verdict = "worth-investing" | "partial" | "narrow" | "not-applicable";

export type Borrowable = {
  title: L;
  formula?: string;
  file: string;
  mapping: L;
};

export type Framework = {
  id: string;
  name: string;
  repo: string;
  what: L;
  verdict: Verdict;
  summary: L;
  borrowables: Borrowable[];
  notApplicable: { reason: L; evidence: string }[];
};

export const VERDICT_LABEL: Record<Verdict, L> = {
  "worth-investing": { zh: "值得投入", en: "Worth investing" },
  partial: { zh: "部分可借", en: "Partly borrowable" },
  narrow: { zh: "窄而真", en: "Narrow but real" },
  "not-applicable": { zh: "不适用", en: "Not applicable" },
};

export const FRAMEWORKS: Framework[] = [
  {
    id: "erop",
    name: "AdaptiveEROP / BRACE",
    repo: "dssi-decsci-assay-prediction",
    what: {
      zh: "ADMET 试验优先级排序：36 个 assay 的残差协方差 + 分层贝叶斯 + 阈值 probit",
      en: "ADMET assay prioritisation: residual covariance over 36 assays + hierarchical Bayes + threshold probit",
    },
    verdict: "worth-investing",
    summary: {
      zh: "三处精确同构。它的阈值 probit 就是我们独立推出的可检出性公式；Ledoit–Wolf→ν₀ 恒等式零超参、维度无关，而 Stage 1 没有验证集，这一点是决定性的。唯一障碍是维度：原代码 p≲40，需把维度轴从基因转置成细胞系。",
      en: "Three exact isomorphisms. Its threshold probit is the detectability formula we derived independently; the Ledoit–Wolf→ν₀ identity is hyperparameter-free and dimension-independent, which is decisive because Stage 1 has no validation set. The only obstacle is dimension: the original code assumes p≲40, so the axis must be transposed from genes to cell lines.",
    },
    borrowables: [
      {
        title: { zh: "阈值 probit ↔ 可检出性", en: "Threshold probit ↔ detectability" },
        formula: String.raw`P(h\in R_p)=\Phi\!\left(\tfrac{\hat\mu_h-\mathrm{MDE}_{h,c}}{\tilde\sigma_h}\right)+\Phi\!\left(\tfrac{-\mathrm{MDE}_{h,c}-\hat\mu_h}{\tilde\sigma_h}\right)`,
        file: "src/Core/p_success.jl · compute_p_success_cdf",
        mapping: {
          zh: "把 assay 的成功阈值 τ 换成该 (基因, 细胞系) 的 MDE，双侧化。约 40 行，重写成 numpy 比跨 Julia 调用便宜。",
          en: "Swap the assay's success threshold τ for that (gene, cell line)'s MDE and make it two-sided. About 40 lines; rewriting in numpy beats calling across into Julia.",
        },
      },
      {
        title: { zh: "Ledoit–Wolf → ν₀：零超参收缩", en: "Ledoit–Wolf → ν₀: hyperparameter-free shrinkage" },
        formula: String.raw`\begin{aligned}\nu_0&=(p+1)+\frac{\alpha^\ast n_c}{1-\alpha^\ast}\\ \Psi_0&=(\nu_0-p-1)\Sigma\ \Rightarrow\ \mathbb{E}[\Sigma_c]=\Sigma\end{aligned}`,
        file: "src/HierarchicalBayes/inverse_wishart.jl",
        mapping: {
          zh: "α* 由数据自己定，而且自适应：源细胞系差异小 → α*→1 → 重度池化；差异大 → α*→0 → 各自为政。Stage 1 没有验证集，无处调超参，所以这一条是决定性的。",
          en: "α* is estimated from the data and self-adapts: little variation across source lines → α*→1 → heavy pooling; large variation → α*→0 → no pooling. Stage 1 has no validation set and nowhere to tune, which makes this decisive.",
        },
      },
      {
        title: { zh: "mean_shift 退化成 per-gene James–Stein", en: "mean_shift collapses to per-gene James–Stein" },
        formula: String.raw`\hat b_h=\bar r_h\cdot\frac{n_c\sigma_b^2}{\sigma_h^2+n_c\sigma_b^2}`,
        file: "src/HierarchicalBayes/mean_shift.jl",
        mapping: {
          zh: "Σ、Σ_b 取对角即退化，18,533 个基因 O(p)，M1 上毫秒级。这是把「g 在别的细胞系里的实测反应」收缩成先验均值的正确经验贝叶斯形式，严格优于简单平均。",
          en: "Taking Σ and Σ_b diagonal collapses it to O(p) over 18,533 genes — milliseconds on an M1. This is the correct empirical-Bayes form for shrinking \"g's measured response in other cell lines\" into a prior mean, and strictly beats a plain average.",
        },
      },
      {
        title: { zh: "非参门控：只算需要算的", en: "Non-parametric gating: compute only what matters" },
        formula: String.raw`\left|\hat\mu-\mathrm{MDE}\right|/\tilde\sigma<1`,
        file: "src/Flow/empirical_cdf.jl · src/Pipeline/predict.jl",
        mapping: {
          zh: "只有阈值附近的约 2k 个基因需要经验 CDF 的 MC，其余 16k 用解析 Φ，误差 < 2%。",
          en: "Only the ~2k genes near the threshold need empirical-CDF Monte Carlo; the other 16k use the analytic Φ with under 2% error.",
        },
      },
      {
        title: { zh: "置换零分布 + BH：官方测量算子的镜像", en: "Permutation null + BH: a mirror of the official operator" },
        file: "benchmark/realdata/verify_null_fdr.py",
        mapping: {
          zh: "要求观测值同时超过 null p95 并通过 BH-FDR，而不是裸 p<0.05。这正好是官方打分器（Wilcoxon 秩和 + BH）的镜像，可直接搬来自检 MDE 复刻。",
          en: "Requires the observation to clear both the null p95 and BH-FDR, not a bare p<0.05. That is exactly a mirror of the official scorer (Wilcoxon rank-sum + BH), so it transplants directly into self-checking our MDE reimplementation.",
        },
      },
      {
        title: { zh: "matched operating point 评测", en: "Matched-operating-point evaluation" },
        file: "benchmark/realdata/verify_skipkill_matched_pr.py",
        mapping: {
          zh: "不看绝对 precision，看 recall@matched-precision / PR-AUC / block bootstrap CI。Stage 1 的候选集选择正是这个形状：从 18,533 里选 ~288，base rate 1–2%。",
          en: "Ignores absolute precision in favour of recall@matched-precision / PR-AUC / block bootstrap CI. Stage 1's candidate selection has exactly that shape: pick ~288 out of 18,533 at a 1–2% base rate.",
        },
      },
    ],
    notApplicable: [
      {
        reason: {
          zh: "稠密 O(p²N) EM + 显式 p×p Cholesky 在 p=18,533 不可行（3.4×10⁸ 项，16 GB 装不下）。必须把维度轴从基因转置成细胞系（p=3~8），行改为 (扰动, 基因) 对。",
          en: "The dense O(p²N) EM with an explicit p×p Cholesky is infeasible at p=18,533 (3.4×10⁸ entries; won't fit in 16 GB). The dimension axis must be transposed from genes to cell lines (p=3–8), with (perturbation, gene) as rows.",
        },
        evidence: "src/Core/em_covariance.jl · 注释自述「fine for p≲40」",
      },
      {
        reason: {
          zh: "它自己的 ablation 显示：n_c 小 / 无组效应时分层层反而变差 —— 而这正是 Stage 1 的 regime（n_c = 3~8）。必须实测池化 vs 不池化。",
          en: "Its own ablation shows the hierarchical layer hurts when n_c is small or there is no group effect — precisely Stage 1's regime (n_c = 3–8). Pooling vs no pooling must be measured, not assumed.",
        },
        evidence: "repo 自带 ablation 结果",
      },
    ],
  },
  {
    id: "bnode",
    name: "bnode",
    repo: "dssi-decsci-bnode",
    what: {
      zh: "贝叶斯 Neural ODE：时间连续动力系统的向量场后验推断",
      en: "Bayesian Neural ODE: posterior inference over vector fields of time-continuous dynamical systems",
    },
    verdict: "partial",
    summary: {
      zh: "数学对象根本不匹配（Stage 1 无时间轴、无 ODE）。但它的校准诚实性机制是领域无关的，而且救了我们一个会发出去的 bug：n_eff 必须是独立源细胞系数，不是 18,533 个基因，否则置信区间假窄约 61 倍。只借公式，不借代码。",
      en: "The mathematical objects simply do not match — Stage 1 has no time axis and no ODE. But its calibration-honesty machinery is domain-independent, and it saved us from a bug we would have shipped: n_eff must be the number of independent source cell lines, not 18,533 genes, or the confidence interval is ~61× too narrow. Borrow the formulas, not the code.",
    },
    borrowables: [
      {
        title: { zh: "n_eff 的正确取法（最重要）", en: "Getting n_eff right (the big one)" },
        formula: String.raw`n_{\text{eff}}=\#\{\text{独立源细胞系}\}\;\ne\;18{,}533`,
        file: "docs/MATH_CHANGES.md · D4",
        mapping: {
          zh: "原文：n_eff = 独立轨迹数，不是 d·n_t，因为时间点强自相关。映射：基因之间强相关，按基因数算会让置信区间假窄约 √(18533/5) ≈ 61 倍。",
          en: "Original: n_eff is the number of independent trajectories, not d·n_t, because time points are strongly autocorrelated. Mapping: genes are strongly correlated, so counting genes narrows the interval by roughly √(18533/5) ≈ 61×.",
        },
      },
      {
        title: { zh: "misfit_ratio：区分诚实的宽与假覆盖", en: "misfit_ratio: honest width vs fake coverage" },
        formula: String.raw`\mathrm{misfit}=\sqrt{\hat\sigma^2_{\mathrm{SSE}}/\hat\sigma^2_{\mathrm{lag1}}}`,
        file: "docs/MATH_CHANGES.md · C1",
        mapping: {
          zh: "区分「置信区间宽是因为跨细胞系真的差异大（诚实）」还是「因为均值预测错了（假覆盖）」。给每个 (基因, 细胞系) 打「方向预测可信不可信」标签。",
          en: "Separates \"the interval is wide because variation across cell lines is genuinely large (honest)\" from \"because the mean prediction is wrong (fake coverage)\". Tags each (gene, cell line) with whether its direction prediction can be trusted.",
        },
      },
      {
        title: { zh: "OOB envelope：reach 排序键最强的可借项", en: "OOB envelope: the strongest borrowable for reach" },
        file: "docs/MATH_CHANGES.md · D6",
        mapping: {
          zh: "该基因在目标细胞系的基础表达是否落在源细胞系表达包络内？包络外 → 置信度降级、排序后置。reach 的定义就是「按你自己的置信度排序，方向能保持可靠到多深」。",
          en: "Is this gene's basal expression in the target line inside the envelope observed across source lines? Outside → downgrade confidence and push it down the ranking. reach is defined as exactly \"how deep your own confidence ordering stays reliable\".",
        },
      },
      {
        title: { zh: "边缘化 per-gene σ²", en: "Marginalise per-gene σ²" },
        formula: String.raw`U(\theta)=\sum_j \tfrac{N_j}{2}\log\!\left(\mathrm{SSE}_j+N_j\sigma^2_{\text{fl}}\right)+\tfrac{\lVert\theta\rVert^2}{2\sigma^2_{\text{pr}}}`,
        file: "docs/MATH_CHANGES.md · A1",
        mapping: {
          zh: "每个基因一个自己的 σ²_j 被解析积掉。正好对应 MDE 跨基因一个数量级的动态范围 —— 大 MDE 基因不该被小 MDE 基因的似然压倒。",
          en: "Each gene's own σ²_j is integrated out analytically. This matches the order-of-magnitude MDE dynamic range across genes: high-MDE genes should not be drowned out by low-MDE genes in the likelihood.",
        },
      },
      {
        title: { zh: "tail-aware 尺度", en: "Tail-aware scale" },
        formula: String.raw`\text{tail}=\tfrac{\mathrm{std}}{1.4826\,\mathrm{MAD}};\quad >3\Rightarrow \tfrac{p_{98}-p_{02}}{2\times2.0537489}`,
        file: "docs/MATH_CHANGES.md · B4",
        mapping: {
          zh: "单细胞计数与 LFC 残差都重尾，这个开关比裸 MAD 稳健得多。",
          en: "Single-cell counts and LFC residuals are both heavy-tailed; this switch is far more robust than a bare MAD.",
        },
      },
    ],
    notApplicable: [
      {
        reason: {
          zh: "所有绑 ODE / 时间轴的部分：collocation 需要 dû/dt，multiple-shooting 需要时间窗口，detect_regime 的 stiffness_ratio / spike_recur / n_periods 全部定义在时间差分上。Stage 1 无时间轴，这些量根本没有定义。",
          en: "Everything tied to ODEs or a time axis: collocation needs dû/dt, multiple shooting needs time windows, and detect_regime's stiffness_ratio / spike_recur / n_periods are all defined on time differences. Stage 1 has no time axis, so these quantities are simply undefined.",
        },
        evidence: "A3/A4/B1/B2/E1/E2/F1/F2/G1/G2 全部",
      },
      {
        reason: {
          zh: "分层贝叶斯 / partial pooling 在仓库里一处都没实现（grep hierarch|partial.?pool|random effect|multilevel 在 src/ 零命中）。想要的 (θ_shared, {δ_g,c}) 结构必须自己写。",
          en: "Hierarchical Bayes / partial pooling is implemented nowhere in the repo (grep for hierarch|partial.?pool|random effect|multilevel returns nothing under src/). The desired (θ_shared, {δ_g,c}) structure has to be written from scratch.",
        },
        evidence: "grep 零命中；TYPES.md 的两个类型只是形状包装",
      },
      {
        reason: {
          zh: "那些数学对应的代码不在当前 worktree —— 只存在于未 checkout 的 cleanup/DSSI-1693-multiIC 分支。",
          en: "The code implementing that maths is not in the current worktree — it exists only on the unchecked-out cleanup/DSSI-1693-multiIC branch.",
        },
        evidence: "HEAD = refs/heads/main = Brunner baseline",
      },
    ],
  },
  {
    id: "anm",
    name: "ANM core",
    repo: "ANM/",
    what: {
      zh: "有限窗口响应诊断理论：δu → δz → δO，配嵌套读出 (P_f, Q_f)",
      en: "Finite-window response diagnostics: δu → δz → δO, with nested readouts (P_f, Q_f)",
    },
    verdict: "narrow",
    summary: {
      zh: "核心机制的前提是「能对系统施加扰动、重跑、并读到 endpoint」—— Stage 1 恰好禁止这一点（目标细胞系零标签）。仓库里没有任何学习/迁移/估计机制。但有三样东西窄而真地可借，其中嵌套观测量的映射非常准。",
      en: "The core mechanism presumes you can perturb the system, re-run it, and read the endpoint — which Stage 1 forbids outright (zero labels in the target cell line). The repo contains no learning, transfer, or estimation machinery at all. But three things transfer, narrowly and genuinely, and the nested-observable mapping is remarkably apt.",
    },
    borrowables: [
      {
        title: { zh: "嵌套观测量 → 两个阈值而非一个分数", en: "Nested observables → two thresholds, not one score" },
        formula: String.raw`0\le Q_f\le P_f\le 1`,
        file: "src/universality/observables.jl · paper/arxiv/manuscript.tex",
        mapping: {
          zh: "外层 P_f（workability）↔ 这个基因是否响应 ↔ jac 的集合；内层 Q_f（exactness）↔ 方向是否对 ↔ fid 的正确率。而 Q ⊆ P 的嵌套正好是评分结构（fid 的分子只在你报的集合内数）。结论：用两个独立校准的阈值。",
          en: "Outer P_f (workability) ↔ does this gene respond ↔ jac's set; inner Q_f (exactness) ↔ is the direction right ↔ fid's accuracy. And the nesting Q ⊆ P is exactly the scoring structure (fid's numerator is counted only inside the set you reported). Conclusion: use two independently calibrated thresholds.",
        },
      },
      {
        title: { zh: "half-support 归一化 → 自适应报数 K", en: "Half-support normalisation → adaptive call-set size K" },
        formula: String.raw`S=Q/Q_0\ \ \text{vs}\ \ \lambda=\frac{b_{1/2}-b}{\ell},\quad \ell=\frac{Q(b_0)}{\lvert dQ/db\rvert}`,
        file: "benchmarks/cross_family/compare_ai_physical_local_response.py（< 60 行纯 Python）",
        mapping: {
          zh: "把不同 family 的响应曲线塌缩到同一条归一形式。映射：把报数 K 从常数（288）变成由可计算的 MDE 分布推出的、随细胞系变化的量。",
          en: "Collapses response curves from different families onto one normalised form. Mapping: turn the call-set size K from a constant (288) into a quantity derived from the computable MDE distribution and varying per cell line.",
        },
      },
      {
        title: { zh: "防泄漏审计纪律 —— 必须搬", en: "Anti-leakage audit discipline — must transplant" },
        file: "src/universality/response_scaling.jl · _response_scaling_exclusion_reasons",
        mapping: {
          zh: "先封印 source hash 再读 endpoint；失败留在分母；leave-one-family-out + delete-one-point jackknife。→ leave-one-cell-line-out 交叉验证。给定 2 次/天额度、且决赛轮是不同细胞系 + 不同 panel，这是防止 validation 过拟合的唯一保障。",
          en: "Seal the source hash before reading the endpoint; keep failures in the denominator; leave-one-family-out plus delete-one-point jackknife. → leave-one-cell-line-out cross-validation. Given two submissions a day and a final round on different cell lines with a different panel, this is the only guard against overfitting validation.",
        },
      },
    ],
    notApplicable: [
      {
        reason: {
          zh: "「Active」指 active matter（Vicsek 自驱粒子 + 反应扩散趋化），不是主动实验设计。所以每天 2 次的提交额度不能被 ANM 有意义地建模成 query budget —— 这一点纠正了我们原本的猜测。",
          en: "\"Active\" means active matter (Vicsek self-propelled particles plus reaction–diffusion chemotaxis), not active experimental design. So the two-submissions-a-day budget cannot be meaningfully modelled by ANM as a query budget — which corrects our original guess.",
        },
        evidence: "仓库内唯一的 active allocator 是某 CIViC benchmark 内一条手调常数的贪心打分",
      },
      {
        reason: {
          zh: "整个仓库没有任何学习/迁移/估计机制：无 loss、无拟合、无最优传输、无经验贝叶斯、无低秩估计；wilcoxon / BH / scanpy / anndata 全仓库零命中。",
          en: "The repo contains no learning, transfer, or estimation machinery at all: no loss, no fitting, no optimal transport, no empirical Bayes, no low-rank estimation; wilcoxon / BH / scanpy / anndata return nothing repo-wide.",
        },
        evidence: "全仓库 grep 确认",
      },
    ],
  },
  {
    id: "dead-ends",
    name: "ANM DA template · ANM_application · SciML_Modeling · alphagenome",
    repo: "四处",
    what: {
      zh: "分别是：域适配模板、免疫肿瘤边界引擎、生物反应器 ODE、DeepMind 基因组客户端",
      en: "Respectively: a domain-adaptation template, an immuno-oncology boundary engine, a bioreactor ODE, and DeepMind's genomics client",
    },
    verdict: "not-applicable",
    summary: {
      zh: "四处都是死路，不要投入。第一处是撞词；其余三处在数学对象或空间尺度上都对不上。",
      en: "All four are dead ends; do not invest. The first is a naming collision; the other three simply do not match on mathematical object or spatial scale.",
    },
    borrowables: [
      {
        title: { zh: "唯一残值：翻转概率排序骨架", en: "The one residual: flip-probability ranking" },
        file: "ANM_application · Uncertainty.jl",
        mapping: {
          zh: "「不确定度 → 决策翻转概率 → 排序键」同构于我们的「每基因不确定度 → P(跨过 MDE) → top-K」。但它的 sd 公式是凭空启发式，而我们有闭式解，18533×900×1000 次 MC 纯浪费。只值一段公式改写，不值一次代码集成。",
          en: "\"Uncertainty → decision flip probability → ranking key\" is isomorphic to our \"per-gene uncertainty → P(crossing MDE) → top-K\". But its sd formula is an ad-hoc heuristic while we have a closed form, so 18533×900×1000 Monte Carlo draws would be pure waste. Worth one formula rewrite, not a code integration.",
        },
      },
    ],
    notApplicable: [
      {
        reason: {
          zh: "ANM domain adaptation template 是撞词：它的 “domain adaptation” 指「把 ANM 适配到一个新问题领域（写一份 JSON spec）」。真实算法是「时间折扣有符号证据求和 + argmax」，动作集 2–4 个离散选项。",
          en: "The ANM domain-adaptation template is a naming collision: its \"domain adaptation\" means adapting ANM to a new problem domain by writing a JSON spec. The actual algorithm is time-discounted signed evidence summation plus argmax over 2–4 discrete actions.",
        },
        evidence: "两组正交 grep：optimal_transport|sinkhorn|wasserstein|MMD|CORAL|importance_weight|covariate_shift|reweight 全仓库零算法命中",
      },
      {
        reason: {
          zh: "ANM_application 是 Julia 免疫肿瘤边界引擎：输入 ≤22 个手工命名的免疫学坐标，输出 1 个标量 + 3 类标签，权重阈值全硬编码、零拟合。与我们相差 3–4 个数量级，且坐标是免疫学名词不是基因。",
          en: "ANM_application is a Julia immuno-oncology boundary engine: at most 22 hand-named immunology coordinates in, one scalar plus three labels out, all weights and thresholds hard-coded, zero fitting. Three to four orders of magnitude away from our problem, and its coordinates are immunology terms, not genes.",
        },
        evidence: "权重 0.13/0.15/−0.12…，阈值 0.65/0.35/0.62/0.42/0.45 全硬编码",
      },
      {
        reason: {
          zh: "SciML_Modeling 是 7 状态 MTK 生物反应器 ODE（Monod 动力学）+ 标准 SciML UDE 教程。Stage 1 无时间轴用不上；HMC 包装对 900 组 × 16 GB 完全不可行。",
          en: "SciML_Modeling is a seven-state MTK bioreactor ODE (Monod kinetics) plus a standard SciML UDE tutorial. Stage 1 has no time axis, and the HMC wrapper is wholly infeasible for 900 groups in 16 GB.",
        },
        evidence: "GARNET/example1.jl · Bayesian_UDE/BUDE_LV.jl · src/core/hmc_sampler.jl",
      },
      {
        reason: {
          zh: "alphagenome 是 DeepMind 官方客户端库的未修改克隆（无权重，纯 gRPC 远程 API）。它预测 cis 顺式、1 Mb 窗口内的调控效应；CRISPRi 敲低的下游响应基因是 trans 反式/通路级的 —— 结构上给不出。",
          en: "alphagenome is an unmodified clone of DeepMind's official client library (no weights, pure remote gRPC). It predicts cis regulatory effects within a 1 Mb window; the downstream responders of a CRISPRi knockdown are trans, pathway-level — structurally out of reach.",
        },
        evidence: "Apache-2.0，1 天前拉取，无用户改动",
      },
      {
        reason: {
          zh: "关键空白：四个 repo 里没有任何共表达网络、STRING、Reactome、GO、基因模块划分可以立刻当「候选响应基因集」先验。唯一现成生物学资源是一个 gencode 注释文件，只能做基因名↔Ensembl ID 映射。",
          en: "A critical gap: none of the four repos contains a co-expression network, STRING, Reactome, GO, or gene-module partition usable straight away as a candidate-responder prior. The only ready biological resource is a gencode annotation file, good only for gene-symbol↔Ensembl-ID mapping.",
        },
        evidence: "gencode.v46.annotation.gtf.gz.feather（远程 GCS URL）",
      },
    ],
  },
];

export const UNIFYING: LN = {
  zh: (
    <>
      三套基本不适用于「预测什么」，但能借的部分<b>全部落在同一个地方</b> ——
      「你的预测该有多确信、该排在第几位」。而这不是边角料：
      <code>reach</code> 这个指标的定义<b>就是</b>「按你自己的置信度排序，方向能保持可靠到多深」，
      而 <code>fid</code>/<code>jac</code> 的最优阈值本身就是校准决策。
    </>
  ),
  en: (
    <>
      Three of them barely help with <em>what</em> to predict — but everything that does transfer
      lands in <b>the same place</b>: how confident you should be, and how to order your own
      predictions. That is not a side concern here. <code>reach</code> is defined as exactly
      &ldquo;how deep your own confidence ordering stays reliable&rdquo;, and the optimal thresholds
      for <code>fid</code> and <code>jac</code> are themselves calibration decisions.
    </>
  ),
};
