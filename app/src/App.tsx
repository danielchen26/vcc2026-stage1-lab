import { useEffect, useRef, useState } from "react";
import { CallsetPanel } from "./components/CallsetPanel";
import { FrameworkMatrix } from "./components/FrameworkMatrix";
import { MDE_META, MDE_QUANTILES, MDE_RANGE, MdeExplorer } from "./components/MdeExplorer";
import { ProbitPanel } from "./components/ProbitPanel";
import { Chip, M, Plain, Section, Stat } from "./components/ui";
import { LangProvider, LangToggle, type Lang } from "./i18n";

const REPO = "https://github.com/danielchen26/vcc2026-stage1-lab";
const DOSSIER = "https://danielchen26.github.io/vcc2026-dossier/";

const IDS = ["problem", "detect", "frameworks", "estimator", "callset", "next"];

const C = {
  zh: {
    sub: "Stage 1 实验室",
    nav: ["剩下的问题", "答案的一半在输入里", "四套框架的判决", "装配好的估计器", "该报多少个基因", "接下来"],
    eyebrow: "ARC INSTITUTE · VIRTUAL CELL CHALLENGE 2026 · STAGE 1",
    title: "答案的一半，就藏在给你的输入里。",
    thesis: (
      <>
        我们已经证明打分器每个基因只读两个数，并把「构造」那一级做成了无损解码器。
        剩下唯一的问题是<b>预测</b>。而预测的目标不是一个生物量 ——
        它是<b>一个特定统计检验作用在一个有限样本上的输出</b>。
        于是「哪些基因会被判变了」= 生物学 ∩ 可检出性，而<b>后者完全由官方发给你的对照细胞决定</b>。
      </>
    ),
    meta: [
      ["检出门槛动态范围 p95/p25", `${MDE_RANGE.toFixed(1)}×`],
      ["全量计算耗时", `${MDE_META.elapsedSec}s`],
      ["报数最优点 vs 解析预测", "287 / 288"],
      ["评估的自有框架", "4 套 · 1 套真金"],
    ],
    secs: {
      problem: { t: "剩下的问题是什么", k: "Stage 2 已经做完。这一节说清 Stage 1 到底要输出什么。" },
      detect: { t: "答案的一半在输入里", k: "这是这一轮最重要的发现，而且可以当场算出来。" },
      frameworks: { t: "四套自有框架的判决", k: "三套基本不适用，但能借的部分落在同一个地方。" },
      estimator: { t: "装配好的估计器", k: "AdaptiveEROP 的阈值 probit + 我们算出的检出门槛。" },
      callset: { t: "该报多少个基因", k: "有解析答案，而且数值确认了。" },
      next: { t: "接下来", k: "三个决定路线的实验，都不需要 GPU。" },
    },
    probLede: (
      <>
        对每个「组」（一个待预测基因 × 一个细胞系，共 900 组），Stage 1 要输出的是
        <b>一个稀疏有符号向量</b>：哪几百个基因响应、各自涨还是跌、各自幅度多大。
        规模是 300 个基因 × 3 个细胞系，训练素材不到 2 GB —— 经典统计的尺寸，不是深度学习的尺寸。
      </>
    ),
    reframe: "两个根本的重构",
    reframe1t: "预测测量结果，不是预测现象",
    reframe1: (
      <>
        所有人都在建模生物学，然后<em>希望</em>统计检验会同意。
        但要预测的东西不是一个生物量，是「一个特定的检验，作用在一个特定的有限样本上，得到的输出」。
        既然我们已经精确复刻了那个检验，就可以把它拆开。
      </>
    ),
    reframe2t: "这是分类 + 排序，不是回归",
    reframe2: (
      <>
        六个指标里<b>四个只需要「哪些基因响应」和「涨跌方向」</b>。
        幅度只进 <code>nmae</code>，而且是归一化的。
        全场都在做回归（预测表达量），而评分要的是分类 —— 后者的样本复杂度低得多。
      </>
    ),
    detectLede: (
      <>
        参考答案 <M tex="R_p" /> 是两个集合的交：
        <b>「生物上真的变了」</b>（未知，要建模）∩ <b>「在这个细胞系里检得出来」</b>
        （<b>官方已经把它交给你了</b>）。第二项跨基因动态范围 {MDE_RANGE.toFixed(1)} 倍（p95/p25），
        而且 9,929 个基因全量算完只要 {MDE_META.elapsedSec} 秒。
      </>
    ),
    qTitle: "全量实测的检出门槛分位数",
    estLede: (
      <>
        把「源细胞系里的实测反应」收缩成先验均值（James–Stein，零超参），
        配上跨源细胞系的不确定度，再过一遍可检出性 probit ——
        就得到每个基因「会被判变了」的校准概率，以及它的方向。
        <b>这三个公式在你自己的 AdaptiveEROP 框架里已经写好了。</b>
      </>
    ),
    nextItems: [
      {
        n: "E05",
        t: "零生物学基线",
        d: "只用可检出性排序、报 ~288 个基因，jac 能到多少？如果落在 0.15 以上，不用等 Stage 1 建模就能上线拿第一。",
      },
      {
        n: "E06",
        t: "共表达符号",
        d: "目标细胞系自己那 18,400 个对照细胞里的共表达，能否预测涨跌方向？这是唯一细胞系特异且免费的符号信号。",
      },
      {
        n: "E07",
        t: "源域覆盖率",
        d: "300 个待预测基因里有多少在 Replogle 全基因组数据里有实测反应？这个数字决定路线：迁移已知响应 vs 预测未知扰动。",
      },
    ],
    gap: "一个必须自己造的东西",
    gapBody: (
      <>
        四个 repo 里<b>没有任何</b>共表达网络、STRING、Reactome、GO、基因模块划分
        可以立刻当「候选响应基因集」先验。唯一现成的生物学资源是一个 gencode 注释文件，
        只能做基因名↔Ensembl ID 映射。
        <br />
        <b>但材料现成</b>：目标细胞系那 18,400 个未扰动细胞的基因×基因相关矩阵 ——
        细胞系特异的、免费的，而且正是迁移最难补的那块。
      </>
    ),
    footLinks: "完整讨论与可复现实验在仓库；问题与打分规则的入门版在 dossier",
  },
  en: {
    sub: "Stage 1 Lab",
    nav: ["What's left", "Half the answer is in the input", "Framework verdicts", "The estimator", "How many to flag", "What's next"],
    eyebrow: "ARC INSTITUTE · VIRTUAL CELL CHALLENGE 2026 · STAGE 1",
    title: "Half the answer is already in the input you were given.",
    thesis: (
      <>
        We have shown the scorer reads only two numbers per gene, and turned the &ldquo;construct&rdquo;
        stage into a lossless decoder. The only thing left is <b>prediction</b> — and the target of that
        prediction is not a biological quantity. It is{" "}
        <b>the output of one specific statistical test applied to one finite sample</b>.
        So &ldquo;which genes get called changed&rdquo; = biology ∩ detectability, and{" "}
        <b>detectability is fixed entirely by the control cells the organisers handed you</b>.
      </>
    ),
    meta: [
      ["Threshold dynamic range p95/p25", `${MDE_RANGE.toFixed(1)}×`],
      ["Full computation", `${MDE_META.elapsedSec}s`],
      ["Optimal K vs analytic prediction", "287 / 288"],
      ["Own frameworks audited", "4 · 1 is gold"],
    ],
    secs: {
      problem: { t: "What is actually left", k: "Stage 2 is done. This section pins down what Stage 1 must output." },
      detect: { t: "Half the answer is in the input", k: "The most important finding of this round — and computable on the spot." },
      frameworks: { t: "Verdicts on four in-house frameworks", k: "Three barely apply, but what transfers lands in one place." },
      estimator: { t: "The assembled estimator", k: "AdaptiveEROP's threshold probit plus the detection threshold we computed." },
      callset: { t: "How many genes to flag", k: "There is a closed-form answer, and it checks out numerically." },
      next: { t: "What's next", k: "Three route-deciding experiments, none needing a GPU." },
    },
    probLede: (
      <>
        For each &ldquo;group&rdquo; (one gene to predict × one cell line; 900 in total), Stage 1 must
        output <b>a sparse signed vector</b>: which few hundred genes respond, which way each moves, and
        by how much. The problem is 300 genes × 3 cell lines with under 2 GB of training material — a
        classical-statistics size, not a deep-learning size.
      </>
    ),
    reframe: "Two fundamental reframings",
    reframe1t: "Predict the measurement, not the phenomenon",
    reframe1: (
      <>
        Everyone models the biology and then <em>hopes</em> the statistical test agrees. But the thing
        being predicted is not a biological quantity — it is the output of one specific test on one
        specific finite sample. Since we have reimplemented that test exactly, we can take it apart.
      </>
    ),
    reframe2t: "This is classification and ranking, not regression",
    reframe2: (
      <>
        <b>Four of the six metrics need only which genes respond and which way.</b> Magnitude enters
        only <code>nmae</code>, and there it is normalised. The field is doing regression (predicting
        expression) while the score asks for classification — which has far better sample complexity.
      </>
    ),
    detectLede: (
      <>
        The reference answer <M tex="R_p" /> is the intersection of two sets:{" "}
        <b>&ldquo;genuinely changed&rdquo;</b> (unknown, needs modelling) ∩{" "}
        <b>&ldquo;detectable in this cell line&rdquo;</b> (<b>already handed to you</b>). The second
        spans a {MDE_RANGE.toFixed(1)}× range across genes (p95/p25), and computing it for all 9,929 genes
        takes {MDE_META.elapsedSec} seconds.
      </>
    ),
    qTitle: "Measured detection-threshold quantiles, full population",
    estLede: (
      <>
        Shrink the measured responses from source cell lines into a prior mean (James–Stein,
        hyperparameter-free), pair it with the spread across those source lines, and push it through
        the detectability probit — out comes a calibrated probability that each gene will be called
        changed, plus its direction.{" "}
        <b>All three formulas were already written in your own AdaptiveEROP framework.</b>
      </>
    ),
    nextItems: [
      {
        n: "E05",
        t: "Zero-biology baseline",
        d: "Rank by detectability alone, flag ~288 genes — what jac does that reach? If it lands above 0.15, we can take first place before any Stage 1 modelling exists.",
      },
      {
        n: "E06",
        t: "Co-expression sign",
        d: "Can co-expression inside the target line's own 18,400 control cells predict direction? It is the only cell-line-specific sign signal that is free.",
      },
      {
        n: "E07",
        t: "Source coverage",
        d: "How many of the 300 target genes have a measured response in Replogle's genome-wide data? That number decides the route: transfer a known response vs predict an unseen perturbation.",
      },
    ],
    gap: "One thing we have to build ourselves",
    gapBody: (
      <>
        <b>None</b> of the four repos contains a co-expression network, STRING, Reactome, GO, or gene
        modules usable straight away as a candidate-responder prior. The only ready biological resource
        is a gencode annotation file, good only for gene-symbol↔Ensembl-ID mapping.
        <br />
        <b>But the material is at hand</b>: the gene-by-gene correlation matrix of the target line's own
        18,400 unperturbed cells — cell-line-specific, free, and exactly the part transfer struggles with.
      </>
    ),
    footLinks: "Full discussion and reproducible experiments live in the repo; the primer on the problem and scoring rules is in the dossier",
  },
};

function useReveal() {
  useEffect(() => {
    const els = document.querySelectorAll<HTMLElement>(".reveal");
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      els.forEach((el) => el.classList.add("in"));
      return;
    }
    const io = new IntersectionObserver(
      (es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }),
      { rootMargin: "0px 0px -8% 0px", threshold: 0.05 },
    );
    els.forEach((el) => io.observe(el));
    return () => io.disconnect();
  }, []);
}

export default function App() {
  return <LangProvider>{(lang, set) => <Page lang={lang} setLang={set} />}</LangProvider>;
}

function Page({ lang, setLang }: { lang: Lang; setLang: (l: Lang) => void }) {
  const c = C[lang];
  const bar = useRef<HTMLDivElement>(null);
  const [active, setActive] = useState(IDS[0]);
  useReveal();

  useEffect(() => {
    let raf = 0;
    const tick = () => {
      raf = 0;
      const max = document.documentElement.scrollHeight - window.innerHeight;
      if (bar.current) bar.current.style.transform = `scaleX(${max > 0 ? Math.min(1, window.scrollY / max) : 0})`;
    };
    const onScroll = () => { if (!raf) raf = requestAnimationFrame(tick); };
    tick();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => { window.removeEventListener("scroll", onScroll); if (raf) cancelAnimationFrame(raf); };
  }, []);

  useEffect(() => {
    const io = new IntersectionObserver(
      (es) => { for (const e of es) if (e.isIntersecting) setActive(e.target.id); },
      { rootMargin: "-15% 0px -65% 0px", threshold: 0 },
    );
    IDS.forEach((id) => { const el = document.getElementById(id); if (el) io.observe(el); });
    return () => io.disconnect();
  }, []);

  const zh = lang === "zh";

  return (
    <>
      <div className="progress" ref={bar} style={{ transform: "scaleX(0)" }} />
      <div className="shell">
        <aside className="rail">
          <div className="rail-mark">VCC 2026<span>{c.sub}</span></div>
          <LangToggle lang={lang} onChange={setLang} />
          <nav>
            {IDS.map((id, i) => (
              <a key={id} href={`#${id}`} aria-current={active === id ? "true" : undefined}>
                <b>{String(i + 1).padStart(2, "0")}</b><span>{c.nav[i]}</span>
              </a>
            ))}
          </nav>
          <div className="rail-foot">
            <div><span>{zh ? "仓库" : "Repo"}</span><b><a href={REPO}>GitHub</a></b></div>
            <div><span>{zh ? "入门版" : "Primer"}</span><b><a href={DOSSIER}>dossier</a></b></div>
            <div><span>{zh ? "修订" : "Revised"}</span><b>{MDE_META.date}</b></div>
            <div><span>{zh ? "机器" : "Machine"}</span><b>M1 Pro · {zh ? "无 GPU" : "no GPU"}</b></div>
          </div>
        </aside>

        <main>
          <div className="hero">
            <span className="eyebrow">{c.eyebrow}</span>
            <h1>{c.title}</h1>
            <p className="thesis">{c.thesis}</p>
            <div className="hero-meta">
              {c.meta.map(([k, v]) => <span key={k}>{k}<b>{v}</b></span>)}
            </div>
          </div>

          {/* ---------------------------------------------------- 01 剩下的问题 */}
          <Section id="problem" num="01"
            title={{ zh: c.secs.problem.t, en: C.en.secs.problem.t }}
            kicker={{ zh: c.secs.problem.k, en: C.en.secs.problem.k }}>
            <div className="reveal in">
              <Plain>{c.probLede}</Plain>
              <h3 style={{ margin: "28px 0 16px" }}>{c.reframe}</h3>
              <div className="grid g2">
                <div className="card">
                  <h3>{c.reframe1t}</h3>
                  <p style={{ fontSize: 15 }}>{c.reframe1}</p>
                  <M block tex={String.raw`R_p=\underbrace{\{h:\ \text{true}\}}_{\text{unknown}}\ \cap\ \underbrace{\{h:\ \text{detectable}\}}_{\textbf{given to you}}`} />
                </div>
                <div className="card">
                  <h3>{c.reframe2t}</h3>
                  <p style={{ fontSize: 15, marginBottom: 0 }}>{c.reframe2}</p>
                </div>
              </div>
            </div>
          </Section>

          {/* ------------------------------------------------------- 02 可检出性 */}
          <Section id="detect" num="02"
            title={{ zh: c.secs.detect.t, en: C.en.secs.detect.t }}
            kicker={{ zh: c.secs.detect.k, en: C.en.secs.detect.k }}>
            <div className="reveal">
              <Plain>{c.detectLede}</Plain>
              <div className="grid g4" style={{ marginBottom: 18 }}>
                {(["p5", "p25", "p50", "p75", "p95"] as const).slice(0, 4).map((q) => (
                  <Stat key={q} k={`${q.slice(1)}% ${zh ? "分位" : "pct"}`}
                    v={MDE_QUANTILES[q].toFixed(3)}
                    n={`= ${(2 ** MDE_QUANTILES[q]).toFixed(2)}× ${zh ? "倍数变化" : "fold change"}`} />
                ))}
              </div>
              <div className="dial-intro">
                <h3>{zh ? "动手试试" : "Try it"}</h3>
                <Chip p="measured">{zh ? `${MDE_META.nResolved.toLocaleString()} / ${MDE_META.nGenes.toLocaleString()} 个受检基因求出门槛` : `threshold resolved for ${MDE_META.nResolved.toLocaleString()} of ${MDE_META.nGenes.toLocaleString()} tested genes`}</Chip>
              </div>
              <MdeExplorer />
            </div>
          </Section>

          {/* -------------------------------------------------------- 03 框架 */}
          <Section id="frameworks" num="03"
            title={{ zh: c.secs.frameworks.t, en: C.en.secs.frameworks.t }}
            kicker={{ zh: c.secs.frameworks.k, en: C.en.secs.frameworks.k }}>
            <div className="reveal"><FrameworkMatrix /></div>
          </Section>

          {/* ------------------------------------------------------ 04 估计器 */}
          <Section id="estimator" num="04"
            title={{ zh: c.secs.estimator.t, en: C.en.secs.estimator.t }}
            kicker={{ zh: c.secs.estimator.k, en: C.en.secs.estimator.k }}>
            <div className="reveal">
              <Plain>{c.estLede}</Plain>
              <ProbitPanel />
            </div>
          </Section>

          {/* -------------------------------------------------------- 05 报数 */}
          <Section id="callset" num="05"
            title={{ zh: c.secs.callset.t, en: C.en.secs.callset.t }}
            kicker={{ zh: c.secs.callset.k, en: C.en.secs.callset.k }}>
            <div className="reveal"><CallsetPanel /></div>
          </Section>

          {/* ------------------------------------------------------ 06 下一步 */}
          <Section id="next" num="06"
            title={{ zh: c.secs.next.t, en: C.en.secs.next.t }}
            kicker={{ zh: c.secs.next.k, en: C.en.secs.next.k }}>
            <div className="reveal">
              <div className="grid g3" style={{ marginBottom: 18 }}>
                {c.nextItems.map((it) => (
                  <div className="card" key={it.n}>
                    <span className="eyebrow" style={{ color: "var(--up)" }}>{it.n}</span>
                    <h3 style={{ margin: "10px 0" }}>{it.t}</h3>
                    <p style={{ fontSize: 14, color: "var(--text-2)", marginBottom: 0 }}>{it.d}</p>
                  </div>
                ))}
              </div>
              <div className="note">
                <b>{c.gap}</b>
                <div style={{ marginTop: 10 }}>{c.gapBody}</div>
              </div>
            </div>
          </Section>

          <div className="foot">
            <div>VCC 2026 · {c.sub} · {MDE_META.date}</div>
            <div>{MDE_META.toolVersions} · {MDE_META.machine}</div>
            <div>{c.footLinks} — <a href={REPO}>{REPO.replace("https://", "")}</a></div>
          </div>
        </main>
      </div>
    </>
  );
}
