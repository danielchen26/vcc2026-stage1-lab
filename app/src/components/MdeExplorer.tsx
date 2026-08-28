import { useMemo, useState } from "react";
import mde from "../data/mde.json";
import { useLang } from "../i18n";
import { M } from "./ui";

const { hist, quantiles, scatter, genes, meta } = mde;

const W = 660;
const H = 260;
const P = { l: 52, r: 24, t: 18, b: 40 };
const XMAX = 2.0;
const hx = (v: number) => P.l + (Math.min(v, XMAX) / XMAX) * (W - P.l - P.r);
const MAXC = Math.max(...hist.counts);
const hy = (c: number) => P.t + (1 - c / MAXC) * (H - P.t - P.b);

const SW = 660;
const SH = 250;
const SP = { l: 52, r: 24, t: 18, b: 40 };
const XLO = 5;
const XHI = 4000;
const sx = (v: number) =>
  SP.l + ((Math.log10(Math.max(v, XLO)) - Math.log10(XLO)) / (Math.log10(XHI) - Math.log10(XLO))) *
    (SW - SP.l - SP.r);
const sy = (v: number) => SP.t + (1 - Math.min(v, XMAX) / XMAX) * (SH - SP.t - SP.b);

// 门槛用 BH 有效阈值 z=3.184（alpha*k/m），不是单基因 alpha=0.05 的 1.96。
// 范围保守地用 p95/p25（p95/p5 = 12.8×）。
const RANGE = quantiles.p95 / quantiles.p25;

const TXT = {
  zh: {
    tag: "动手试试",
    lede: (
      <>
        拖动下面的滑块选一个<b>真实变化幅度</b>。左图会告诉你：在 context A 的 9,929 个受检基因里，
        有多少个会因为这个幅度<b>被判成「变了」</b>。
      </>
    ),
    slider: "假设真实变化幅度",
    detectable: "会被判「变了」的基因",
    ofGate: "占受检基因",
    histTitle: "最小可检出效应的分布",
    histX: "需要的 |log₂ 倍数变化|",
    histY: "基因数",
    scatterTitle: "表达量越低，越难被检出",
    scatterX: "对照里的平均表达（ppm，对数刻度）",
    scatterY: "最小可检出效应 |lfc|",
    geneTitle: "逐基因查",
    genePick: "选一个基因",
    geneMde: "它的检出门槛",
    geneCpm: "对照平均表达",
    geneZero: "对照里检不到它的细胞比例",
    verdictYes: "在当前幅度下会被判「变了」",
    verdictNo: "在当前幅度下判不出来",
    caption: (
      <>
        这条门槛<b>完全由官方发给你的对照细胞决定，不含一丝生物学</b>。
        跨基因动态范围 <b>{RANGE.toFixed(1)}×</b>（p95/p25）：四分位处涨 {(2 ** quantiles.p25).toFixed(2)} 倍就被判显著，
        而 95 分位处要涨 {(2 ** quantiles.p95).toFixed(2)} 倍。
        <br />
        <b>这里用的是打分器真实的阈值。</b>BH 在每个扰动内、对 9,929 个基因做校正，
        在解点 <M tex="k=|R_p|" /> 处 <M tex="p" /> 值截断为 <M tex="\alpha k/m" />，对应双侧
        <M tex="z=3.184" /> —— 而不是单基因 α=0.05 的 1.96。用后者会把门槛**低估 1.6 倍**。
        <br />
        ⚠️ 但<b>可检出性本身赢不了比赛</b>：只按门槛排序、报 288 个基因，理论上限只到
        <M tex="h=0.130" />，而追平榜首需要 <M tex="h \ge 0.134" />。它是边际修正项，不是策略。
        <br />
        所以判定规则不该是「预测幅度超过某个固定阈值」，而应该是 <b>|预测幅度| &gt; 该基因自己的门槛</b>。
      </>
    ),
  },
  en: {
    tag: "Try it",
    lede: (
      <>
        Drag the slider to pick a <b>true effect size</b>. The chart on the left shows how many of
        context A's 9,929 tested genes would be <b>called &ldquo;changed&rdquo;</b> at that size.
      </>
    ),
    slider: "assumed true effect size",
    detectable: "genes that would be called changed",
    ofGate: "of tested genes",
    histTitle: "Distribution of the minimum detectable effect",
    histX: "required |log₂ fold change|",
    histY: "genes",
    scatterTitle: "Lower expression, harder to detect",
    scatterX: "mean expression in controls (ppm, log scale)",
    scatterY: "minimum detectable effect |lfc|",
    geneTitle: "Per gene",
    genePick: "pick a gene",
    geneMde: "its detection threshold",
    geneCpm: "mean expression in controls",
    geneZero: "share of control cells where it is undetected",
    verdictYes: "would be called changed at the current size",
    verdictNo: "would not be detected at the current size",
    caption: (
      <>
        This threshold is <b>determined entirely by the control cells the organisers hand you — no
        biology involved</b>. The dynamic range across genes is <b>{RANGE.toFixed(1)}×</b> (p95/p25): at the lower quartile a
        {" "}{(2 ** quantiles.p25).toFixed(2)}× change is already called significant, while at the 95th
        percentile it takes {(2 ** quantiles.p95).toFixed(2)}×.
        <br />
        <b>This uses the scorer's real threshold.</b> BH corrects within each perturbation across 9,929
        genes, so at the solution point <M tex="k=|R_p|" /> the p-value cutoff is <M tex="\alpha k/m" />,
        a two-sided <M tex="z=3.184" /> — not the per-gene 1.96 of α=0.05. Using the latter
        **understates the threshold by 1.6×**.
        <br />
        ⚠️ But <b>detectability alone cannot win</b>: ranking by threshold and flagging 288 genes caps
        out at <M tex="h=0.130" /> in theory, while tying the leader needs <M tex="h \ge 0.134" />.
        It is a marginal correction, not a strategy.
        <br />
        So the call rule should not be &ldquo;predicted effect above some fixed threshold&rdquo; but{" "}
        <b>|predicted effect| &gt; that gene&rsquo;s own threshold</b>.
      </>
    ),
  },
};

export function MdeExplorer() {
  const lang = useLang();
  const c = TXT[lang];
  const [eff, setEff] = useState(0.3);
  const [gi, setGi] = useState(100);

  const detectable = useMemo(() => {
    let n = 0;
    for (let i = 0; i < hist.counts.length; i++) {
      if (hist.binEdges[i + 1] <= eff) n += hist.counts[i];
      else if (hist.binEdges[i] < eff) {
        const frac = (eff - hist.binEdges[i]) / (hist.binEdges[i + 1] - hist.binEdges[i]);
        n += hist.counts[i] * frac;
      }
    }
    return Math.round(n);
  }, [eff]);

  const g = genes[gi];
  const gDetect = g.mde <= eff;

  return (
    <div className="dial">
      <div className="dial-top">
        <div className="dial-plot">
          <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={c.histTitle}>
            <text x={P.l} y={12} className="lb strong">{c.histTitle}</text>
            {hist.counts.map((n, i) => {
              const on = hist.binEdges[i + 1] <= eff;
              const part = !on && hist.binEdges[i] < eff;
              return (
                <rect key={i} x={hx(hist.binEdges[i])} y={hy(n)}
                  width={Math.max(hx(hist.binEdges[i + 1]) - hx(hist.binEdges[i]) - 1, 1)}
                  height={hy(0) - hy(n)}
                  fill={on || part ? "#ff7a5c" : "#4cc2ff"} opacity={on ? 0.85 : part ? 0.55 : 0.3} />
              );
            })}
            <line x1={hx(eff)} y1={P.t - 6} x2={hx(eff)} y2={hy(0)} stroke="#5fe3b0" strokeWidth="2" />
            <text x={hx(eff) + 6} y={P.t + 4} className="lb" fill="#5fe3b0">{eff.toFixed(2)}</text>
            <line x1={P.l} y1={hy(0)} x2={W - P.r} y2={hy(0)} stroke="#ffffff26" />
            <line x1={P.l} y1={P.t} x2={P.l} y2={hy(0)} stroke="#ffffff26" />
            {[0, 0.5, 1.0, 1.5, 2.0].map((v) => (
              <text key={v} x={hx(v)} y={hy(0) + 16} textAnchor="middle" className="lb">{v.toFixed(1)}</text>
            ))}
            {[0, MAXC].map((v) => (
              <text key={v} x={P.l - 8} y={hy(v) + 3.5} textAnchor="end" className="lb">{v}</text>
            ))}
            <text x={W - P.r} y={hy(0) + 32} textAnchor="end" className="lb">{c.histX}</text>
            <text x={P.l + 4} y={P.t - 6} className="lb">{c.histY}</text>
          </svg>

          <svg viewBox={`0 0 ${SW} ${SH}`} role="img" aria-label={c.scatterTitle} style={{ marginTop: 14 }}>
            <text x={SP.l} y={12} className="lb strong">{c.scatterTitle}</text>
            {scatter.map((p, i) => (
              <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r="1.8"
                fill={p.y <= eff ? "#ff7a5c" : "#4cc2ff"} opacity={p.y <= eff ? 0.6 : 0.28} />
            ))}
            <line x1={SP.l} y1={sy(eff)} x2={SW - SP.r} y2={sy(eff)}
              stroke="#5fe3b0" strokeWidth="1.5" strokeDasharray="4 3" />
            <circle cx={sx(g.ctrlMeanCpm)} cy={sy(g.mde)} r="5" fill="none" stroke="#fff" strokeWidth="1.6" />
            <line x1={SP.l} y1={sy(0)} x2={SW - SP.r} y2={sy(0)} stroke="#ffffff26" />
            <line x1={SP.l} y1={SP.t} x2={SP.l} y2={sy(0)} stroke="#ffffff26" />
            {[10, 100, 1000].map((v) => (
              <text key={v} x={sx(v)} y={sy(0) + 16} textAnchor="middle" className="lb">{v}</text>
            ))}
            {[0, 1, 2].map((v) => (
              <text key={v} x={SP.l - 8} y={sy(v) + 3.5} textAnchor="end" className="lb">{v}</text>
            ))}
            <text x={SW - SP.r} y={sy(0) + 32} textAnchor="end" className="lb">{c.scatterX}</text>
            <text x={SP.l + 4} y={SP.t - 6} className="lb">{c.scatterY}</text>
          </svg>
        </div>

        <div className="dial-read">
          <div className="stat hero" style={{ border: 0, background: "transparent", padding: 0 }}>
            <span className="k">{c.detectable}</span>
            <span className="v" style={{ color: "var(--up)" }}>{detectable.toLocaleString()}</span>
            <span className="n">{((detectable / meta.nResolved) * 100).toFixed(1)}% {c.ofGate}（{meta.nResolved.toLocaleString()}）</span>
          </div>

          <div style={{ borderTop: "1px solid var(--line)", paddingTop: 14 }}>
            <h4 style={{ marginBottom: 10 }}>{c.geneTitle}</h4>
            <label htmlFor="gpick" style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-3)" }}>
              {c.genePick}
            </label>
            <input id="gpick" type="range" min={0} max={genes.length - 1} step={1} value={gi}
              onChange={(e) => setGi(Number(e.target.value))} style={{ width: "100%", marginTop: 8 }} />
            <div className="rr" style={{ marginTop: 10 }}>
              <span className="k">gene</span>
              <span className="v" style={{ color: "var(--up)" }}>{g.sym}</span>
            </div>
            {[
              [c.geneMde, g.mde.toFixed(3)],
              [c.geneCpm, `${g.ctrlMeanCpm} ppm`],
              [c.geneZero, `${(g.zeroFrac * 100).toFixed(1)}%`],
            ].map(([k, v]) => (
              <div className="rr" key={k}>
                <span className="k">{k}</span>
                <span className="v">{v}</span>
              </div>
            ))}
            <div className={gDetect ? "verdict sig" : "verdict nul"} style={{ marginTop: 12, fontSize: 14 }}>
              {gDetect ? c.verdictYes : c.verdictNo}
            </div>
          </div>
        </div>
      </div>

      <div className="dial-ctl">
        <label htmlFor="eff">{c.slider}</label>
        <input id="eff" type="range" min={0.01} max={2} step={0.01} value={eff}
          onChange={(e) => setEff(Number(e.target.value))} />
        <span className="tval">{eff.toFixed(2)}</span>
        <span style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-3)" }}>
          = {(2 ** eff).toFixed(2)}×
        </span>
      </div>

      <div className="dial-ctl" style={{ paddingTop: 0, borderTop: 0 }}>
        <p style={{ fontSize: 13, color: "var(--text-2)", margin: 0, maxWidth: "none" }}>{c.caption}</p>
      </div>
    </div>
  );
}

export const MDE_QUANTILES = quantiles;
export const MDE_META = meta;
export const MDE_RANGE = RANGE;
