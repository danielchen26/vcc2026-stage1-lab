import { useMemo, useState } from "react";
import callset from "../data/callset.json";
import { useLang } from "../i18n";
import { M } from "./ui";

const { nReal, curves, anchors } = callset;

const W = 660;
const H = 300;
const P = { l: 56, r: 30, t: 22, b: 42 };
const KLO = 20;
const KHI = 3000;
const kx = (k: number) =>
  P.l + ((Math.log10(k) - Math.log10(KLO)) / (Math.log10(KHI) - Math.log10(KLO))) * (W - P.l - P.r);
const YLO = -0.6;
const YHI = 1.3;
const ky = (v: number) =>
  P.t + (1 - (Math.min(Math.max(v, YLO), YHI) - YLO) / (YHI - YLO)) * (H - P.t - P.b);

const TXT = {
  zh: {
    tag: "动手试试",
    lede: (
      <>
        「该报多少个基因」不是拍脑袋 —— 它有解析答案。<code>fid</code> 罚少报、<code>jac</code> 罚多报，
        两者联立的最优点是 <b>报数 = 真实响应基因数</b>。拖动命中率看曲线怎么动。
      </>
    ),
    hLabel: "命中率 h（你报的基因里真的响应了的比例）",
    kLabel: "你报的基因数 K",
    optLabel: "jac 的最优 K",
    nRealLabel: "真实响应集大小（推算）",
    devLabel: "最优点相对偏差",
    jacLabel: "jac（换算后）",
    fidLabel: "fid（换算后）",
    chartTitle: "换算后的分数 vs 报数 K",
    xLab: "报的基因数 K（对数刻度）",
    legend: ["jac（挑对了哪些基因）", "fid（涨跌方向对不对）", "真实响应集大小"],
    caption: (
      <>
        <b>数值确认</b>：h = 0.3 时 <code>jac</code> 的最优 K 落在 <b>287</b>，
        而由官方基线锚点反推的真实响应集大小是 <b>288</b> —— 偏差 0%。
        <br />
        <b>而这个数是白拿的</b>：一个「预测什么都没变」的提交会被判 99.3% 的基因「变了」，
        所以官方的 jac 基线满足 <M tex={String.raw`b_{\mathrm{jac}}\approx\mathbb{E}|R_p|/9{,}863`} />，
        代入 0.021–0.037 即得 200~375。本来打算花一次线上提交去套这个数。
      </>
    ),
  },
  en: {
    tag: "Try it",
    lede: (
      <>
        How many genes to flag is not a guess — it has a closed-form answer. <code>fid</code> penalises
        flagging too few, <code>jac</code> penalises too many, and the joint optimum is{" "}
        <b>flag exactly as many as really responded</b>. Drag the hit rate and watch the curves move.
      </>
    ),
    hLabel: "hit rate h (share of flagged genes that really responded)",
    kLabel: "genes you flag, K",
    optLabel: "optimal K for jac",
    nRealLabel: "true responder-set size (derived)",
    devLabel: "deviation of the optimum",
    jacLabel: "jac (rescaled)",
    fidLabel: "fid (rescaled)",
    chartTitle: "Rescaled score vs number of genes flagged",
    xLab: "genes flagged K (log scale)",
    legend: ["jac (which genes responded)", "fid (up-or-down accuracy)", "true responder-set size"],
    caption: (
      <>
        <b>Numerically confirmed</b>: at h = 0.3 the optimal K for <code>jac</code> lands on <b>287</b>,
        while the true responder-set size derived from the official baseline anchors is <b>288</b> — a
        0% deviation.
        <br />
        <b>And that number is free</b>: a submission predicting nothing changed gets 99.3% of genes
        called changed, so the official jac baseline satisfies{" "}
        <M tex={String.raw`b_{\mathrm{jac}}\approx\mathbb{E}|R_p|/9{,}863`} />, and substituting
        0.021–0.037 gives 200–375. We had planned to spend an online submission to learn it.
      </>
    ),
  },
};

export function CallsetPanel() {
  const lang = useLang();
  const c = TXT[lang];
  const [hi, setHi] = useState(2);
  const [ki, setKi] = useState(0);

  const cur = curves[hi];
  const pts = cur.points;
  const best = useMemo(
    () => pts.reduce((a, b) => (b.jacScaled > a.jacScaled ? b : a), pts[0]),
    [pts],
  );
  const kIdx = ki || pts.findIndex((p) => p.K === best.K);
  const sel = pts[Math.min(Math.max(kIdx, 0), pts.length - 1)];

  const line = (key: "jacScaled" | "fidScaled") =>
    pts.map((p, i) => `${i === 0 ? "M" : "L"} ${kx(p.K).toFixed(1)} ${ky(p[key]).toFixed(1)}`).join(" ");

  return (
    <div className="dial">
      <div className="dial-top">
        <div className="dial-plot">
          <div className="legend" style={{ marginBottom: 10 }}>
            <span><i style={{ background: "#5fe3b0" }} />{c.legend[0]}</span>
            <span><i style={{ background: "#4cc2ff" }} />{c.legend[1]}</span>
            <span><i style={{ background: "#ff7a5c" }} />{c.legend[2]}</span>
          </div>
          <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={c.chartTitle}>
            <line x1={kx(nReal)} y1={P.t} x2={kx(nReal)} y2={ky(YLO)} stroke="#ff7a5c" strokeWidth="1.5" strokeDasharray="5 3" />
            <text x={kx(nReal) + 6} y={P.t + 10} className="lb" fill="#ff7a5c">{nReal}</text>
            <line x1={P.l} y1={ky(0)} x2={W - P.r} y2={ky(0)} stroke="#ffffff26" />
            <line x1={P.l} y1={P.t} x2={P.l} y2={ky(YLO)} stroke="#ffffff26" />
            <path d={line("jacScaled")} fill="none" stroke="#5fe3b0" strokeWidth="2.2" />
            <path d={line("fidScaled")} fill="none" stroke="#4cc2ff" strokeWidth="2.2" />
            <circle cx={kx(best.K)} cy={ky(best.jacScaled)} r="5" fill="#5fe3b0" stroke="#151b29" strokeWidth="1.5" />
            <line x1={kx(sel.K)} y1={P.t} x2={kx(sel.K)} y2={ky(YLO)} stroke="#ffffff33" strokeWidth="1" />
            {[20, 100, 288, 1000, 3000].map((v) => (
              <text key={v} x={kx(v)} y={ky(YLO) + 16} textAnchor="middle" className="lb">{v}</text>
            ))}
            {[-0.5, 0, 0.5, 1.0].map((v) => (
              <text key={v} x={P.l - 8} y={ky(v) + 3.5} textAnchor="end" className="lb">{v.toFixed(1)}</text>
            ))}
            <text x={W - P.r} y={ky(YLO) + 32} textAnchor="end" className="lb">{c.xLab}</text>
          </svg>
        </div>

        <div className="dial-read">
          <div className="stat hero" style={{ border: 0, background: "transparent", padding: 0 }}>
            <span className="k">{c.optLabel}</span>
            <span className="v" style={{ color: "var(--ver)" }}>{best.K}</span>
            <span className="n">
              {c.nRealLabel} = {nReal} · {c.devLabel} {(Math.abs(best.K - nReal) / nReal * 100).toFixed(0)}%
            </span>
          </div>
          {[
            [c.kLabel, String(sel.K)],
            [c.jacLabel, sel.jacScaled.toFixed(3)],
            [c.fidLabel, sel.fidScaled.toFixed(3)],
            ["jac raw", sel.jacRaw.toFixed(4)],
            ["fid raw", sel.fidRaw.toFixed(4)],
          ].map(([k, v]) => (
            <div className="rr" key={k}>
              <span className="k">{k}</span>
              <span className="v">{v}</span>
            </div>
          ))}
          <div className="pinned">
            {lang === "zh"
              ? <>官方锚点：jac b={anchors.jac.b} r={anchors.jac.r} · fid b={anchors.fid.b} r={anchors.fid.r}</>
              : <>Official anchors: jac b={anchors.jac.b} r={anchors.jac.r} · fid b={anchors.fid.b} r={anchors.fid.r}</>}
          </div>
        </div>
      </div>

      <div className="dial-ctl">
        <label htmlFor="hh">{c.hLabel}</label>
        <input id="hh" type="range" min={0} max={curves.length - 1} step={1} value={hi}
          onChange={(e) => { setHi(Number(e.target.value)); setKi(0); }} />
        <span className="tval">{(cur.h * 100).toFixed(0)}%</span>
      </div>
      <div className="dial-ctl" style={{ paddingTop: 0, borderTop: 0 }}>
        <label htmlFor="kk">{c.kLabel}</label>
        <input id="kk" type="range" min={0} max={pts.length - 1} step={1} value={kIdx}
          onChange={(e) => setKi(Number(e.target.value))} />
        <span className="tval">{sel.K}</span>
      </div>
      <div className="dial-ctl" style={{ paddingTop: 0, borderTop: 0 }}>
        <p style={{ fontSize: 13, color: "var(--text-2)", margin: 0, maxWidth: "none" }}>{c.caption}</p>
      </div>
    </div>
  );
}
