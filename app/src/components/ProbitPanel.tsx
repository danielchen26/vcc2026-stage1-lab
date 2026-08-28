import { useState } from "react";
import mdeData from "../data/mde.json";
import { useLang } from "../i18n";
import { needsNonparametric, pDetect } from "../lib/stats";
import { M } from "./ui";

const GENES = mdeData.genes;

const W = 660;
const H = 250;
const P = { l: 54, r: 26, t: 20, b: 40 };
const XMAX = 2.0;
const gx = (v: number) => P.l + (Math.min(Math.max(v, -XMAX), XMAX) / XMAX / 2 + 0.5) * (W - P.l - P.r);
const gy = (p: number) => P.t + (1 - p) * (H - P.t - P.b);

const TXT = {
  zh: {
    tag: "动手试试",
    lede: (
      <>
        这就是装配好的估计器核心。左边两个滑块是 Stage 1 要预测的两个量；
        右边是它们经过<b>可检出性 probit</b> 之后得到的「这个基因会被判变了」的概率。
      </>
    ),
    muLabel: "预测的变化幅度 μ̂（log₂）",
    sigmaLabel: "跨源细胞系的不确定度 σ̃",
    genePick: "目标基因（决定检出门槛）",
    pLabel: "P(会被判「变了」)",
    mdeLabel: "该基因的检出门槛 MDE",
    signLabel: "报告的方向",
    up: "涨",
    down: "跌",
    npLabel: "需要非参处理？",
    npYes: "是（落在门槛附近）",
    npNo: "否（解析 Φ 够用，误差 < 2%）",
    curveTitle: "P(判变了) 随预测幅度的变化",
    curveX: "预测的变化幅度 μ̂（log₂）",
    caption: (
      <>
        <b>灰带</b>是「检不出来」的区间 ±MDE。曲线在带外迅速趋近 1，在带内被压向 0 ——
        所以同一个预测幅度，换个基因结论可以完全相反。
        <br />
        <b>省算力的门控</b>：只有 |μ̂|−MDE 落在 1σ 以内的基因（约 2k 个）需要走经验 CDF 的非参处理，
        其余 16k 用解析 Φ。这条规则在 AdaptiveEROP 的 <code>src/Pipeline/predict.jl</code> 里现成。
      </>
    ),
  },
  en: {
    tag: "Try it",
    lede: (
      <>
        This is the core of the assembled estimator. The two sliders on the left are the quantities
        Stage 1 has to predict; the panel on the right is what they become after the{" "}
        <b>detectability probit</b> — the probability that this gene gets called changed.
      </>
    ),
    muLabel: "predicted effect μ̂ (log₂)",
    sigmaLabel: "uncertainty across source lines σ̃",
    genePick: "target gene (sets the threshold)",
    pLabel: "P(called changed)",
    mdeLabel: "this gene's threshold MDE",
    signLabel: "reported direction",
    up: "up",
    down: "down",
    npLabel: "needs non-parametric handling?",
    npYes: "yes (sits near the threshold)",
    npNo: "no (analytic Φ suffices, error < 2%)",
    curveTitle: "P(called changed) as the predicted effect varies",
    curveX: "predicted effect μ̂ (log₂)",
    caption: (
      <>
        The <b>grey band</b> is the undetectable interval ±MDE. The curve approaches 1 quickly outside
        it and is crushed toward 0 inside — so the same predicted effect can give opposite conclusions
        on different genes.
        <br />
        <b>The compute-saving gate</b>: only genes whose |μ̂|−MDE falls within 1σ (about 2k of them)
        need empirical-CDF Monte Carlo; the other 16k use the analytic Φ. That rule already exists in
        AdaptiveEROP's <code>src/Pipeline/predict.jl</code>.
      </>
    ),
  },
};

const CURVE_TEX = String.raw`P(h\in R_p)=\Phi\!\left(\frac{\hat\mu_h-\mathrm{MDE}_{h,c}}{\tilde\sigma_h}\right)
  +\Phi\!\left(\frac{-\mathrm{MDE}_{h,c}-\hat\mu_h}{\tilde\sigma_h}\right)`;

export function ProbitPanel() {
  const lang = useLang();
  const c = TXT[lang];
  const [mu, setMu] = useState(0.45);
  const [sigma, setSigma] = useState(0.25);
  const [gi, setGi] = useState(100);

  const g = GENES[gi];
  const p = pDetect(mu, sigma, g.mde);
  const np = needsNonparametric(mu, sigma, g.mde);

  const path: string[] = [];
  for (let i = 0; i <= 200; i++) {
    const x = -XMAX + (2 * XMAX * i) / 200;
    path.push(`${i === 0 ? "M" : "L"} ${gx(x).toFixed(1)} ${gy(pDetect(x, sigma, g.mde)).toFixed(1)}`);
  }

  return (
    <div className="dial">
      <div className="dial-top">
        <div className="dial-plot">
          <M block tex={CURVE_TEX} />
          <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={c.curveTitle}>
            <text x={P.l} y={12} className="lb strong">{c.curveTitle}</text>
            <rect x={gx(-g.mde)} y={P.t} width={gx(g.mde) - gx(-g.mde)} height={gy(0) - P.t}
              fill="#ffffff" opacity="0.05" />
            <line x1={gx(-g.mde)} y1={P.t} x2={gx(-g.mde)} y2={gy(0)} stroke="#5fe3b0" strokeWidth="1" strokeDasharray="3 3" />
            <line x1={gx(g.mde)} y1={P.t} x2={gx(g.mde)} y2={gy(0)} stroke="#5fe3b0" strokeWidth="1" strokeDasharray="3 3" />
            <text x={gx(g.mde) + 5} y={P.t + 11} className="lb" fill="#5fe3b0">±MDE = {g.mde.toFixed(3)}</text>

            <line x1={gx(0)} y1={P.t} x2={gx(0)} y2={gy(0)} stroke="#ffffff1a" />
            <line x1={P.l} y1={gy(0.5)} x2={W - P.r} y2={gy(0.5)} stroke="#ffffff12" strokeDasharray="2 4" />
            <path d={path.join(" ")} fill="none" stroke="#4cc2ff" strokeWidth="2.2" />

            <line x1={gx(mu)} y1={P.t} x2={gx(mu)} y2={gy(0)} stroke="#ff7a5c" strokeWidth="1.6" />
            <circle cx={gx(mu)} cy={gy(p)} r="5" fill="#ff7a5c" stroke="#151b29" strokeWidth="1.5" />

            <line x1={P.l} y1={gy(0)} x2={W - P.r} y2={gy(0)} stroke="#ffffff26" />
            <line x1={P.l} y1={P.t} x2={P.l} y2={gy(0)} stroke="#ffffff26" />
            {[-2, -1, 0, 1, 2].map((v) => (
              <text key={v} x={gx(v)} y={gy(0) + 16} textAnchor="middle" className="lb">{v}</text>
            ))}
            {[0, 0.5, 1].map((v) => (
              <text key={v} x={P.l - 8} y={gy(v) + 3.5} textAnchor="end" className="lb">{v.toFixed(1)}</text>
            ))}
            <text x={W - P.r} y={gy(0) + 32} textAnchor="end" className="lb">{c.curveX}</text>
          </svg>
        </div>

        <div className="dial-read">
          <div className="stat hero" style={{ border: 0, background: "transparent", padding: 0 }}>
            <span className="k">{c.pLabel}</span>
            <span className="v" style={{ color: p > 0.5 ? "var(--up)" : "var(--down)" }}>
              {p.toFixed(3)}
            </span>
          </div>
          {[
            ["gene", g.sym],
            [c.mdeLabel, g.mde.toFixed(3)],
            [c.signLabel, mu >= 0 ? c.up : c.down],
          ].map(([k, v]) => (
            <div className="rr" key={k}>
              <span className="k">{k}</span>
              <span className="v">{v}</span>
            </div>
          ))}
          <div className="pinned" style={{
            background: np ? "var(--warn-dim)" : "var(--ver-dim)",
            borderColor: np ? "#ffc66b3d" : "#5fe3b02a",
          }}>
            <b style={{ color: np ? "var(--warn)" : "var(--ver)" }}>{c.npLabel}</b>
            <br />
            {np ? c.npYes : c.npNo}
          </div>
          <label htmlFor="gp2" style={{ fontFamily: "var(--mono)", fontSize: 10, color: "var(--text-3)" }}>
            {c.genePick}
          </label>
          <input id="gp2" type="range" min={0} max={GENES.length - 1} step={1} value={gi}
            onChange={(e) => setGi(Number(e.target.value))} />
        </div>
      </div>

      <div className="dial-ctl">
        <label htmlFor="mu">{c.muLabel}</label>
        <input id="mu" type="range" min={-2} max={2} step={0.01} value={mu}
          onChange={(e) => setMu(Number(e.target.value))} />
        <span className="tval">{mu >= 0 ? "+" : ""}{mu.toFixed(2)}</span>
      </div>
      <div className="dial-ctl" style={{ paddingTop: 0, borderTop: 0 }}>
        <label htmlFor="sg">{c.sigmaLabel}</label>
        <input id="sg" type="range" min={0.02} max={1.2} step={0.01} value={sigma}
          onChange={(e) => setSigma(Number(e.target.value))} />
        <span className="tval">{sigma.toFixed(2)}</span>
      </div>
      <div className="dial-ctl" style={{ paddingTop: 0, borderTop: 0 }}>
        <p style={{ fontSize: 13, color: "var(--text-2)", margin: 0, maxWidth: "none" }}>{c.caption}</p>
      </div>
    </div>
  );
}
