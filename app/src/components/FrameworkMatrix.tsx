import { useState } from "react";
import { FRAMEWORKS, UNIFYING, VERDICT_LABEL, type Verdict } from "../data/frameworks";
import { useLang } from "../i18n";
import { M } from "./ui";

const DOT: Record<Verdict, string> = {
  "worth-investing": "var(--ver)",
  partial: "var(--warn)",
  narrow: "var(--warn)",
  "not-applicable": "var(--text-3)",
};

const TXT = {
  zh: {
    hint: "点一行展开",
    borrow: "可借的",
    na: "不适用的部分",
    mapping: "映射到 Stage 1",
    evidence: "证据",
    cols: ["框架", "它是什么", "判决"],
  },
  en: {
    hint: "click a row to expand",
    borrow: "What transfers",
    na: "What does not",
    mapping: "Mapping to Stage 1",
    evidence: "Evidence",
    cols: ["Framework", "What it is", "Verdict"],
  },
};

export function FrameworkMatrix() {
  const lang = useLang();
  const c = TXT[lang];
  const [open, setOpen] = useState<string | null>("erop");

  return (
    <>
      <div className="plain" style={{ marginBottom: 20 }}>
        <span className="plain-tag">{lang === "zh" ? "一句话" : "In short"}</span>
        <div>{UNIFYING[lang]}</div>
      </div>

      <div className="cmp">
        <div className="cmp-head" style={{ gridTemplateColumns: "216px minmax(0,1fr) 168px" }}>
          {c.cols.map((h, i) => <div key={h} className={i === 2 ? "us" : ""}>{h}</div>)}
        </div>

        {FRAMEWORKS.map((f) => {
          const isOpen = open === f.id;
          return (
            <div key={f.id}>
              <button type="button"
                onClick={() => setOpen(isOpen ? null : f.id)}
                aria-expanded={isOpen}
                className="fw-row"
                style={{ gridTemplateColumns: "216px minmax(0,1fr) 168px" }}>
                <div className="axis">
                  {f.name}
                  <span>{f.repo}</span>
                </div>
                <div className="them">{f.what[lang]}</div>
                <div className="ev">
                  <span className="chip" style={{ color: DOT[f.verdict], borderColor: DOT[f.verdict] }}>
                    {VERDICT_LABEL[f.verdict][lang]}
                  </span>
                </div>
              </button>

              {isOpen && (
                <div className="fw-body">
                  <p style={{ fontSize: 15, marginBottom: 20 }}>{f.summary[lang]}</p>

                  {f.borrowables.length > 0 && (
                    <>
                      <h4 style={{ color: "var(--ver)", marginBottom: 12 }}>{c.borrow}</h4>
                      <div className="grid g2" style={{ marginBottom: 20 }}>
                        {f.borrowables.map((b) => (
                          <div className="card" key={b.file} style={{ padding: "18px 20px" }}>
                            <h4 style={{ marginBottom: 10 }}>{b.title[lang]}</h4>
                            {b.formula && <M block tex={b.formula} />}
                            <p style={{ fontSize: 13, color: "var(--text-2)", marginBottom: 10 }}>
                              {b.mapping[lang]}
                            </p>
                            <code style={{ fontSize: 11 }}>{b.file}</code>
                          </div>
                        ))}
                      </div>
                    </>
                  )}

                  {f.notApplicable.length > 0 && (
                    <>
                      <h4 style={{ color: "var(--text-3)", marginBottom: 12 }}>{c.na}</h4>
                      <div className="tw">
                        <table>
                          <tbody>
                            {f.notApplicable.map((n) => (
                              <tr key={n.evidence}>
                                <td style={{ color: "var(--text-2)" }}>{n.reason[lang]}</td>
                                <td className="mono" style={{ color: "var(--text-3)", minWidth: 200, fontSize: 11.5 }}>
                                  {n.evidence}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
      <p style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text-3)", marginTop: 12 }}>
        {c.hint}
      </p>
    </>
  );
}
