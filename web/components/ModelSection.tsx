"use client";

import { useLang } from "./LangProvider";
import ui from "./ui.module.css";
import styles from "./ModelSection.module.css";

export function ModelSection() {
  const { t } = useLang();
  const m = t.model;
  return (
    <section id="model" className={ui.section} aria-labelledby="model-title">
      <div className={ui.wrap}>
        <div className={ui.head}>
          <p className={ui.kicker} data-reveal="left">{m.kicker}</p>
          <h2 id="model-title" className={ui.title} data-reveal="blur" style={{ "--i": 1 } as React.CSSProperties}>
            {m.title}
          </h2>
          <p className={ui.lead} data-reveal="up" style={{ "--i": 2 } as React.CSSProperties}>{m.lead}</p>
        </div>

        <div className={`${styles.panel} lg-thick`} data-reveal="up">
          <div className={styles.scroller} tabIndex={0} role="region" aria-labelledby="model-title">
            <table className={styles.table}>
              <thead>
                <tr>
                  {m.headers.map((h, i) => (
                    <th key={h} scope="col" data-ours={i === 2}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {m.rows.map(([aspect, paper, ours], i) => (
                  <tr key={aspect} data-row style={{ "--i": i } as React.CSSProperties}>
                    <th scope="row">{aspect}</th>
                    <td>{paper}</td>
                    <td data-ours>{ours}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <dl className={styles.metrics}>
          {m.metrics.map((x, i) => (
            <div key={x.value} className="lg" data-reveal="zoom" style={{ "--i": i } as React.CSSProperties}>
              <dt>{x.label}</dt>
              <dd className="mono">{x.value}</dd>
            </div>
          ))}
        </dl>
      </div>
    </section>
  );
}
