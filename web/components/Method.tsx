"use client";

import Image from "next/image";
import { CompareSlider } from "./CompareSlider";
import { External } from "./Icons";
import { useLang } from "./LangProvider";
import ui from "./ui.module.css";
import styles from "./Method.module.css";

// the public repository, where the scripts for every stage are (also linked from the footer)
const REPO = process.env.NEXT_PUBLIC_REPO_URL;

export function Method() {
  const { t } = useLang();
  const m = t.method;

  return (
    <section id="method" className={ui.section} aria-labelledby="method-title">
      <div className={`${ui.wrap} ${styles.layout}`}>
        <div className={styles.aside}>
          <div className={`${ui.head} rise`}>
            <p className={ui.kicker}>{m.kicker}</p>
            <h2 id="method-title" className={ui.title}>
              {m.title}
            </h2>
            <p className={ui.lead}>{m.lead}</p>
            {REPO && (
              <a href={REPO} className={`${ui.secondary} lg-clear ${styles.repo}`} target="_blank" rel="noreferrer">
                {m.repo}
                <External />
              </a>
            )}
          </div>
        </div>

        <ol className={`${styles.list} lg-thick rise`}>
          {m.steps.map((step, i) => (
            <li key={step.code} className={styles.row}>
              <span className={`${styles.icon} mono`} aria-hidden>
                {String(i + 1).padStart(2, "0")}
              </span>
              <div className={styles.text}>
                <div className={styles.rowHead}>
                  <h3 className={styles.stepTitle}>{step.title}</h3>
                  <span className={`${styles.code} mono`}>{step.code}</span>
                </div>
                <p className={styles.body}>{step.body}</p>
                {step.code === "DIFIX3D+" && (
                  <div className={styles.media}>
                    <CompareSlider
                      before="/media/difix/before.webp"
                      after="/media/difix/after.webp"
                      width={962}
                      height={496}
                      beforeLabel={m.before}
                      afterLabel={m.after}
                      label={m.compareLabel}
                      hint={m.compareHint}
                    />
                    <p className={styles.caption}>{m.compareCaption}</p>
                  </div>
                )}
                {step.code === "DEEPCRACK" && (
                  <figure className={styles.media}>
                    <Image
                      src="/media/crack_labels_hole6.jpg"
                      alt={m.labelsCaption}
                      width={1280}
                      height={337}
                      sizes="(min-width: 1100px) 680px, 100vw"
                      className={styles.figureImg}
                    />
                    <figcaption className={styles.caption}>{m.labelsCaption}</figcaption>
                  </figure>
                )}
              </div>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
