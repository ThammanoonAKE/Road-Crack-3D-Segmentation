"use client";

import Image from "next/image";
import Link from "next/link";
import { useState } from "react";
import { CrackTrace } from "./CrackTrace";
import { ArrowRight, Cube } from "./Icons";
import { useLang } from "./LangProvider";
import { Segmented } from "./Segmented";
import { getSection, share } from "@/lib/sections";
import ui from "./ui.module.css";
import styles from "./Hero.module.css";

type Mode = "model" | "trace" | "highlight";
const HERO = getSection(2)!;

export function Hero() {
  const { t } = useLang();
  const [mode, setMode] = useState<Mode>("trace");
  const [run, setRun] = useState(0);

  const choose = (next: Mode) => {
    if (next === "trace") setRun((r) => r + 1); // replay the propagation each time the trace is shown
    setMode(next);
  };

  return (
    <section className={styles.hero} aria-labelledby="hero-title">
      <figure className={styles.stage} data-mode={mode}>
        <div className={styles.photo}>
          <Image src="/media/sections/hole_2_off.webp" alt="" fill preload sizes="100vw" className={styles.img} />
          <Image src="/media/sections/hole_2_on.webp" alt="" fill sizes="100vw" className={`${styles.img} ${styles.on}`} />
          {mode === "trace" && <CrackTrace run={run} cover />}
          <div className={styles.shade} aria-hidden />
        </div>

        <div className={`${styles.chips}`} aria-hidden>
          <span className="lg-clear mono">SEC-02 · {HERO.clip}</span>
          <span className="lg-clear mono">
            {t.hero.chip((HERO.gaussians / 1e6).toFixed(2), (share(HERO) * 100).toFixed(1))}
          </span>
        </div>

        <div className={styles.overlay}>
          <div className={`${styles.card} lg-thick`}>
            <p className={styles.eyebrow}>{t.hero.eyebrow}</p>
            <h1 id="hero-title" className={styles.title}>
              {t.hero.title}
            </h1>
            <p className={styles.lead}>{t.hero.lead}</p>
            <div className={styles.actions}>
              <Link href="/viewer/2" className={ui.primary}>
                <Cube />
                {t.hero.primary}
              </Link>
              <a href="#method" className={`${ui.secondary} lg-clear`}>
                {t.hero.secondary}
                <ArrowRight />
              </a>
            </div>
          </div>

          <div className={styles.modes}>
            <Segmented
              label={t.hero.viewsLabel}
              value={mode}
              onChange={choose}
              options={(["model", "trace", "highlight"] as const).map((m) => ({ value: m, label: t.hero.views[m] }))}
            />
            <figcaption className={`${styles.caption} lg-clear`}>{t.hero.caption}</figcaption>
          </div>
        </div>
      </figure>

      <div className={styles.widgetsWrap}>
        <dl className={styles.widgets}>
          {t.figures.map((f) => (
            <div key={f.value} className={`${styles.widget} lg`}>
              <dt className={styles.widgetLabel}>{f.label}</dt>
              <dd className={`${styles.widgetValue} mono`}>{f.value}</dd>
            </div>
          ))}
        </dl>
      </div>
    </section>
  );
}
