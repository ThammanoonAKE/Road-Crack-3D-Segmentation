"use client";

import Image from "next/image";
import Link from "next/link";
import { useState } from "react";
import { ArrowRight } from "./Icons";
import { useLang } from "./LangProvider";
import { MAX_CRACK_SHARE, SECTIONS, megabytes, pad2, share } from "@/lib/sections";
import ui from "./ui.module.css";
import styles from "./SectionGrid.module.css";

export function SectionGrid() {
  const { t } = useLang();
  const [all, setAll] = useState(false);

  return (
    <section id="sections" className={ui.section} aria-labelledby="sections-title">
      <div className={ui.wrap}>
        <div className={styles.top}>
          <div className={ui.head}>
            <p className={ui.kicker}>{t.sections.kicker}</p>
            <h2 id="sections-title" className={ui.title}>
              {t.sections.title}
            </h2>
            <p className={ui.lead}>{t.sections.lead}</p>
          </div>
          <button
            type="button"
            role="switch"
            aria-checked={all}
            className={`${styles.toggle} lg-clear`}
            onClick={() => setAll(!all)}
          >
            {t.sections.showAll}
            <span className={ui.switch} aria-hidden />
          </button>
        </div>

        <ul className={styles.grid} data-all={all}>
          {SECTIONS.map((s) => {
            const pct = share(s) * 100;
            return (
              <li key={s.n} className={`${styles.card} lg`}>
                <div className={styles.media}>
                  <Image
                    src={`/media/sections/hole_${s.n}_off.webp`}
                    alt=""
                    fill
                    sizes="(min-width: 1200px) 290px, (min-width: 720px) 45vw, 100vw"
                    className={styles.img}
                  />
                  <Image
                    src={`/media/sections/hole_${s.n}_on.webp`}
                    alt=""
                    fill
                    sizes="(min-width: 1200px) 290px, (min-width: 720px) 45vw, 100vw"
                    className={`${styles.img} ${styles.on}`}
                  />
                  <span className={`${styles.badge} lg-clear mono`}>{pad2(s.n)}</span>
                </div>
                <div className={styles.body}>
                  <div className={styles.titleRow}>
                    <h3 className={styles.name}>
                      {t.sections.section} {s.n}
                    </h3>
                    <span className={`${styles.pct} mono`}>
                      <span className="visually-hidden">{t.sections.crack} </span>
                      {pct.toFixed(1)}%
                    </span>
                  </div>
                  <span className={styles.meter} aria-hidden>
                    <span style={{ width: `${(100 * share(s)) / MAX_CRACK_SHARE}%` }} />
                  </span>
                  <dl className={styles.facts}>
                    <div>
                      <dt>{t.sections.gaussians}</dt>
                      <dd className="mono">{t.sections.millions((s.gaussians / 1e6).toFixed(2))}</dd>
                    </div>
                    <div>
                      <dt>{t.sections.size}</dt>
                      <dd className="mono">{megabytes(s.bytes)} MB</dd>
                    </div>
                  </dl>
                  <p className={`${styles.clip} mono`}>{s.clip}</p>
                  <Link href={`/viewer/${s.n}`} className={styles.open}>
                    {t.sections.open(s.n)}
                    <ArrowRight />
                  </Link>
                </div>
              </li>
            );
          })}
        </ul>
      </div>
    </section>
  );
}
