"use client";

import Image from "next/image";
import Link from "next/link";
import { useState, type PointerEvent } from "react";
import { ArrowRight } from "./Icons";
import { useLang } from "./LangProvider";
import { MAX_CRACK_SHARE, SECTIONS, pad2, share } from "@/lib/sections";
import ui from "./ui.module.css";
import styles from "./SectionGrid.module.css";

/** A card leans towards the mouse and a highlight follows it; ui: SectionGrid.module.css `.card`. */
function lean(e: PointerEvent<HTMLLIElement>) {
  if (e.pointerType !== "mouse") return;
  const card = e.currentTarget;
  const box = card.getBoundingClientRect();
  const x = (e.clientX - box.left) / box.width;
  const y = (e.clientY - box.top) / box.height;
  card.style.setProperty("--tilt-x", `${((0.5 - y) * 7).toFixed(2)}deg`);
  card.style.setProperty("--tilt-y", `${((x - 0.5) * 9).toFixed(2)}deg`);
  card.style.setProperty("--glare-x", `${(x * 100).toFixed(1)}%`);
  card.style.setProperty("--glare-y", `${(y * 100).toFixed(1)}%`);
}

function settle(e: PointerEvent<HTMLLIElement>) {
  e.currentTarget.style.removeProperty("--tilt-x");
  e.currentTarget.style.removeProperty("--tilt-y");
}

export function SectionGrid() {
  const { t } = useLang();
  const [all, setAll] = useState(false);

  return (
    <section id="sections" className={ui.section} aria-labelledby="sections-title">
      <div className={ui.wrap}>
        <div className={styles.top}>
          <div className={ui.head}>
            <p className={ui.kicker} data-reveal="left">{t.sections.kicker}</p>
            <h2 id="sections-title" className={ui.title} data-reveal="blur" style={{ "--i": 1 } as React.CSSProperties}>
              {t.sections.title}
            </h2>
            <p className={ui.lead} data-reveal="up" style={{ "--i": 2 } as React.CSSProperties}>{t.sections.lead}</p>
          </div>
          <button
            type="button"
            role="switch"
            aria-checked={all}
            className={`${styles.toggle} lg-clear`}
            data-reveal="right"
            style={{ "--i": 2 } as React.CSSProperties}
            onClick={() => setAll(!all)}
          >
            {t.sections.showAll}
            <span className={ui.switch} aria-hidden />
          </button>
        </div>

        <ul className={styles.grid} data-all={all}>
          {SECTIONS.map((s, i) => {
            const pct = share(s) * 100;
            return (
              <li
                key={s.n}
                className={`${styles.card} lg`}
                data-reveal="up"
                style={{ "--i": i % 4 } as React.CSSProperties}
                onPointerMove={lean}
                onPointerLeave={settle}
              >
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
                    <span data-grow style={{ width: `${(100 * share(s)) / MAX_CRACK_SHARE}%` }} />
                  </span>
                  <dl className={styles.facts}>
                    <div>
                      <dt>{t.sections.gaussians}</dt>
                      <dd className="mono">{t.sections.millions((s.gaussians / 1e6).toFixed(2))}</dd>
                    </div>
                  </dl>
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
