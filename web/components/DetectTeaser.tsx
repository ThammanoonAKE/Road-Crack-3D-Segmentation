"use client";

import { useEffect, useState } from "react";
import { CrackReveal } from "./CrackReveal";
import { ArrowRight } from "./Icons";
import { useLang } from "./LangProvider";
import ui from "./ui.module.css";
import styles from "./DetectTeaser.module.css";

const PHOTO = "/media/detect/sample_1.webp";
const MASK = "/media/detect/sample_1_mask.png"; // the detector's own output for that photo (edge_ai/onnx_web.py)

function usePicture(src: string) {
  const [picture, setPicture] = useState<HTMLImageElement | null>(null);
  useEffect(() => {
    const image = new Image();
    image.onload = () => setPicture(image);
    image.src = src;
    return () => {
      image.onload = null;
    };
  }, [src]);
  return picture;
}

/** The home page's way into /detect: one of the sample photos with its cracks spreading, and the link. */
export function DetectTeaser() {
  const { t } = useLang();
  const d = t.teaser;
  const photo = usePicture(PHOTO);
  const mask = usePicture(MASK);

  return (
    <section id="detect" className={ui.section} aria-labelledby="detect-title">
      <div className={ui.wrap}>
        <div className={styles.grid}>
          <div className={styles.text}>
            <p className={ui.kicker} data-reveal="left">{d.kicker}</p>
            <h2 id="detect-title" className={ui.title} data-reveal="blur" style={{ "--i": 1 } as React.CSSProperties}>
              {d.title}
            </h2>
            <p className={ui.lead} data-reveal="up" style={{ "--i": 2 } as React.CSSProperties}>{d.lead}</p>
            {/* a plain link, not next/link: /detect is loaded as a document so that its own response headers apply */}
            <a href="/detect" className={`${ui.primary} ${styles.open}`} data-reveal="up" style={{ "--i": 3 } as React.CSSProperties}>
              {d.open}
              <ArrowRight />
            </a>
          </div>

          <figure className={`${styles.figure} lg`} data-reveal="right" style={{ "--i": 1 } as React.CSSProperties}>
            <div className={styles.screen}>
              <CrackReveal className={styles.canvas} photo={photo} mask={photo && mask} replayInView label={d.figure} />
            </div>
            <figcaption className={styles.caption}>{d.figure}</figcaption>
          </figure>
        </div>
      </div>
    </section>
  );
}
