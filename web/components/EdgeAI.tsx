"use client";

import { useEffect, useRef, useState } from "react";
import { Pause, Play } from "./Icons";
import { useLang } from "./LangProvider";
import ui from "./ui.module.css";
import styles from "./EdgeAI.module.css";

const ROWS = [
  { name: "PyTorch bf16", portrait: 19.1, landscape: 19.2, iou: null as string | null, accent: false },
  { name: "TensorRT BF16", portrait: 7.85, landscape: 7.78, iou: "0.995", accent: true },
  { name: "TensorRT NVFP4", portrait: 7.68, landscape: 7.63, iou: "0.93", accent: false },
];
const SCALE = 20; // ms at full bar width

export function EdgeAI() {
  const { t } = useLang();
  const e = t.edge;
  const video = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);

  // Autoplay unless the visitor asked for reduced motion; the pause control is always there.
  useEffect(() => {
    const v = video.current;
    if (!v) return;
    if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      v.play().then(() => setPlaying(true)).catch(() => setPlaying(false));
    }
  }, []);

  const toggle = () => {
    const v = video.current;
    if (!v) return;
    if (v.paused) v.play().then(() => setPlaying(true)).catch(() => {});
    else {
      v.pause();
      setPlaying(false);
    }
  };

  return (
    <section id="edge" className={ui.section} aria-labelledby="edge-title">
      <div className={ui.wrap}>
        <div className={`${ui.head} rise`}>
          <p className={ui.kicker}>{e.kicker}</p>
          <h2 id="edge-title" className={ui.title}>
            {e.title}
          </h2>
          <p className={ui.lead}>{e.lead}</p>
        </div>

        <div className={styles.grid}>
          <figure className={`${styles.player} lg rise`}>
            <div className={styles.screen}>
              <video
                ref={video}
                className={styles.video}
                src="/media/video/stream_walk.mp4"
                poster="/media/video/stream_walk.jpg"
                muted
                loop
                playsInline
                preload="metadata"
                width={960}
                height={544}
              />
              <button
                type="button"
                className={`${styles.play} lg-clear`}
                onClick={toggle}
                aria-label={playing ? e.pause : e.play}
              >
                {playing ? <Pause /> : <Play />}
              </button>
            </div>
            <figcaption className={styles.caption}>{e.videoCaption}</figcaption>
          </figure>

          <div className={`${styles.stats} lg-thick rise`}>
            <table className={styles.table}>
              <caption className={styles.tableCaption}>{e.tableCaption}</caption>
              <thead>
                <tr>
                  <th scope="col">{e.backend}</th>
                  <th scope="col">{e.landscape}</th>
                  <th scope="col">{e.agreement}</th>
                </tr>
              </thead>
              <tbody>
                {ROWS.map((r) => (
                  <tr key={r.name} data-accent={r.accent}>
                    <th scope="row">{r.name}</th>
                    <td>
                      <span className={`${styles.ms} mono`}>{r.landscape.toFixed(2)} ms</span>
                      <span className={styles.bar} aria-hidden>
                        <span style={{ width: `${(100 * r.landscape) / SCALE}%` }} />
                      </span>
                      <span className={`${styles.sub} mono`}>
                        {e.portrait}: {r.portrait.toFixed(2)} ms
                      </span>
                    </td>
                    <td className="mono">{r.iou ? e.iou(r.iou) : e.reference}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className={styles.notes}>
              <p>{e.note}</p>
              <p>{e.stability}</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
