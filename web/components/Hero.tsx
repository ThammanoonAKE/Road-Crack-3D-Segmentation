"use client";

import dynamic from "next/dynamic";
import Image from "next/image";
import Link from "next/link";
import { Component, useState, useSyncExternalStore, type ReactNode } from "react";
import { CrackTrace } from "./CrackTrace";
import { ArrowRight, Cube } from "./Icons";
import { useLang } from "./LangProvider";
import { Segmented } from "./Segmented";
import { getSection, share } from "@/lib/sections";
import ui from "./ui.module.css";
import styles from "./Hero.module.css";

type Mode = "model" | "trace" | "highlight";
const HERO = getSection(2)!;

// three.js and the point cloud load after the page is interactive, and only in the browser
const HeroCloud = dynamic(() => import("./HeroCloud"), { ssr: false });

const STILL = "(prefers-reduced-motion: reduce)";
const watchMotion = (changed: () => void) => {
  const query = window.matchMedia(STILL);
  query.addEventListener("change", changed);
  return () => query.removeEventListener("change", changed);
};
/** The live point cloud is for visitors who have not asked for less motion or less data; the rest get the photograph. */
const heroKind = () =>
  window.matchMedia(STILL).matches || (navigator as { connection?: { saveData?: boolean } }).connection?.saveData
    ? "photo"
    : "cloud";

/** Without WebGL the canvas throws: the hero then stays on its photograph. */
class CloudBoundary extends Component<{ children: ReactNode; onError: () => void }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  componentDidCatch() {
    this.props.onError();
  }
  render() {
    return this.state.failed ? null : this.props.children;
  }
}

export function Hero() {
  const { t } = useLang();
  const [mode, setMode] = useState<Mode>("trace");
  const [run, setRun] = useState(0);
  // null on the server and until hydration: the stage stays dark, so the photograph never flashes before the cloud
  const kind = useSyncExternalStore(watchMotion, heroKind, () => null);
  const [failed, setFailed] = useState(false);
  const live = kind === "cloud" && !failed; // the cloud is mounted, though perhaps not drawn yet
  const photo = kind === "photo" || failed;

  const choose = (next: Mode) => {
    if (next === "trace") setRun((r) => r + 1); // replay the propagation each time the trace is shown
    setMode(next);
  };

  return (
    <section className={styles.hero} aria-labelledby="hero-title">
      <figure className={styles.stage} data-mode={mode} data-photo={photo}>
        <div className={styles.photo}>
          <Image src="/media/sections/hole_2_off.webp" alt="" fill preload sizes="100vw" className={styles.img} />
          <Image src="/media/sections/hole_2_on.webp" alt="" fill sizes="100vw" className={`${styles.img} ${styles.on}`} />
          {mode === "trace" && photo && <CrackTrace run={run} cover />}
          {live && (
            <CloudBoundary onError={() => setFailed(true)}>
              <HeroCloud mode={mode} run={run} onError={() => setFailed(true)} />
            </CloudBoundary>
          )}
          <div className={styles.shade} aria-hidden />
        </div>

        <div className={`${styles.chips}`} aria-hidden>
          <span className="lg-clear mono">
            {t.hero.chip((HERO.gaussians / 1e6).toFixed(2), (share(HERO) * 100).toFixed(1))}
          </span>
        </div>

        <div className={styles.overlay}>
          <div className={`${styles.card} lg-thick`}>
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
          </div>
        </div>
      </figure>

      <div className={styles.widgetsWrap}>
        <dl className={styles.widgets}>
          {t.figures.map((f) => (
            <div key={f.value} className={`${styles.widget} lg rise`}>
              <dt className={styles.widgetLabel}>{f.label}</dt>
              <dd className={`${styles.widgetValue} mono`}>{f.value}</dd>
            </div>
          ))}
        </dl>
      </div>
    </section>
  );
}
