"use client";

import Image from "next/image";
import { useEffect, useMemo, useRef, useState } from "react";
import { CrackReveal, type RevealMode } from "@/components/CrackReveal";
import { Upload } from "@/components/Icons";
import { useLang } from "@/components/LangProvider";
import { Segmented } from "@/components/Segmented";
import ui from "@/components/ui.module.css";
import { DETECTOR_MB, LONG_SIDE, detect, detectorReady, loadDetector, metered, shrink, type Detection } from "@/lib/detector";
import styles from "./DetectApp.module.css";

type Phase = "idle" | "model" | "detecting" | "done" | "error";
type Failure = "image" | "model" | "detect";

const SAMPLES = [1, 2, 3].map((n) => `/media/detect/sample_${n}.webp`);
const SHOWN = 1600; // the photo is drawn from a copy this long at most

/**
 * The /detect page: the visitor picks a photo, the detector runs in their browser (lib/detector.ts) and the cracks
 * it finds spread over the photo from the middle (CrackReveal). The photo is only ever read into this page's memory:
 * nothing is uploaded and nothing is stored.
 */
export function DetectApp() {
  const { t } = useLang();
  const d = t.detect;
  const [photo, setPhoto] = useState<HTMLCanvasElement | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [progress, setProgress] = useState(0);
  const [ready, setReady] = useState(detectorReady);
  const [result, setResult] = useState<Detection | null>(null);
  const [failure, setFailure] = useState<Failure | null>(null);
  const [mode, setMode] = useState<RevealMode>("trace");
  const [run, setRun] = useState(0);
  const [over, setOver] = useState(false);
  const job = useRef(0); // a newer photo makes the older one's result unwanted
  const last = useRef<Blob | null>(null);
  const picker = useRef<HTMLInputElement>(null);
  const mask = useMemo(() => result && { data: result.mask, width: result.width, height: result.height }, [result]);

  // The model is fetched and opened as soon as the page is, so the first photo starts at once; from the home page
  // it is usually in the browser's cache already. On a metered connection it waits for the first photo.
  useEffect(() => {
    if (metered()) return;
    loadDetector(setProgress).then(
      () => setReady(true),
      () => {}, // the first photo tries again and reports the failure
    );
  }, []);

  const analyse = async (source: Blob) => {
    const id = ++job.current;
    const current = () => id === job.current;
    const fail = (why: Failure) => {
      if (!current()) return;
      setFailure(why);
      setPhase("error");
    };
    last.current = source;
    setFailure(null);
    setResult(null);
    setMode("trace");

    let small: HTMLCanvasElement;
    let scale: number;
    try {
      const bitmap = await createImageBitmap(source);
      if (!current()) return bitmap.close();
      small = shrink(bitmap, bitmap.width, bitmap.height, LONG_SIDE);
      scale = small.width / bitmap.width;
      setPhoto(shrink(bitmap, bitmap.width, bitmap.height, SHOWN));
      bitmap.close();
    } catch {
      return fail("image");
    }

    if (!detectorReady()) {
      setProgress(0);
      setPhase("model");
    } else setPhase("detecting");
    try {
      await loadDetector((share) => current() && setProgress(share));
    } catch {
      return fail("model");
    }
    setReady(true);
    if (!current()) return;
    setPhase("detecting");
    try {
      const pixels = small.getContext("2d")!.getImageData(0, 0, small.width, small.height);
      const found = await detect(pixels, scale);
      if (!current()) return;
      setResult(found);
      setRun((r) => r + 1);
      setPhase("done");
    } catch {
      fail("detect");
    }
  };

  const sample = async (url: string) => {
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(response.statusText);
      await analyse(await response.blob());
    } catch {
      setFailure("image");
      setPhase("error");
    }
  };

  const choose = (next: RevealMode) => {
    if (next === "trace") setRun((r) => r + 1); // replay the spread each time the trace is shown
    setMode(next);
  };

  const drop = (event: React.DragEvent) => {
    event.preventDefault();
    setOver(false);
    const file = [...event.dataTransfer.files].find((f) => f.type.startsWith("image/"));
    if (file) analyse(file);
  };

  const working = phase === "model" || phase === "detecting";
  const percent = Math.round(progress * 100);
  const status =
    phase === "model"
      ? percent < 100
        ? d.downloading(percent)
        : d.preparing
      : phase === "detecting"
        ? d.detecting
        : phase === "error" && failure
          ? d.errors[failure]
          : phase === "done" && result && result.share === 0
            ? d.none
            : "";
  // under the photo, until there is a result: where the model is
  const note = ready
    ? phase === "idle"
      ? d.ready
      : ""
    : phase === "idle" && progress > 0
      ? percent < 100
        ? d.downloading(percent)
        : d.preparing
      : d.note(DETECTOR_MB);
  const share = result ? (100 * result.share).toFixed(result.share < 0.1 ? 1 : 0) : "";
  const seconds = result ? (result.ms / 1000).toFixed(result.ms < 9950 ? 1 : 0) : "";

  return (
    <section className={styles.page} aria-labelledby="detect-title">
      <div className={ui.wrap}>
        <div className={`${ui.head} ${styles.head}`}>
          <p className={ui.kicker}>{d.kicker}</p>
          <h1 id="detect-title" className={ui.title}>
            {d.title}
          </h1>
          <p className={ui.lead}>{d.lead}</p>
        </div>

        <input
          ref={picker}
          type="file"
          accept="image/*"
          className={styles.file}
          tabIndex={-1}
          aria-hidden
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) analyse(file);
            event.target.value = ""; // the same file can be chosen again
          }}
        />

        <figure
          className={`${styles.frame} lg`}
          style={photo ? ({ "--aspect": photo.width / photo.height } as React.CSSProperties) : undefined}
          data-photo={!!photo}
        >
          <div
            className={styles.screen}
            data-over={over}
            onDragOver={(event) => {
              event.preventDefault();
              setOver(true);
            }}
            onDragLeave={() => setOver(false)}
            onDrop={drop}
          >
            {photo ? (
              <CrackReveal
                className={styles.canvas}
                photo={photo}
                mask={mask}
                scanning={working}
                mode={mode}
                run={run}
                label={result ? d.resultLabel(share) : d.photoLabel}
              />
            ) : (
              <div className={styles.empty}>
                <Upload className={styles.emptyIcon} />
                <p className={styles.emptyTitle}>{d.drop}</p>
                <button type="button" className={ui.primary} onClick={() => picker.current?.click()}>
                  {d.choose}
                </button>
              </div>
            )}

            {status && (
              <div className={`${styles.status} lg-clear`} data-error={phase === "error"}>
                <p>{status}</p>
                {phase === "model" && (
                  <span className={styles.meter} aria-hidden>
                    <span style={{ width: `${percent}%` }} />
                  </span>
                )}
                {phase === "error" && failure !== "image" && (
                  <button type="button" className={styles.retry} onClick={() => last.current && analyse(last.current)}>
                    {d.retry}
                  </button>
                )}
              </div>
            )}
          </div>

          <figcaption className={styles.bar}>
            {phase === "done" && result ? (
              <>
                <dl className={styles.figures}>
                  <div>
                    <dt>{d.shareLabel}</dt>
                    <dd className="mono">{share}%</dd>
                  </div>
                  <div>
                    <dt>{d.timeLabel}</dt>
                    <dd className="mono">{d.seconds(seconds)}</dd>
                  </div>
                </dl>
                {result.share > 0 && (
                  <Segmented
                    label={t.hero.viewsLabel}
                    value={mode}
                    onChange={choose}
                    options={(["photo", "trace", "highlight"] as const).map((m) => ({ value: m, label: d.views[m] }))}
                  />
                )}
              </>
            ) : (
              <p className={styles.note}>{note}</p>
            )}
            {photo && (
              <button
                type="button"
                className={`${ui.secondary} lg-clear ${styles.another}`}
                onClick={() => picker.current?.click()}
              >
                {d.another}
              </button>
            )}
          </figcaption>
        </figure>

        <div className={styles.samples}>
          <p id="detect-samples">{d.samples}</p>
          <ul aria-labelledby="detect-samples">
            {SAMPLES.map((url, i) => (
              <li key={url}>
                <button type="button" className={styles.sample} onClick={() => sample(url)} aria-label={d.sample(i + 1)}>
                  <Image src={url} alt="" width={192} height={108} sizes="96px" />
                </button>
              </li>
            ))}
          </ul>
        </div>

        <p className={styles.live} role="status" aria-live="polite">
          {phase === "done" && result ? d.resultLabel(share) : status}
        </p>
      </div>
    </section>
  );
}
