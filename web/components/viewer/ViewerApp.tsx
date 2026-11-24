"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowLeft, Collapse, Expand, Mark, Reset, Rotate, Sliders } from "../Icons";
import { LangSwitch } from "../LangSwitch";
import { useLang } from "../LangProvider";
import { MAX_CRACK_SHARE, SECTIONS, getMetrics, getSection, megabytes, pad2, share } from "@/lib/sections";
import { SplatController, hasWebGL2, type Stage } from "@/lib/splat-controller";
import ui from "../ui.module.css";
import styles from "./ViewerApp.module.css";

type Status = Stage | "error" | "nowebgl";

export function ViewerApp({ n }: { n: number }) {
  const section = getSection(n)!;
  const metrics = getMetrics(n);
  const { t } = useLang();
  const v = t.viewer;
  const router = useRouter();

  const app = useRef<HTMLDivElement>(null);
  const stage = useRef<HTMLDivElement>(null);
  const ctrl = useRef<SplatController | null>(null);

  const [status, setStatus] = useState<Status>("download");
  const [percent, setPercent] = useState(0);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);
  const [highlight, setHighlight] = useState(true);
  const [strength, setStrength] = useState(0.8);
  const [autoRotate, setAutoRotate] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);
  const [panelOpen, setPanelOpen] = useState(true);
  const [info, setInfo] = useState({ ss: 1, gpu: "" });

  // On a phone the panel starts closed: the reconstruction comes first.
  useEffect(() => {
    if (window.matchMedia("(max-width: 860px)").matches) setPanelOpen(false);
  }, []);

  // Create the viewer for this section; dispose it when the route is left or hidden.
  useEffect(() => {
    const root = stage.current;
    if (!root) return;
    if (!hasWebGL2()) {
      setStatus("nowebgl");
      return;
    }
    const c = new SplatController();
    ctrl.current = c;
    let live = true;
    setStatus("download");
    setPercent(0);
    c.load(root, n, (s, p) => {
      if (!live) return;
      setStatus(s);
      setPercent(p);
    })
      .then(() => {
        if (live) setInfo({ ss: c.info.ss, gpu: c.info.gpu });
      })
      .catch((err: unknown) => {
        if (!live) return;
        setError(err instanceof Error ? err.message : String(err));
        setStatus("error");
      });
    return () => {
      live = false;
      c.dispose();
      if (ctrl.current === c) ctrl.current = null;
    };
  }, [n, attempt]);

  // For automated screenshots (web/tools/page_shot.py --ready).
  useEffect(() => {
    (window as { splatReady?: boolean }).splatReady = status === "ready";
  }, [status]);

  useEffect(() => {
    if (status === "ready") ctrl.current?.setHighlight(highlight ? strength : 0);
  }, [status, highlight, strength]);

  useEffect(() => {
    if (status === "ready") ctrl.current?.setAutoRotate(autoRotate);
  }, [status, autoRotate]);

  useEffect(() => {
    const onChange = () => setFullscreen(document.fullscreenElement === app.current);
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, []);

  const toggleFullscreen = useCallback(() => {
    if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
    else app.current?.requestFullscreen().catch(() => {});
  }, []);

  // Shortcuts: arrows orbit and tilt, +/- zoom, C highlight, R reset, A auto-rotate, F full screen, 1–8 section.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable)) return;
      const k = e.key.toLowerCase();
      const c = ctrl.current;
      if (k === "arrowleft") c?.nudge(-6, 0);
      else if (k === "arrowright") c?.nudge(6, 0);
      else if (k === "arrowup") c?.nudge(0, 4);
      else if (k === "arrowdown") c?.nudge(0, -4);
      else if (k === "+" || k === "=") c?.nudge(0, 0, 0.9);
      else if (k === "-" || k === "_") c?.nudge(0, 0, 1.1);
      else if (k === "c") setHighlight((h) => !h);
      else if (k === "r") c?.resetView();
      else if (k === "a") setAutoRotate((a) => !a);
      else if (k === "f") toggleFullscreen();
      else if (/^[1-8]$/.test(k) && Number(k) !== n) router.push(`/viewer/${k}`);
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [n, router, toggleFullscreen]);

  const mb = megabytes(section.bytes);
  const pct = share(section) * 100;
  const ready = status === "ready";

  return (
    <div ref={app} className={styles.app} data-panel={panelOpen}>
      <div ref={stage} className={styles.stage} aria-label={v.stage(n)} role="img" />

      <header className={styles.top}>
        <Link href="/" className={`${styles.back} lg-clear`} aria-label={v.back}>
          <ArrowLeft />
          <Mark className={styles.mark} />
        </Link>
        <nav className={`${styles.sections} lg-clear`} aria-label={v.sectionNav}>
          {SECTIONS.map((s) => (
            <Link
              key={s.n}
              href={`/viewer/${s.n}`}
              className="mono"
              aria-current={s.n === n ? "page" : undefined}
              aria-label={`${v.section} ${s.n}`}
            >
              {pad2(s.n)}
            </Link>
          ))}
        </nav>
        <LangSwitch className={styles.lang} />
      </header>

      <aside id="viewer-panel" className={`${styles.panel} lg-thick`} data-open={panelOpen} aria-label={v.data}>
        <div className={styles.panelHead}>
          <p className={`${styles.code} mono`}>SEC-{pad2(n)}</p>
          <h1 className={styles.title}>
            {v.section} {n}
          </h1>
          <p className={`${styles.clip} mono`}>{section.clip}</p>
        </div>

        <dl className={styles.group}>
          <div>
            <dt>{v.crack}</dt>
            <dd className="mono">
              <span className={styles.meter} aria-hidden>
                <span style={{ width: `${(100 * share(section)) / MAX_CRACK_SHARE}%` }} />
              </span>
              {pct.toFixed(1)}%
            </dd>
          </div>
          <div>
            <dt>{v.gaussians}</dt>
            <dd className="mono">{section.gaussians.toLocaleString("en-US")}</dd>
          </div>
          <div>
            <dt>{v.size}</dt>
            <dd className="mono">{mb} MB</dd>
          </div>
          <div>
            <dt>{v.render}</dt>
            <dd className="mono" title={info.gpu}>
              {ready ? `${+info.ss.toFixed(2)}× SS` : "—"}
            </dd>
          </div>
        </dl>

        {metrics && (
          <div className={styles.measure}>
            <dl className={styles.group}>
              <div>
                <dt>{v.surfaceArea}</dt>
                <dd className="mono">{metrics.surface_m2.toFixed(1)} m²</dd>
              </div>
              <div>
                <dt>{v.crackArea}</dt>
                <dd className="mono">
                  {metrics.crack_m2.toFixed(2)} m² · {(metrics.crack_share * 100).toFixed(1)}%
                </dd>
              </div>
              <div>
                <dt>{v.crackLength}</dt>
                <dd className="mono">{metrics.length_m.toFixed(1)} m</dd>
              </div>
              <div>
                <dt>{v.crackWidth}</dt>
                <dd className="mono">{metrics.width_mm} mm</dd>
              </div>
            </dl>
            <p className={styles.note}>{v.measureNote}</p>
          </div>
        )}

        <div className={styles.group}>
          <button
            type="button"
            role="switch"
            aria-checked={highlight}
            className={styles.row}
            onClick={() => setHighlight((h) => !h)}
            disabled={!ready}
          >
            <span className={styles.rowLabel}>
              <span className={styles.swatch} aria-hidden />
              {v.highlight}
            </span>
            <span className={ui.switch} aria-hidden />
          </button>
          <label className={styles.range}>
            <span>{v.strength}</span>
            <input
              type="range"
              min={0.2}
              max={1}
              step={0.05}
              value={strength}
              disabled={!ready || !highlight}
              onChange={(e) => setStrength(Number(e.target.value))}
              style={{ "--pct": `${((strength - 0.2) / 0.8) * 100}%` } as React.CSSProperties}
            />
            <span className="mono">{Math.round(strength * 100)}%</span>
          </label>
          <button
            type="button"
            role="switch"
            aria-checked={autoRotate}
            className={styles.row}
            onClick={() => setAutoRotate((a) => !a)}
            disabled={!ready}
          >
            <span className={styles.rowLabel}>
              <Rotate />
              {v.autoRotate}
            </span>
            <span className={ui.switch} aria-hidden />
          </button>
        </div>

        <dl className={`${styles.group} ${styles.keys}`}>
          <div>
            <dt>
              <kbd>←</kbd>
              <kbd>→</kbd>
              <kbd>↑</kbd>
              <kbd>↓</kbd>
            </dt>
            <dd>{v.orbitKeys}</dd>
          </div>
          <div>
            <dt>
              <kbd>+</kbd>
              <kbd>−</kbd>
            </dt>
            <dd>{v.zoomKeys}</dd>
          </div>
          <div>
            <dt>
              <kbd>1</kbd>–<kbd>8</kbd>
            </dt>
            <dd>{v.sectionKeys}</dd>
          </div>
        </dl>
      </aside>

      {/* floating toolbar, iOS tab-bar style */}
      <div className={`${styles.toolbar} lg`} role="toolbar" aria-label={v.controls}>
        <button
          type="button"
          className={styles.tool}
          aria-pressed={highlight}
          data-accent
          onClick={() => setHighlight((h) => !h)}
          disabled={!ready}
          title={`${v.highlight} (C)`}
        >
          <span className={styles.toolSwatch} aria-hidden />
          <span className={styles.toolText}>{v.highlight}</span>
        </button>
        <button
          type="button"
          className={styles.tool}
          aria-pressed={autoRotate}
          onClick={() => setAutoRotate((a) => !a)}
          disabled={!ready}
          aria-label={v.autoRotate}
          title={`${v.autoRotate} (A)`}
        >
          <Rotate />
        </button>
        <button
          type="button"
          className={styles.tool}
          onClick={() => ctrl.current?.resetView()}
          disabled={!ready}
          aria-label={v.reset}
          title={`${v.reset} (R)`}
        >
          <Reset />
        </button>
        <button
          type="button"
          className={styles.tool}
          onClick={toggleFullscreen}
          aria-label={fullscreen ? v.exitFullscreen : v.fullscreen}
          title={`${fullscreen ? v.exitFullscreen : v.fullscreen} (F)`}
        >
          {fullscreen ? <Collapse /> : <Expand />}
        </button>
        <span className={styles.divider} aria-hidden />
        <button
          type="button"
          className={styles.tool}
          aria-expanded={panelOpen}
          aria-controls="viewer-panel"
          onClick={() => setPanelOpen((o) => !o)}
          aria-label={panelOpen ? v.hidePanel : v.showPanel}
          title={panelOpen ? v.hidePanel : v.showPanel}
        >
          <Sliders />
        </button>
      </div>

      <p className={`${styles.hint} lg-clear`}>{v.hint}</p>

      {(status === "download" || status === "prepare") && (
        <div className={styles.overlay} role="status" aria-live="polite">
          <div className={`${styles.loader} lg`}>
            <p className={`${styles.code} mono`}>SEC-{pad2(n)}</p>
            <p className={styles.loadTitle}>{status === "download" ? v.downloading : v.preparing}</p>
            <div className={styles.progress} data-indeterminate={status === "prepare"}>
              <span style={{ width: `${status === "download" ? percent : 100}%` }} />
            </div>
            <p className={`${styles.loadFigures} mono`}>
              {status === "download"
                ? `${Math.round(percent)}% · ${Math.round((percent / 100) * mb)} / ${mb} MB`
                : `${mb} MB`}
            </p>
            <p className={styles.slow}>{v.slow(mb)}</p>
          </div>
        </div>
      )}

      {status === "error" && (
        <div className={styles.overlay} role="alert">
          <div className={`${styles.loader} lg`}>
            <p className={styles.loadTitle}>{v.errorTitle(n)}</p>
            <p className={styles.slow}>{v.errorBody}</p>
            {error && <p className={`${styles.errorCode} mono`}>{error}</p>}
            <button type="button" className={ui.primary} onClick={() => setAttempt((a) => a + 1)}>
              {v.retry}
            </button>
          </div>
        </div>
      )}

      {status === "nowebgl" && (
        <div className={styles.overlay} role="alert">
          <div className={`${styles.loader} lg`}>
            <p className={styles.slow}>{v.noWebgl}</p>
          </div>
        </div>
      )}
    </div>
  );
}
