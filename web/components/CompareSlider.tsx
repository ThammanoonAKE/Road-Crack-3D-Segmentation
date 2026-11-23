"use client";

import Image from "next/image";
import { useCallback, useRef, useState } from "react";
import styles from "./CompareSlider.module.css";

interface Props {
  before: string;
  after: string;
  width: number;
  height: number;
  beforeLabel: string;
  afterLabel: string;
  label: string;
  hint: string;
}

/** Before/after comparison. The divider is a slider: drag it, click anywhere, or use the arrow keys. */
export function CompareSlider({ before, after, width, height, beforeLabel, afterLabel, label, hint }: Props) {
  const [pos, setPos] = useState(50);
  const box = useRef<HTMLDivElement>(null);

  const fromPointer = useCallback((clientX: number) => {
    const r = box.current?.getBoundingClientRect();
    if (!r) return;
    setPos(Math.min(100, Math.max(0, ((clientX - r.left) / r.width) * 100)));
  }, []);

  const onKey = (e: React.KeyboardEvent) => {
    const step = e.shiftKey ? 10 : 2;
    const next =
      e.key === "ArrowLeft" || e.key === "ArrowDown"
        ? pos - step
        : e.key === "ArrowRight" || e.key === "ArrowUp"
          ? pos + step
          : e.key === "Home"
            ? 0
            : e.key === "End"
              ? 100
              : null;
    if (next === null) return;
    e.preventDefault();
    setPos(Math.min(100, Math.max(0, next)));
  };

  return (
    <div className={styles.wrap}>
      <div
        ref={box}
        className={styles.box}
        style={{ aspectRatio: `${width} / ${height}`, "--pos": `${pos}%` } as React.CSSProperties}
        onPointerDown={(e) => {
          (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
          fromPointer(e.clientX);
        }}
        onPointerMove={(e) => {
          if (e.buttons) fromPointer(e.clientX);
        }}
      >
        <Image src={after} alt={afterLabel} fill sizes="(min-width: 1100px) 640px, 100vw" className={styles.img} />
        <div className={styles.before}>
          <Image src={before} alt={beforeLabel} fill sizes="(min-width: 1100px) 640px, 100vw" className={styles.img} />
        </div>
        <span className={`${styles.tag} ${styles.tagBefore} lg-clear`}>{beforeLabel}</span>
        <span className={`${styles.tag} ${styles.tagAfter} lg-clear`}>{afterLabel}</span>
        <div
          className={styles.handle}
          role="slider"
          tabIndex={0}
          aria-label={label}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(pos)}
          aria-valuetext={`${beforeLabel} ${Math.round(pos)}% · ${afterLabel} ${Math.round(100 - pos)}%`}
          onKeyDown={onKey}
        >
          <span className={`${styles.knob} lg-clear`} aria-hidden>
            <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M6 4 2 8l4 4M10 4l4 4-4 4" />
            </svg>
          </span>
        </div>
      </div>
      <p className={styles.hint}>{hint}</p>
    </div>
  );
}
