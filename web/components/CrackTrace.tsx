"use client";

import trace from "@/lib/crack-trace.json";
import styles from "./CrackTrace.module.css";

interface Segment {
  d: string;
  r: number; // distance of the segment's midpoint from the image centre, 0 (centre) to 1 (corner)
  len: number; // length in source pixels
}

/**
 * The centre lines of the cracks labelled in section 2, traced from the reconstruction itself
 * (lib/crack-trace.json, made from the viewer screenshots). Segments draw in order of distance from the centre, so the
 * network appears to propagate outwards.
 */
export function CrackTrace({ className, run = 0, cover = false }: { className?: string; run?: number; cover?: boolean }) {
  const segments = trace.segments as Segment[];
  return (
    <svg
      key={run}
      className={`${styles.trace} ${className ?? ""}`}
      viewBox={`0 0 ${trace.width} ${trace.height}`}
      // "slice" matches an <img> drawn with object-fit: cover, so the trace stays on the cracks when cropped
      preserveAspectRatio={cover ? "xMidYMid slice" : "none"}
      aria-hidden
    >
      {segments.map((s, i) => (
        <path
          key={i}
          d={s.d}
          pathLength={1}
          style={
            {
              "--delay": `${(s.r * 1.6).toFixed(3)}s`,
              "--dur": `${Math.min(0.9, 0.25 + s.len / 700).toFixed(3)}s`,
            } as React.CSSProperties
          }
        />
      ))}
    </svg>
  );
}
