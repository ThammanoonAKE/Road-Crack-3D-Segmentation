"use client";

import { useLayoutEffect, useRef, useState } from "react";
import ui from "./ui.module.css";

interface Props<T extends string> {
  options: readonly { value: T; label: React.ReactNode; lang?: string }[];
  value: T;
  onChange: (value: T) => void;
  label: string;
  className?: string;
}

/** iOS segmented control: a glass track with a glass thumb that slides to the selected option. */
export function Segmented<T extends string>({ options, value, onChange, label, className }: Props<T>) {
  const track = useRef<HTMLDivElement>(null);
  const [thumb, setThumb] = useState({ x: 0, w: 0 });

  useLayoutEffect(() => {
    const el = track.current?.querySelector<HTMLButtonElement>('button[aria-pressed="true"]');
    if (el) setThumb({ x: el.offsetLeft - 3, w: el.offsetWidth });
  }, [value, options]);

  return (
    <div ref={track} role="group" aria-label={label} className={`${ui.segmented} lg-clear ${className ?? ""}`}>
      <span className={ui.segThumb} style={{ width: thumb.w, transform: `translateX(${thumb.x}px)` }} aria-hidden />
      {options.map((o) => (
        <button key={o.value} type="button" aria-pressed={o.value === value} lang={o.lang} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}
