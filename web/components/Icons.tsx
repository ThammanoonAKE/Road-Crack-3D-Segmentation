// Hand-drawn 16 px line icons (1.5 px stroke), so the set stays consistent with the hairline style.

type P = React.SVGProps<SVGSVGElement>;
const base = {
  viewBox: "0 0 16 16",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.5,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
};

export const ArrowRight = (p: P) => (
  <svg {...base} {...p}>
    <path d="M2.5 8h11M9.5 4l4 4-4 4" />
  </svg>
);

export const ArrowLeft = (p: P) => (
  <svg {...base} {...p}>
    <path d="M13.5 8h-11M6.5 4l-4 4 4 4" />
  </svg>
);

export const Cube = (p: P) => (
  <svg {...base} {...p}>
    <path d="M8 1.75 13.75 5v6L8 14.25 2.25 11V5L8 1.75Z" />
    <path d="M2.25 5 8 8.25 13.75 5M8 8.25v6" />
  </svg>
);

export const Reset = (p: P) => (
  <svg {...base} {...p}>
    <path d="M2.75 8a5.25 5.25 0 1 0 1.6-3.78" />
    <path d="M2.5 2.5v3.25h3.25" />
  </svg>
);

export const Rotate = (p: P) => (
  <svg {...base} {...p}>
    <ellipse cx="8" cy="8" rx="6.25" ry="2.75" />
    <path d="m11.5 3.9 1.6 1.4-1.9.95" />
  </svg>
);

export const Expand = (p: P) => (
  <svg {...base} {...p}>
    <path d="M2.5 6V2.5H6M10 2.5h3.5V6M13.5 10v3.5H10M6 13.5H2.5V10" />
  </svg>
);

export const Collapse = (p: P) => (
  <svg {...base} {...p}>
    <path d="M6 2.5V6H2.5M13.5 6H10V2.5M10 13.5V10h3.5M2.5 10H6v3.5" />
  </svg>
);

export const Sliders = (p: P) => (
  <svg {...base} {...p}>
    <path d="M2.5 4.5h11M2.5 11.5h11" />
    <circle cx="6" cy="4.5" r="1.6" fill="currentColor" stroke="none" />
    <circle cx="10.5" cy="11.5" r="1.6" fill="currentColor" stroke="none" />
  </svg>
);

export const Play = (p: P) => (
  <svg {...base} {...p}>
    <path d="M5 3.25v9.5L12.5 8 5 3.25Z" />
  </svg>
);

export const Pause = (p: P) => (
  <svg {...base} {...p}>
    <path d="M5.5 3.5v9M10.5 3.5v9" />
  </svg>
);

export const Upload = (p: P) => (
  <svg {...base} {...p}>
    <path d="M8 10.5v-8M4.75 5.5 8 2.25l3.25 3.25M2.5 10.5v3h11v-3" />
  </svg>
);

export const External = (p: P) => (
  <svg {...base} {...p}>
    <path d="M9.5 2.5h4v4M13.5 2.5 7.5 8.5M12 9.5v4H2.5V4h4" />
  </svg>
);

/** The mark: a survey frame with one crack running through it. */
export const Mark = (p: P) => (
  <svg viewBox="0 0 20 20" fill="none" aria-hidden {...p}>
    <path d="M1 6V1h5M14 1h5v5M19 14v5h-5M6 19H1v-5" stroke="currentColor" strokeWidth="1.5" />
    <path
      d="M4.5 13.5 7.5 11l1.5 1.2L12 8.4l1.3.8 2.4-3.4"
      stroke="var(--red)"
      strokeWidth="1.75"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);
