// The eight reconstructed road sections. Figures come from the files in public/splats:
// gaussians = the .json "gaussians" field, crack = the number of indices in the .cracks.bin, bytes = the .ksplat size.

import measured from "./crack-metrics.json";

export type Refinement = "top" | "tiles4k";

/** A section's cracks measured on the road plane (tools/crack_metrics.mjs). Metres assume the phone was held
 *  1.5 m above the road; crack_share, crack area over surveyed surface, does not depend on that. */
export interface CrackMetrics {
  surface_m2: number;
  crack_m2: number;
  crack_share: number;
  length_m: number;
  /** Mean width of the labelled band (crack area / crack length), wider than the crack's opening. */
  width_mm: number;
}

export function getMetrics(n: number): CrackMetrics | undefined {
  return (measured.sections as Record<string, CrackMetrics>)[String(n)];
}

export interface Section {
  n: number;
  clip: string;
  gaussians: number;
  crack: number;
  bytes: number;
  refinement: Refinement;
}

export const SECTIONS: readonly Section[] = [
  { n: 1, clip: "VID_20261102_151605", gaussians: 3_617_853, crack: 226_076, bytes: 79_760_148, refinement: "top" },
  { n: 2, clip: "VID_20261102_151720", gaussians: 4_916_086, crack: 203_949, bytes: 110_233_092, refinement: "tiles4k" },
  { n: 3, clip: "VID_20261102_151928", gaussians: 3_976_875, crack: 221_939, bytes: 87_837_900, refinement: "top" },
  { n: 4, clip: "VID_20261102_152216", gaussians: 4_797_081, crack: 341_327, bytes: 106_909_188, refinement: "top" },
  { n: 5, clip: "VID_20261102_152251", gaussians: 4_155_981, crack: 95_624, bytes: 90_196_884, refinement: "top" },
  { n: 6, clip: "VID_20261102_152326", gaussians: 4_252_470, crack: 198_696, bytes: 96_458_760, refinement: "top" },
  { n: 7, clip: "VID_20261102_152410", gaussians: 6_276_884, crack: 379_753, bytes: 90_914_264, refinement: "top" },
  { n: 8, clip: "VID_20261102_152618", gaussians: 4_165_973, crack: 135_303, bytes: 91_921_092, refinement: "top" },
];

export const MAX_CRACK_SHARE = Math.max(...SECTIONS.map((s) => s.crack / s.gaussians));

export function getSection(n: number): Section | undefined {
  return SECTIONS.find((s) => s.n === n);
}

/** Where the .ksplat / .json / .cracks.bin files are served from. Defaults to this app's /splats; set
 *  NEXT_PUBLIC_SPLAT_BASE_URL to serve them from object storage or a CDN instead (e.g. when deploying to Vercel). */
export const SPLAT_BASE = (process.env.NEXT_PUBLIC_SPLAT_BASE_URL || "/splats").replace(/\/$/, "");

export function splatUrl(n: number, ext: "ksplat" | "json" | "cracks.bin"): string {
  return `${SPLAT_BASE}/hole_${n}.${ext}`;
}

export const pad2 = (n: number) => String(n).padStart(2, "0");
export const megabytes = (bytes: number) => Math.round(bytes / 1e6);
export const share = (s: Section) => s.crack / s.gaussians;
