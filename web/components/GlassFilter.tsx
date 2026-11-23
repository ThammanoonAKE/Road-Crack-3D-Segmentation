import { GLASS_MAP } from "@/lib/glass-map";

/**
 * The refraction used by the .lg materials (backdrop-filter: url(#lg-refract)): the backdrop is displaced along
 * the outward normal near the rim, so the edge of each glass shape bends what is behind it while the middle stays
 * clear. Units are relative to each element's box. Browsers without url() in backdrop-filter keep plain blur.
 */
export function GlassFilter() {
  return (
    <svg width="0" height="0" style={{ position: "absolute" }} aria-hidden focusable="false">
      <filter
        id="lg-refract"
        x="0"
        y="0"
        width="1"
        height="1"
        filterUnits="objectBoundingBox"
        primitiveUnits="objectBoundingBox"
        colorInterpolationFilters="sRGB"
      >
        <feImage href={GLASS_MAP} x="0" y="0" width="1" height="1" preserveAspectRatio="none" result="map" />
        <feDisplacementMap in="SourceGraphic" in2="map" scale="0.035" xChannelSelector="R" yChannelSelector="G" />
      </filter>
    </svg>
  );
}

/** The photograph every page floats on: the eight highlighted reconstructions, blurred into colour. */
export function Ambient() {
  return (
    <div
      aria-hidden
      style={{
        position: "fixed",
        inset: 0,
        zIndex: -1,
        background:
          "radial-gradient(120% 80% at 50% 0%, rgb(0 0 0 / 0) 0%, rgb(0 0 0 / 0.55) 100%), url(/media/ambient.webp) center / cover no-repeat, #0b0b0d",
      }}
    />
  );
}
