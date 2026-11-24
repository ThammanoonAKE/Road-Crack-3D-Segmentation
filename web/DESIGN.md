# Design system: Road Crack 3D web

The site follows Apple's **Liquid Glass** language (iOS 26/27), in its dark appearance:
- the reconstructions are the content;
- controls and panels are glass that floats over the content and refracts it.

Crack red is iOS *systemRed*, so the brand accent and the data colour are the same thing.

## Tokens (`app/globals.css`)

| Group | Tokens | Notes |
|---|---|---|
| Labels | `--label` `#fff`, `--label-2` 64% white, `--label-3` 34% | `--label-2` measures ≥ 6.5:1 on the glass tints; `--label-3` is decorative only |
| Accent | `--red` `#ff453a` (marks, red text on dark), `--red-fill` `#d9302a` (button fills), `--red-fill-pressed` | White on `--red-fill` measures 4.77:1 |
| Fills | `--fill`, `--fill-2`, `--separator`, `--separator-strong` | iOS dark system fills and separators |
| Type | `--font-sans`, `--font-display`, `--font-mono`; sizes from `--t-caption` up to `--t-large` | See below |
| Shape | `--r-sm` 12, `--r-md` 20, `--r-lg` 28, `--r-xl` 36, `--r-capsule` | Every glass shape also sets `corner-shape: squircle` for continuous corners where the browser supports it |
| Motion | `--spring` (`cubic-bezier(0.32, 0.72, 0, 1)`), `--dur-fast` 160 ms, `--dur` 320 ms, `--dur-slow` 700 ms | See [Motion](#motion) |

**Type families:**
- `--font-sans` and `--font-display` use SF on Apple devices, with IBM Plex Sans Thai for Thai and on other
  systems.
- `--font-mono` is SF Mono or IBM Plex Mono, for codes and figures. It falls back to Plex Sans Thai for any
  Thai text.

**Type sizes:** `--t-caption` 12, `--t-footnote` 13, `--t-sub` 15, `--t-body` 17, `--t-title3` 20,
`--t-title2` 24, `--t-title1` 28–40 and `--t-large` 36–72.

## Glass materials

| Class | Use | Tint | Blur |
|---|---|---|---|
| `.lg-clear` | Small controls over imagery: chips, segmented controls, the slider knob | Smoke tint at 34%, plus a light text shadow so labels read over bright concrete | 4–6 px |
| `.lg` | Floating bars, cards, widgets | Dark tint at 42% | 18–22 px |
| `.lg-thick` | Panels with running text: the hero card, the method list, the viewer sheet | Dark tint at 66% | 30–34 px |

Each material combines:
- a top highlight gradient;
- an inner highlight;
- a soft drop shadow;
- a **specular rim**: a 1 px gradient ring drawn by `::after`. It is bright where light enters (top left)
  and exits (bottom right).

**Refraction.** In Chromium, `backdrop-filter: url(#lg-refract)` displaces the backdrop near each shape's edge.
- The filter is defined in `components/GlassFilter.tsx`, with its map in `lib/glass-map.ts`.
- The map is a rounded-rectangle bevel normal map. Its units are relative to each element's box.
- Other browsers ignore that declaration and keep plain blur.

**Accessibility fallbacks:**
- `prefers-reduced-transparency: reduce` turns all three materials into opaque `--bg-elevated`.
- `prefers-contrast: more` raises `--label-2` and turns the rim solid.

## Components

| Component | File | Notes |
|---|---|---|
| Floating navigation capsule | `SiteHeader` | Sticky; narrows once the page scrolls |
| Segmented control | `Segmented` | Glass track with a sliding glass thumb; buttons use `aria-pressed` |
| iOS switch | `ui.switch` | 51 × 31 px; red when on; the containing button has `role="switch"` |
| Capsule buttons | `ui.primary`, `ui.secondary` | Primary is tinted red glass; secondary is clear glass; both are 48 px high |
| Before/after slider | `CompareSlider` | `role="slider"`; arrow keys (Shift for ×5), Home/End; the knob is a glass lens |
| Live point cloud | `HeroCloud` | Section 2 as about 180,000 points (react-three-fiber, drei); the stage is dark until it forms. The hero photograph is only its fallback |
| Crack trace | `CrackTrace` | Centre lines traced from section 2, drawn over the hero photograph when the point cloud is not shown; static under reduced motion |
| Inset grouped list | `Method`, viewer sheet | iOS Settings-style rows with inset hairline separators |
| Floating toolbar | `viewer/ViewerApp` | Tab-bar capsule: highlight (labelled), auto-rotate, reset, full screen, panel |

## Motion

Motion shows the data or answers the pointer. Nothing moves only for decoration.

| Where | What moves | How |
|---|---|---|
| Hero | The road condenses out of dust, a scan line crosses it, the camera drifts and leans towards the mouse; in *Crack trace* the red spreads along the cracks from the middle | `HeroCloud`: one shader on a three.js point cloud, drei `Sparkles` for the dust |
| Sections of the home page | Each part plays an entrance as it scrolls into view. Kickers slide in from the left, titles come out of a blur, cards and panels rise one after another, the method steps arrive from the right with their numbers springing in, the Edge AI clip and its table come from opposite sides, the figure tiles zoom in, table rows follow their panel, and the red bars fill from the left | `data-reveal` rules in `globals.css`, started by `ScrollReveal` (an IntersectionObserver) |
| Section cards | A card leans towards the mouse and a highlight follows it | `SectionGrid`: CSS variables set on `pointermove` |
| Viewer | The camera glides down to its opening pose when a section has loaded | `SplatController.flyIn` |

**Fallbacks:**
- Under `prefers-reduced-motion: reduce` the hero keeps its photograph, entrances do not play, cards
  do not lean and the viewer opens on its pose.
- The point cloud is also skipped when the browser asks to save data or has no WebGL.
- Entrances hide nothing until `ScrollReveal` runs, so without JavaScript every element is in place. Elements
  already on screen when the page opens are left as they are.
- The hero stops drawing while it is scrolled out of view, and drops to 1× pixel density if frames slow down.

## Rules

**Red is for cracks, and for one primary action per view.** Nothing else uses it.

**Glass sits on content, not on glass.**
- Panels float over the photograph or the reconstruction.
- Avoid stacking more than two glass layers.

**Text-heavy surfaces use `.lg-thick`.** Clear glass is for short labels only.

**Touch targets are at least 44 px, focus is always visible:**
- Targets are 44 px high or more: toolbar buttons are 48 px and section links 42–44 px.
- Focus shows as a 2 px `--focus` ring.

**Every shortcut has a visible control, and every control has a label.** Icon-only buttons carry
`aria-label`.
