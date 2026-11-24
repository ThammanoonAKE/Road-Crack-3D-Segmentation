// Browser-only wrapper round the GaussianSplats3D viewer: loading with progress, supersampling, the crack
// highlight, auto-rotate and an animated reset. Ported from the original static viewer (index.html).

import type { Viewer } from "@mkkellogg/gaussian-splats-3d";
import { splatUrl } from "./sections";

export type Stage = "download" | "prepare" | "ready";

interface CameraFile {
  position: [number, number, number];
  target: [number, number, number];
  up: [number, number, number];
  gaussians: number;
  shDegree?: number;
}

const CRACK_RGB = [255, 40, 40];
const DIMMED = 0.45; // brightness left to the rest of the road when it is dimmed, as in the home page's highlight
const DOWNLOADING = 0; // LoaderStatus.Downloading in GaussianSplats3D 0.4.7 (not exported)

export function hasWebGL2(): boolean {
  try {
    return !!document.createElement("canvas").getContext("webgl2");
  } catch {
    return false;
  }
}

function gpuName(): string {
  try {
    const gl = document.createElement("canvas").getContext("webgl2") || document.createElement("canvas").getContext("webgl");
    if (!gl) return "";
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    const name: string = ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : "";
    const angle = name.match(/ANGLE \([^,]*,\s*(.*?)\s*\(0x/); // "ANGLE (vendor, device (0xid) Direct3D11 ...)"
    return angle ? angle[1].trim() : name;
  } catch {
    return "";
  }
}

/** Render at ss times the screen's pixel density. On a discrete GPU, the most that keeps the drawing buffer near
 *  32 M pixels (4x on a 1080p screen); on an integrated one, 2x for small scenes and 1x for large ones. */
function supersampling(gaussians: number, gpu: string, ratio: number, w: number, h: number): number {
  const widest = 16384 / (w * ratio);
  if (/NVIDIA|GeForce|Quadro|Radeon (RX|Pro)|Arc\(TM\) A/i.test(gpu)) {
    return Math.max(1, Math.min(4, widest, Math.floor(Math.sqrt(32e6 / (w * h)) / ratio)));
  }
  return gaussians < 1.5e6 ? 2 : 1;
}

export class SplatController {
  private viewer: Viewer | null = null;
  private disposed = false;
  private indices: Uint32Array | null = null;
  private base: Uint8Array | null = null; // every Gaussian's colour as loaded, rgba
  private isCrack: Uint8Array | null = null;
  private dimmed = false;
  private home: CameraFile | null = null;
  private resetFrame = 0;
  info = { ss: 1, gpu: "", count: 0, labelled: 0 };

  async load(root: HTMLElement, n: number, onStage: (stage: Stage, percent: number) => void): Promise<void> {
    const GS = await import("@mkkellogg/gaussian-splats-3d");
    const response = await fetch(splatUrl(n, "json"));
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    const camera: CameraFile = await response.json();
    if (this.disposed) return;
    this.home = camera;

    const ratio = window.devicePixelRatio || 1;
    const gpu = gpuName();
    const ss = supersampling(camera.gaussians, gpu, ratio, root.clientWidth || innerWidth, root.clientHeight || innerHeight);
    this.info = { ...this.info, ss, gpu };

    // The viewer reads window.devicePixelRatio once, in its constructor: override it for that call only.
    const original = Object.getOwnPropertyDescriptor(window, "devicePixelRatio");
    Object.defineProperty(window, "devicePixelRatio", { configurable: true, get: () => ss * ratio });
    try {
      this.viewer = new GS.Viewer({
        rootElement: root,
        cameraUp: camera.up,
        initialCameraPosition: camera.position,
        initialCameraLookAt: camera.target,
        // A plain static host sends no cross-origin isolation headers, so no SharedArrayBuffer.
        sharedMemoryForWorkers: false,
        gpuAcceleratedSort: false,
        // Zoomed out past the photographs' scale, Gaussians fade instead of spreading to a blur.
        antialiased: true,
        sphericalHarmonicsDegree: camera.shDegree || 0,
        sceneRevealMode: GS.SceneRevealMode.Instant,
        maxScreenSpaceSplatSize: 1024 * ss,
      });
    } finally {
      if (original) Object.defineProperty(window, "devicePixelRatio", original);
      else delete (window as { devicePixelRatio?: number }).devicePixelRatio;
    }
    // The library binds its own debugging keys on window (F changes the focal length, C shows a cursor,
    // arrows roll the camera); this app has its own shortcuts.
    if (this.viewer.keyDownListener) window.removeEventListener("keydown", this.viewer.keyDownListener);

    onStage("download", 0);
    await this.viewer.addSplatScene(splatUrl(n, "ksplat"), {
      showLoadingUI: false,
      progressiveLoad: false,
      // The library does not export its LoaderStatus enum; 0 is Downloading, then Processing (1) and Done (2).
      // An exception thrown here would abort the download, so this callback only records state.
      onProgress: (percent, _label, status) =>
        status === DOWNLOADING ? onStage("download", Number(percent) || 0) : onStage("prepare", 100),
    });
    if (this.disposed) return;
    this.viewer.start();
    this.info.count = this.viewer.splatMesh.getSplatCount();
    await this.loadCracks(n);
    if (this.disposed) return;
    if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) this.flyIn();
    onStage("ready", 100);
  }

  /** <name>.cracks.bin: uint32 indices, in the viewer's order, of the Gaussians labelled as crack. */
  private async loadCracks(n: number) {
    const response = await fetch(splatUrl(n, "cracks.bin"));
    if (!response.ok || !this.viewer) return;
    const indices = new Uint32Array(await response.arrayBuffer());
    const count = this.viewer.splatMesh.getSplatCount();
    if (!indices.length || indices[indices.length - 1] >= count) {
      console.warn(`hole_${n}.cracks.bin does not fit this reconstruction (${count} Gaussians)`);
      return;
    }
    const isCrack = new Uint8Array(count);
    for (let k = 0; k < indices.length; k++) isCrack[indices[k]] = 1;
    this.indices = indices;
    this.isCrack = isCrack;
    this.base = this.viewer.splatMesh.getSplatDataTextures().baseData.colors.slice(0, count * 4);
    this.info.labelled = indices.length;
  }

  get canHighlight() {
    return !!this.indices;
  }

  /** Tint the crack Gaussians towards red by `strength` (0 = off) in the viewer's own colour data, and with `dim`
   *  darken every other Gaussian so that the cracks stand out. */
  setHighlight(strength: number, dim = false) {
    if (!this.viewer || !this.indices || !this.base || !this.isCrack) return;
    const textures = this.viewer.splatMesh.getSplatDataTextures();
    const colors = textures.baseData.colors;
    const packed = textures.centerColors.data;
    const { indices, base, isCrack } = this;
    // As the library packs a colour (rgbaArrayToInteger): R | G << 8 | B << 16 | A << 24.
    const pack = (o: number) => colors[o] | (colors[o + 1] << 8) | (colors[o + 2] << 16) | (colors[o + 3] << 24);
    for (let k = 0; k < indices.length; k++) {
      const o = indices[k] * 4;
      for (let c = 0; c < 3; c++) colors[o + c] = Math.round(base[o + c] * (1 - strength) + CRACK_RGB[c] * strength);
      packed[o] = pack(o);
    }
    // The rest of the road is millions of Gaussians: it is rewritten only when the dimming is switched.
    if (dim !== this.dimmed) {
      this.dimmed = dim;
      const keep = dim ? DIMMED : 1;
      for (let i = 0; i < isCrack.length; i++) {
        if (isCrack[i]) continue;
        const o = i * 4;
        for (let c = 0; c < 3; c++) colors[o + c] = Math.round(base[o + c] * keep);
        packed[o] = pack(o);
      }
    }
    textures.centerColors.texture.needsUpdate = true;
  }

  setAutoRotate(on: boolean) {
    const c = this.viewer?.controls;
    if (!c) return;
    c.autoRotate = on;
    c.autoRotateSpeed = 0.6;
  }

  /** Keyboard orbit: turn about the target by dAz degrees round the up axis (+Y in every reconstruction here),
   *  tilt by dEl degrees, and scale the distance by `zoom`. */
  nudge(dAz: number, dEl: number, zoom = 1) {
    const v = this.viewer;
    if (!v || !v.controls) return;
    cancelAnimationFrame(this.resetFrame);
    const p = v.camera.position;
    const t = v.controls.target;
    const x = p.x - t.x, y = p.y - t.y, z = p.z - t.z;
    const dist = Math.hypot(x, y, z);
    const r = Math.min(dist * 6, Math.max(dist / 6, dist * zoom));
    const theta = Math.atan2(x, z) + (dAz * Math.PI) / 180;
    const phi = Math.min(Math.PI * 0.49, Math.max(0.05, Math.acos(y / dist) - (dEl * Math.PI) / 180));
    p.set(
      t.x + r * Math.sin(phi) * Math.sin(theta),
      t.y + r * Math.cos(phi),
      t.z + r * Math.sin(phi) * Math.cos(theta),
    );
    v.controls.update();
  }

  /** Start a quarter-turn round, higher and farther out, and glide down to the opening pose. */
  flyIn(ms = 1800) {
    this.nudge(55, 22, 1.6);
    this.resetView(ms);
  }

  /** Ease the camera back to the opening pose over `ms`. */
  resetView(ms = 700) {
    const v = this.viewer;
    const home = this.home;
    if (!v || !v.controls || !home) return;
    cancelAnimationFrame(this.resetFrame);
    const p0 = [v.camera.position.x, v.camera.position.y, v.camera.position.z];
    const t0 = [v.controls.target.x, v.controls.target.y, v.controls.target.z];
    const start = performance.now();
    const ease = (x: number) => 1 - Math.pow(1 - x, 3);
    const step = (now: number) => {
      if (!this.viewer || !this.viewer.controls) return;
      const a = ease(Math.min(1, (now - start) / ms));
      const lerp = (from: number[], to: number[]) => from.map((f, i) => f + (to[i] - f) * a) as [number, number, number];
      this.viewer.camera.position.set(...lerp(p0, home.position));
      this.viewer.controls.target.set(...lerp(t0, home.target));
      this.viewer.camera.up.set(...home.up);
      this.viewer.controls.update();
      if (a < 1) this.resetFrame = requestAnimationFrame(step);
    };
    this.resetFrame = requestAnimationFrame(step);
  }

  dispose() {
    this.disposed = true;
    cancelAnimationFrame(this.resetFrame);
    const v = this.viewer;
    this.viewer = null;
    // dispose() ends by removing its root from document.body, which throws for a root inside React's tree;
    // everything that matters is released before that line.
    v?.dispose().catch(() => {});
  }
}
