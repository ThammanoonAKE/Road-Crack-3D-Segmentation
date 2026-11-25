"use client";

import { useEffect, useRef } from "react";

export type RevealMode = "photo" | "trace" | "highlight";
export interface MaskData {
  data: Uint8Array; // width x height, 255 = crack
  width: number;
  height: number;
}
type Picture = HTMLCanvasElement | HTMLImageElement | ImageBitmap;

/**
 * A photograph with its detected cracks, shown the way the hero shows section 2:
 *   - while `scanning`, a bright line sweeps down the photo;
 *   - when the mask arrives (and each time `run` changes) the red spreads along the cracks from the middle, with a
 *     bright front, and then keeps pulsing;
 *   - "highlight" shows every crack at once and dims the rest, "photo" hides them.
 * One fragment shader on a WebGL canvas, with the colours and timings of HeroCloud. Under reduced motion the result
 * appears at once and nothing moves; without WebGL it is painted once on a 2D canvas.
 */

const VERTEX = /* glsl */ `
attribute vec2 aCorner;
varying vec2 vUv;
void main() {
  vUv = vec2(aCorner.x * 0.5 + 0.5, 0.5 - aCorner.y * 0.5); // row 0 of the image at the top
  gl_Position = vec4(aCorner, 0.0, 1.0);
}
`;

const FRAGMENT = /* glsl */ `
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
uniform sampler2D uPhoto;
uniform sampler2D uMask;
uniform vec2 uAspect; // turns uv - 0.5 into a distance that is 1 at the corners
uniform float uFront; // how far the red has spread from the middle
uniform float uCrack; // 0..1: how red the cracks are
uniform float uDim;   // 0..1: how far the rest of the photo is dimmed
uniform float uPhase; // the pulse that runs along the cracks
uniform vec2 uScan;   // the scan line: position down the photo, and its strength
varying vec2 vUv;

void main() {
  vec3 photo = texture2D(uPhoto, vUv).rgb;
  float r = length((vUv - 0.5) * uAspect);
  float crack = smoothstep(0.35, 0.65, texture2D(uMask, vUv).r);

  float lit = crack * uCrack * (1.0 - smoothstep(uFront - 0.06, uFront, r));
  float front = uCrack * exp(-pow((r - uFront) / 0.045, 2.0));
  float pulse = 0.5 + 0.5 * sin(r * 26.0 - uPhase);
  float band = uScan.y * exp(-pow((vUv.y - uScan.x) / 0.02, 2.0));

  vec3 base = photo * (1.0 - uDim * 0.55 * (1.0 - crack)) + band * 0.3;
  vec3 red = vec3(1.0, 0.271, 0.227) * (0.82 + 0.3 * pulse);
  // the front is bright on the cracks and only a faint ring on the road between them
  vec3 colour = mix(base, red, lit * 0.94) + front * (crack * vec3(1.0, 0.62, 0.5) + 0.045);
  gl_FragColor = vec4(colour, 1.0);
}
`;

const SPEED = 0.5; // how fast the front travels, in corner distances per second (the hero's speed)
const SWEEP = 2.4; // seconds for the scan line to cross the photo

interface Scene {
  gl: WebGLRenderingContext;
  uniforms: Record<string, WebGLUniformLocation | null>;
  photo: WebGLTexture;
  mask: WebGLTexture;
}

function texture(gl: WebGLRenderingContext, unit: number): WebGLTexture {
  const t = gl.createTexture()!;
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, 1, 1, 0, gl.LUMINANCE, gl.UNSIGNED_BYTE, new Uint8Array([0]));
  return t;
}

function build(canvas: HTMLCanvasElement): Scene | null {
  const gl = canvas.getContext("webgl", { alpha: false, antialias: false, powerPreference: "low-power" });
  if (!gl) return null;
  const program = gl.createProgram()!;
  for (const [type, source] of [
    [gl.VERTEX_SHADER, VERTEX],
    [gl.FRAGMENT_SHADER, FRAGMENT],
  ] as const) {
    const shader = gl.createShader(type)!;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    gl.attachShader(program, shader);
  }
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) return null;
  gl.useProgram(program);
  gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const corner = gl.getAttribLocation(program, "aCorner");
  gl.enableVertexAttribArray(corner);
  gl.vertexAttribPointer(corner, 2, gl.FLOAT, false, 0, 0);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  const uniforms: Scene["uniforms"] = {};
  for (const name of ["uPhoto", "uMask", "uAspect", "uFront", "uCrack", "uDim", "uPhase", "uScan"]) {
    uniforms[name] = gl.getUniformLocation(program, name);
  }
  const scene = { gl, uniforms, photo: texture(gl, 0), mask: texture(gl, 1) };
  gl.uniform1i(uniforms.uPhoto, 0);
  gl.uniform1i(uniforms.uMask, 1);
  return scene;
}

const isData = (mask: Picture | MaskData): mask is MaskData => "data" in mask;

/** Without WebGL: the photo with its cracks painted red, once. */
function paint(canvas: HTMLCanvasElement, photo: Picture | null, mask: Picture | MaskData | null, mode: RevealMode) {
  const context = canvas.getContext("2d");
  if (!context || !photo) return;
  context.drawImage(photo, 0, 0, canvas.width, canvas.height);
  if (!mask || mode === "photo") return;
  const layer = document.createElement("canvas");
  layer.width = mask.width;
  layer.height = mask.height;
  const lc = layer.getContext("2d");
  if (!lc) return;
  let alpha: Uint8Array | Uint8ClampedArray;
  let step = 1;
  if (isData(mask)) alpha = mask.data;
  else {
    lc.drawImage(mask, 0, 0);
    alpha = lc.getImageData(0, 0, layer.width, layer.height).data;
    step = 4;
  }
  const red = lc.createImageData(layer.width, layer.height);
  for (let i = 0; i < layer.width * layer.height; i++) {
    red.data[i * 4] = 255;
    red.data[i * 4 + 1] = 69;
    red.data[i * 4 + 2] = 58;
    red.data[i * 4 + 3] = alpha[i * step] > 127 ? 240 : 0;
  }
  lc.putImageData(red, 0, 0);
  context.drawImage(layer, 0, 0, canvas.width, canvas.height);
}

interface Props {
  photo: Picture | null;
  mask: Picture | MaskData | null;
  scanning?: boolean;
  mode?: RevealMode;
  /** Change it to play the spread again. */
  run?: number;
  /** Play the spread again each time the canvas scrolls into view. */
  replayInView?: boolean;
  label: string;
  className?: string;
}

export function CrackReveal({ photo, mask, scanning = false, mode = "trace", run = 0, replayInView = false, label, className }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const scene = useRef<Scene | null>(null);
  const flat = useRef(false); // no WebGL: painted on a 2D canvas
  const still = useRef(false); // reduced motion
  const state = useRef({
    photo: photo as Picture | null,
    mask: mask as Picture | MaskData | null,
    mode,
    scanning,
    front: 2,
    crack: 0,
    dim: 0,
    scan: 0,
    scanAt: 0,
    visible: true,
    frame: 0,
    last: 0,
  });
  const kick = useRef(() => {});

  useEffect(() => {
    const el = canvas.current;
    if (!el) return;
    const s = state.current;
    still.current = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    scene.current = build(el);
    flat.current = !scene.current;

    const draw = (now: number) => {
      const dt = Math.min((now - s.last) / 1000, 0.1);
      s.last = now;
      const moving = !still.current;
      const ease = (value: number, target: number) => (moving ? value + (target - value) * (1 - Math.exp(-5 * dt)) : target);
      s.crack = ease(s.crack, s.mask && s.mode !== "photo" ? 1 : 0);
      s.dim = ease(s.dim, s.mask && s.mode === "highlight" ? 1 : 0);
      s.scan = ease(s.scan, s.scanning && moving ? 1 : 0);
      if (moving) {
        s.front = Math.min(2, s.front + dt * SPEED);
        s.scanAt = (s.scanAt + dt / SWEEP) % 1;
      } else s.front = 2;

      const sc = scene.current;
      if (!sc) {
        paint(el, s.photo, s.mask, s.mode);
        return;
      }
      const { gl, uniforms: u } = sc;
      gl.viewport(0, 0, el.width, el.height);
      const diagonal = Math.hypot(el.width, el.height) || 1;
      gl.uniform2f(u.uAspect, (2 * el.width) / diagonal, (2 * el.height) / diagonal);
      gl.uniform1f(u.uFront, s.front);
      gl.uniform1f(u.uCrack, s.crack);
      gl.uniform1f(u.uDim, s.dim);
      gl.uniform1f(u.uPhase, moving ? ((now / 1000) * 2.2) % (Math.PI * 2) : 0);
      gl.uniform2f(u.uScan, -0.1 + 1.2 * s.scanAt, s.scan);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    };

    const loop = (now: number) => {
      s.frame = 0;
      draw(now);
      // the pulse runs for as long as cracks are shown; without them the loop stops once everything has settled
      const busy = !still.current && !flat.current && s.visible && s.photo && (s.mask ? s.mode !== "photo" || s.crack > 0.004 : s.scanning || s.scan > 0.004);
      if (busy) s.frame = requestAnimationFrame(loop);
    };
    kick.current = () => {
      if (s.frame) return;
      s.last = performance.now();
      s.frame = requestAnimationFrame(loop);
    };

    // the drawing buffer follows the size on screen, up to the photo's own size
    const fit = () => {
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(1, Math.round(el.clientWidth * ratio));
      const h = Math.max(1, Math.round(el.clientHeight * ratio));
      if (el.width !== w || el.height !== h) {
        el.width = w;
        el.height = h;
      }
      kick.current();
    };
    const resized = new ResizeObserver(fit);
    resized.observe(el);
    const seen = new IntersectionObserver(([entry]) => {
      const entered = entry.isIntersecting && !s.visible;
      s.visible = entry.isIntersecting;
      if (entered && replayInView && s.mask && s.mode === "trace") s.front = 0;
      if (s.visible) kick.current();
    });
    seen.observe(el);
    return () => {
      resized.disconnect();
      seen.disconnect();
      cancelAnimationFrame(s.frame);
      s.frame = 0;
      scene.current = null; // the context goes with the canvas; a remount of the same canvas builds on it again
    };
  }, [replayInView]);

  useEffect(() => {
    const s = state.current;
    s.photo = photo;
    const sc = scene.current;
    if (sc && photo) {
      sc.gl.activeTexture(sc.gl.TEXTURE0);
      sc.gl.bindTexture(sc.gl.TEXTURE_2D, sc.photo);
      sc.gl.texImage2D(sc.gl.TEXTURE_2D, 0, sc.gl.RGB, sc.gl.RGB, sc.gl.UNSIGNED_BYTE, photo);
    }
    kick.current();
  }, [photo]);

  useEffect(() => {
    const s = state.current;
    s.mask = mask;
    const sc = scene.current;
    if (sc) {
      const { gl } = sc;
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, sc.mask);
      if (!mask) gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, 1, 1, 0, gl.LUMINANCE, gl.UNSIGNED_BYTE, new Uint8Array([0]));
      else if (isData(mask)) {
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, mask.width, mask.height, 0, gl.LUMINANCE, gl.UNSIGNED_BYTE, mask.data);
      } else gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, gl.LUMINANCE, gl.UNSIGNED_BYTE, mask);
    }
    kick.current();
  }, [mask]);

  // "trace" replays the spread each time it is chosen or a new result arrives; the other modes show it complete
  useEffect(() => {
    const s = state.current;
    s.mode = mode;
    s.front = mode === "trace" && mask ? 0 : 2;
    kick.current();
  }, [mode, run, mask]);

  useEffect(() => {
    state.current.scanning = scanning;
    kick.current();
  }, [scanning]);

  return <canvas ref={canvas} className={className} role="img" aria-label={label} />;
}
