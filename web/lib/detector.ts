/**
 * The crack detector of the /detect page. The model runs in a worker in the visitor's own browser
 * (public/detect/worker.js): a photo is turned into pixels here, the worker returns a mask, and nothing is uploaded.
 */

/** The model file edge_ai/onnx_web.py writes. Next to the site when run locally; the deployed site names a bucket. */
export const DETECTOR_URL = process.env.NEXT_PUBLIC_DETECTOR_URL || "/models/deepcrack_1class_2026-11-20.web.onnx";
/** Its size, for the note shown before the first download. */
export const DETECTOR_MB = 110;
/** Photos are shrunk to this many pixels on their long side, as the Edge AI pipeline does with video frames. */
export const LONG_SIDE = 960;

export interface Detection {
  mask: Uint8Array; // width x height, 255 = crack
  width: number;
  height: number;
  share: number; // crack pixels / all pixels
  ms: number;
  backend: "webgpu" | "wasm";
}

type Message =
  | { type: "progress"; loaded: number; total: number }
  | { type: "ready"; backend: string }
  | ({ type: "result"; id: number } & Detection)
  | { type: "error"; id?: number; stage: "load" | "detect"; message: string };

let worker: Worker | null = null;
let ready: Promise<void> | null = null;
let opened = false;
let report: ((share: number) => void) | null = null;
let nextId = 1;
const waiting = new Map<number, { resolve: (found: Detection) => void; reject: (error: Error) => void }>();

function start(): Promise<void> {
  const w = new Worker("/detect/worker.js", { type: "module" });
  worker = w;
  return new Promise((resolve, reject) => {
    const fail = (message: string) => {
      const error = new Error(message);
      for (const job of waiting.values()) job.reject(error);
      waiting.clear();
      reject(error);
    };
    w.onmessage = ({ data }: MessageEvent<Message>) => {
      if (data.type === "progress") report?.(data.total ? data.loaded / data.total : 0);
      else if (data.type === "ready") resolve();
      else if (data.type === "result") {
        waiting.get(data.id)?.resolve(data);
        waiting.delete(data.id);
      } else if (data.stage === "load") fail(data.message);
      else if (data.id !== undefined) {
        waiting.get(data.id)?.reject(new Error(data.message));
        waiting.delete(data.id);
      }
    };
    w.onerror = (event) => fail(event.message || "the detector could not start");
    w.postMessage({ type: "load", model: new URL(DETECTOR_URL, location.href).href });
  });
}

/** True once the model is in memory: the next detection starts at once. */
export const detectorReady = () => opened;

/**
 * Downloads the model (once: the browser keeps it) and opens it. Safe to call again; after a failure the next call
 * starts over. `progress` receives the downloaded share, 0 to 1.
 */
export function loadDetector(progress?: (share: number) => void): Promise<void> {
  report = progress ?? null;
  ready ??= start().then(
    () => {
      opened = true;
    },
    (error) => {
      worker?.terminate();
      worker = null;
      ready = null;
      throw error;
    },
  );
  return ready;
}

/** Runs the model on pixels already shrunk to LONG_SIDE; `scale` is how far they were shrunk (1 = not at all). */
export function detect(pixels: ImageData, scale: number): Promise<Detection> {
  const w = worker;
  if (!w || !opened) return Promise.reject(new Error("the detector is not loaded"));
  const id = nextId++;
  return new Promise((resolve, reject) => {
    waiting.set(id, { resolve, reject });
    w.postMessage({ type: "detect", id, pixels: pixels.data, width: pixels.width, height: pixels.height, scale }, [
      pixels.data.buffer,
    ]);
  });
}

/**
 * A copy of the image no longer than `long` on its long side. Each step halves it at most, so a 12-megapixel photo
 * is averaged down properly and thin cracks are not skipped over.
 */
export function shrink(source: CanvasImageSource, width: number, height: number, long: number): HTMLCanvasElement {
  const scale = Math.min(1, long / Math.max(width, height));
  const tw = Math.max(1, Math.round(width * scale));
  const th = Math.max(1, Math.round(height * scale));
  let from = source;
  let w = width;
  let h = height;
  for (;;) {
    const last = w <= tw * 2 && h <= th * 2;
    const nw = last ? tw : Math.ceil(w / 2);
    const nh = last ? th : Math.ceil(h / 2);
    const canvas = document.createElement("canvas");
    canvas.width = nw;
    canvas.height = nh;
    const context = canvas.getContext("2d", { willReadFrequently: last });
    if (!context) throw new Error("no 2D canvas");
    context.imageSmoothingQuality = "high";
    context.drawImage(from, 0, 0, w, h, 0, 0, nw, nh);
    if (last) return canvas;
    from = canvas;
    w = nw;
    h = nh;
  }
}
