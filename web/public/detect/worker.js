// The crack detector of the /detect page: crack_detector_1class_best as ONNX (edge_ai/onnx_web.py), run by
// onnxruntime-web in this worker so the page keeps animating. Nothing here talks to a server except to download the
// model and the runtime: the photo arrives as pixels from the page and only a mask goes back.
//
//   page -> { type: "load", model, backend? }            download the model (kept in Cache Storage) and open it;
//                                                        backend "wasm" skips WebGPU
//   page -> { type: "detect", id, pixels, width, height } RGBA pixels, already shrunk by the page
//   worker -> { type: "progress", loaded, total } | { type: "ready", backend }
//           | { type: "result", id, mask, width, height, share, ms, backend } | { type: "error", id?, stage, message }
//
// This file is served as it is (not bundled); tools/copy_ort.mjs puts the runtime next to it in /ort.
import * as ort from "/ort/ort.webgpu.min.js";

ort.env.wasm.wasmPaths = {
  mjs: "/ort/ort-wasm-simd-threaded.asyncify.js",
  wasm: "/ort/ort-wasm-simd-threaded.asyncify.wasm",
};

const CACHE = "crack-model-v1";
const THRESHOLD = Math.log(0.16 / 0.84); // the package's 0.16 probability, as a logit
const MIN_AREA = 150; // px at the photo's own size, as the package drops; scaled with the photo below

let bytes = null; // the model file, kept until a session exists on the backend that works
let session = null;
let backend = "";

async function download(url) {
  let cache = null;
  try {
    cache = await caches.open(CACHE);
    const hit = await cache.match(url, { ignoreVary: true }); // the home page may have stored it already
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch {
    cache = null; // private windows and old browsers: download every time
  }
  const response = await fetch(url);
  if (!response.ok || !response.body) throw new Error(`${response.status} ${response.statusText}`);
  const total = Number(response.headers.get("content-length")) || 0;
  const reader = response.body.getReader();
  const parts = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    loaded += value.length;
    postMessage({ type: "progress", loaded, total });
  }
  const data = new Uint8Array(loaded);
  let at = 0;
  for (const part of parts) {
    data.set(part, at);
    at += part.length;
  }
  if (cache) {
    // the stored copy is the whole point of the cache, but a full disk must not stop the detection
    cache.put(url, new Response(data, { headers: { "content-type": "application/octet-stream" } })).catch(() => {});
  }
  return data;
}

async function open(provider) {
  session = await ort.InferenceSession.create(bytes, { executionProviders: [provider], graphOptimizationLevel: "all" });
  backend = provider;
}

async function load(url, only) {
  bytes = await download(url);
  try {
    if (only === "wasm" || !navigator.gpu || !(await navigator.gpu.requestAdapter())) throw new Error("no WebGPU adapter");
    await open("webgpu");
  } catch {
    await open("wasm");
  }
  postMessage({ type: "ready", backend });
}

/** RGBA pixels -> 1x3xHxW floats in 0-1, mirrored out to multiples of 32 as the package pads. */
function tensor(pixels, width, height) {
  const W = Math.ceil(width / 32) * 32;
  const H = Math.ceil(height / 32) * 32;
  const data = new Float32Array(3 * H * W);
  const plane = H * W;
  for (let y = 0; y < H; y++) {
    const sy = y < height ? y : Math.max(0, 2 * height - 2 - y);
    for (let x = 0; x < W; x++) {
      const sx = x < width ? x : Math.max(0, 2 * width - 2 - x);
      const from = (sy * width + sx) * 4;
      const to = y * W + x;
      data[to] = pixels[from] / 255;
      data[plane + to] = pixels[from + 1] / 255;
      data[2 * plane + to] = pixels[from + 2] / 255;
    }
  }
  return new ort.Tensor("float32", data, [1, 3, H, W]);
}

/** Logits -> mask (255 = crack) with the regions smaller than minArea removed, and the share of crack pixels. */
function mask(logits, W, width, height, minArea) {
  const out = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) if (logits[y * W + x] > THRESHOLD) out[y * width + x] = 1;
  }
  // 8-connected regions by flood fill: 1 = not visited yet, 2 = in the region being filled, then 255 or 0
  const stack = new Int32Array(out.length);
  const region = new Int32Array(out.length);
  let kept = 0;
  for (let start = 0; start < out.length; start++) {
    if (out[start] !== 1) continue;
    let top = 0;
    let size = 0;
    stack[top++] = start;
    out[start] = 2;
    while (top > 0) {
      const p = stack[--top];
      region[size++] = p;
      const px = p % width;
      const py = (p - px) / width;
      for (let dy = -1; dy <= 1; dy++) {
        const ny = py + dy;
        if (ny < 0 || ny >= height) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const nx = px + dx;
          if (nx < 0 || nx >= width) continue;
          const q = ny * width + nx;
          if (out[q] === 1) {
            out[q] = 2;
            stack[top++] = q;
          }
        }
      }
    }
    const value = size >= minArea ? 255 : 0;
    if (value) kept += size;
    for (let i = 0; i < size; i++) out[region[i]] = value;
  }
  return { out, share: kept / out.length };
}

async function detect({ id, pixels, width, height, scale }) {
  const started = performance.now();
  const input = tensor(pixels, width, height);
  let output;
  try {
    output = await session.run({ image: input });
  } catch (error) {
    if (backend !== "webgpu") throw error;
    // the graphics card ran out of room or lost the device: the same model on the processor
    await session.release().catch(() => {});
    await open("wasm");
    output = await session.run({ image: input });
  }
  const logits = output.logits;
  const values = logits.data ?? (await logits.getData());
  const minArea = Math.max(1, Math.round(MIN_AREA * scale * scale));
  const { out, share } = mask(values, input.dims[3], width, height, minArea);
  postMessage({ type: "result", id, mask: out, width, height, share, ms: performance.now() - started, backend }, [out.buffer]);
}

let busy = Promise.resolve();
onmessage = ({ data }) => {
  // one job at a time: a second photo waits for the first
  busy = busy.then(async () => {
    try {
      if (data.type === "load") await load(data.model, data.backend);
      else if (data.type === "detect") await detect(data);
    } catch (error) {
      postMessage({ type: "error", id: data.id, stage: data.type, message: String(error?.message ?? error) });
    }
  });
};
