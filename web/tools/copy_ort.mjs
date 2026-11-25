// Puts the onnxruntime-web files the /detect page loads into public/ort (not committed: they come from
// node_modules). Runs before `next dev` and `next build`.
//   node tools/copy_ort.mjs
import { copyFileSync, mkdirSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const web = dirname(dirname(fileURLToPath(import.meta.url)));
const from = join(web, "node_modules", "onnxruntime-web", "dist");
const to = join(web, "public", "ort");

// The WebGPU build (it also holds the WebAssembly fallback), the script that drives the binary, and the binary.
// The driver is a file of its own, not the copy inside the "bundle" build, because its threads start from its
// address. The modules are stored as .js: every host serves that as JavaScript, which a module worker insists on.
const FILES = [
  ["ort.webgpu.min.mjs", "ort.webgpu.min.js"],
  ["ort-wasm-simd-threaded.asyncify.mjs", "ort-wasm-simd-threaded.asyncify.js"],
  ["ort-wasm-simd-threaded.asyncify.wasm", "ort-wasm-simd-threaded.asyncify.wasm"],
];

mkdirSync(to, { recursive: true });
for (const [source, name] of FILES) {
  copyFileSync(join(from, source), join(to, name));
  console.log(`public/ort/${name}  ${(statSync(join(to, name)).size / 1e6).toFixed(1)} MB`);
}
