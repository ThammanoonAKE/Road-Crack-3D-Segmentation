// Reduce a road section to the point cloud the home page animates (components/HeroCloud.tsx).
//
//   node --max-old-space-size=8000 tools/hero_points.mjs [--section 2] [--cell 0.007] [--fine 0.0035] [--out FILE]
//
// Run it from web/. It reads public/splats/hole_<n>.ksplat and hole_<n>.cracks.bin and writes
// public/media/hero_points.bin.
//
// The Gaussians are binned by their centres on the road plane (y = 0 in the web frame):
//   - one point per --cell square, at the opacity-weighted mean position and colour of the Gaussians in it;
//   - one more per --fine square wherever the Gaussians labelled as crack hold more than half of the opacity,
//     so the cracks keep their shape when they are drawn in red.
// Squares holding less than SOLID of summed opacity are left out: they are the frayed edge of the reconstruction.
//
// File layout (little-endian):
//   "RCP1"  uint32 count  float32 min[3]  float32 size[3]          32 bytes
//   uint16 x y z per point, as fractions of `size` from `min`       6 bytes each
//   uint8 r g b crack per point; crack is the crack share, 0..255   4 bytes each
import * as fs from 'fs';
import * as path from 'path';
import * as THREE from 'three';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';

const SOLID = 1.0; // summed opacity a coarse square needs to become a point
const SOLID_FINE = 0.3; // the same for a fine crack square, which holds a quarter as many Gaussians

const args = process.argv.slice(2);
const option = (name, fallback) => {
  const i = args.indexOf(name);
  return i < 0 ? fallback : args.splice(i, 2)[1];
};
const n = Number(option('--section', '2'));
const cell = Number(option('--cell', '0.007'));
const fine = Number(option('--fine', '0.0035'));
const out = option('--out', path.join('public', 'media', 'hero_points.bin'));

const file = fs.readFileSync(path.join('public', 'splats', `hole_${n}.ksplat`));
const buffer = new GaussianSplats3D.SplatBuffer(file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength));
const count = buffer.getSplatCount();
const bin = fs.readFileSync(path.join('public', 'splats', `hole_${n}.cracks.bin`));
const indices = new Uint32Array(bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
if (!indices.length || indices[indices.length - 1] >= count) throw new Error(`hole_${n}.cracks.bin does not fit hole_${n}.ksplat`);
const isCrack = new Uint8Array(count);
for (const i of indices) isCrack[i] = 1;

// One pass per grid: sum opacity-weighted position, colour and crack share into the square under each centre.
function bin2d(step) {
  const squares = new Map(); // "j,k" packed into a number -> [w, x, y, z, r, g, b, crack]
  const centre = new THREE.Vector3(), colour = new THREE.Vector4();
  for (let i = 0; i < count; i++) {
    buffer.getSplatCenter(i, centre);
    buffer.getSplatColor(i, colour);
    const w = colour.w / 255;
    if (w < 0.05) continue;
    const key = (Math.floor(centre.x / step) + 32768) * 65536 + (Math.floor(centre.z / step) + 32768);
    let s = squares.get(key);
    if (!s) squares.set(key, (s = new Float64Array(8)));
    s[0] += w;
    s[1] += w * centre.x; s[2] += w * centre.y; s[3] += w * centre.z;
    s[4] += w * colour.x; s[5] += w * colour.y; s[6] += w * colour.z;
    s[7] += w * isCrack[i];
  }
  return squares;
}

const points = []; // [x, y, z, r, g, b, crack 0..1]
for (const s of bin2d(cell).values()) {
  if (s[0] < SOLID) continue;
  points.push([s[1] / s[0], s[2] / s[0], s[3] / s[0], s[4] / s[0], s[5] / s[0], s[6] / s[0], s[7] / s[0]]);
}
const coarse = points.length;
for (const s of bin2d(fine).values()) {
  if (s[0] < SOLID_FINE || s[7] / s[0] <= 0.5) continue;
  points.push([s[1] / s[0], s[2] / s[0], s[3] / s[0], s[4] / s[0], s[5] / s[0], s[6] / s[0], 1]);
}

const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
for (const p of points) for (let a = 0; a < 3; a++) { min[a] = Math.min(min[a], p[a]); max[a] = Math.max(max[a], p[a]); }
const size = max.map((v, a) => v - min[a] || 1);

const bytes = Buffer.alloc(32 + points.length * 10);
bytes.write('RCP1', 0, 'latin1');
bytes.writeUInt32LE(points.length, 4);
for (let a = 0; a < 3; a++) { bytes.writeFloatLE(min[a], 8 + a * 4); bytes.writeFloatLE(size[a], 20 + a * 4); }
let o = 32;
for (const p of points) for (let a = 0; a < 3; a++, o += 2) bytes.writeUInt16LE(Math.round(((p[a] - min[a]) / size[a]) * 65535), o);
for (const p of points) {
  bytes[o++] = Math.round(p[3]); bytes[o++] = Math.round(p[4]); bytes[o++] = Math.round(p[5]);
  bytes[o++] = Math.round(p[6] * 255);
}
fs.writeFileSync(out, bytes);
console.log(`section ${n}: ${coarse} surface points + ${points.length - coarse} crack points -> ${out} (${(bytes.length / 1e6).toFixed(2)} MB)`);
console.log(`  x ${min[0].toFixed(2)}..${max[0].toFixed(2)}  y ${min[1].toFixed(2)}..${max[1].toFixed(2)}  z ${min[2].toFixed(2)}..${max[2].toFixed(2)}`);
