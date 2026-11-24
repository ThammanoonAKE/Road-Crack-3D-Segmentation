// Measure each road section's cracks on the road plane: surveyed area, crack area, crack length, mean width.
//
//   node --max-old-space-size=8000 tools/crack_metrics.mjs [--cell 0.002] [--maps DIR] [--no-write] [SECTION ...]
//
// Run it from web/. It reads public/splats/hole_<n>.ksplat and hole_<n>.cracks.bin (every section by default) and
// writes lib/crack-metrics.json, which the site shows. With all eight sections it replaces the file; with fewer it
// updates those entries. --maps writes a picture of each section seen from straight above, with the cracks and
// their centre lines drawn in; --no-write leaves the JSON alone (for trying another --cell).
//
// The web frame (gaussian_splatting/export_web_splat.py) has the road as the plane y = 0, so the splat is
// measured looking straight down, on a grid of --cell metres:
//   1. Every Gaussian adds opacity x its footprint (its covariance projected onto the road plane, out to 3 sigma)
//      to the cells it covers. A cell is road surface where that sum reaches COVERED, and crack where the
//      Gaussians labelled as crack hold more than half of it.
//   2. Crack regions smaller than MIN_REGION are dropped as speckle, and holes that small are filled. What is
//      left is the crack area: the area of the labelled band, which is wider than the crack's opening.
//   3. For the centre lines, gaps up to CLOSE wide and unlabelled islands under ISLAND are closed, the band is
//      thinned to one-cell lines (Zhang-Suen), and branches shorter than SPUR that end freely are pruned.
//   4. Length: each branch is stepped along in strides of STEP and the strides' straight lengths are summed, so
//      the result is the length at that scale, whatever the grid's cell.
//   5. Mean width = crack area / crack length.
//
// Units: export_web_splat.py scales each section so that the phone was 1.5 units above the road, so a unit is a
// metre only as far as the phone was really held 1.5 m high. Lengths and widths scale with that height, areas with
// its square; the crack share of the surface does not depend on it. METRES_PER_UNIT corrects a section whose true
// scale is known from a measurement on site.
import * as fs from 'fs';
import * as path from 'path';
import * as zlib from 'zlib';
import * as THREE from 'three';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';

const METRES_PER_UNIT = {}; // e.g. { 3: 1.08 } when a known length in section 3 measures 8% short in the splat
const COVERED = 0.5; // summed opacity at which a cell counts as drawn road surface
const MIN_REGION = 1e-4; // m2: crack regions (and holes in them) below 1 cm2 are speckle
const CLOSE = 0.003; // m: gaps this wide in a crack band are bridged before thinning
const ISLAND = 1e-3; // m2: unlabelled islands below 10 cm2 inside a crack do not split its centre line
const SPUR = 0.025; // m: free-ended centre-line branches shorter than this are thinning artefacts
const STEP = 0.01; // m: the stride the centre lines are measured with
const MAX_REACH = 0.25; // m: a Gaussian's footprint is cut off this far from its centre

const args = process.argv.slice(2);
const option = (name, fallback) => {
  const i = args.indexOf(name);
  return i < 0 ? fallback : args.splice(i, 2)[1];
};
const flag = (name) => {
  const i = args.indexOf(name);
  return i >= 0 && args.splice(i, 1).length > 0;
};
const cellM = Number(option('--cell', '0.002'));
const mapsDir = option('--maps', '');
const noWrite = flag('--no-write');
const sections = args.length ? args.map(Number) : [1, 2, 3, 4, 5, 6, 7, 8];
const out = path.join('lib', 'crack-metrics.json');

function measure(n) {
  const metres = METRES_PER_UNIT[n] || 1;
  const cell = cellM / metres; // the grid step in web units
  const file = fs.readFileSync(path.join('public', 'splats', `hole_${n}.ksplat`));
  const buffer = new GaussianSplats3D.SplatBuffer(file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength));
  const count = buffer.getSplatCount();
  const bin = fs.readFileSync(path.join('public', 'splats', `hole_${n}.cracks.bin`));
  const indices = new Uint32Array(bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
  if (!indices.length || indices[indices.length - 1] >= count) throw new Error(`hole_${n}.cracks.bin does not fit hole_${n}.ksplat`);
  const isCrack = new Uint8Array(count);
  for (const i of indices) isCrack[i] = 1;

  // Pass 1: centres, footprints (inverse 2x2 covariance on the road plane) and the grid's extent.
  const cx = new Float32Array(count), cz = new Float32Array(count), alpha = new Float32Array(count);
  const ia = new Float32Array(count), ib = new Float32Array(count), ic = new Float32Array(count);
  const rx = new Float32Array(count), rz = new Float32Array(count); // 3-sigma reach along x and z
  const rgb = mapsDir ? new Uint8Array(count * 3) : null;
  const centre = new THREE.Vector3(), scale = new THREE.Vector3(), quat = new THREE.Quaternion();
  const colour = new THREE.Vector4(), m = new THREE.Matrix4();
  const blur = (0.5 * cell) ** 2; // a Gaussian smaller than a cell still covers about one
  const reach = MAX_REACH / metres;
  let x0 = Infinity, x1 = -Infinity, z0 = Infinity, z1 = -Infinity;
  for (let i = 0; i < count; i++) {
    buffer.getSplatCenter(i, centre);
    buffer.getSplatScaleAndRotation(i, scale, quat);
    buffer.getSplatColor(i, colour);
    m.makeRotationFromQuaternion(quat.normalize());
    const e = m.elements; // column-major: e[0], e[1], e[2] is the first axis
    const s0 = scale.x * scale.x, s1 = scale.y * scale.y, s2 = scale.z * scale.z;
    const xx = e[0] * e[0] * s0 + e[4] * e[4] * s1 + e[8] * e[8] * s2 + blur;
    const zz = e[2] * e[2] * s0 + e[6] * e[6] * s1 + e[10] * e[10] * s2 + blur;
    const xz = e[0] * e[2] * s0 + e[4] * e[6] * s1 + e[8] * e[10] * s2;
    const det = xx * zz - xz * xz;
    cx[i] = centre.x; cz[i] = centre.z; alpha[i] = colour.w / 255;
    ia[i] = zz / det; ib[i] = -xz / det; ic[i] = xx / det;
    rx[i] = Math.min(3 * Math.sqrt(xx), reach); rz[i] = Math.min(3 * Math.sqrt(zz), reach);
    if (rgb) rgb.set([colour.x, colour.y, colour.z], i * 3);
    x0 = Math.min(x0, centre.x - rx[i]); x1 = Math.max(x1, centre.x + rx[i]);
    z0 = Math.min(z0, centre.z - rz[i]); z1 = Math.max(z1, centre.z + rz[i]);
  }
  const BORDER = 4; // empty cells round the grid, so neighbourhoods never leave it
  x0 -= BORDER * cell; z0 -= BORDER * cell;
  const W = Math.ceil((x1 - x0) / cell) + BORDER, H = Math.ceil((z1 - z0) / cell) + BORDER, N = W * H;

  // Pass 2: sum the footprints.
  const total = new Float32Array(N), crackSum = new Float32Array(N);
  const colourSum = rgb ? [new Float32Array(N), new Float32Array(N), new Float32Array(N)] : null;
  for (let i = 0; i < count; i++) {
    const u = (cx[i] - x0) / cell, v = (cz[i] - z0) / cell; // cell (j, k) has its middle at (j + 0.5, k + 0.5)
    const j0 = Math.floor(u - rx[i] / cell), j1 = Math.floor(u + rx[i] / cell);
    const k0 = Math.floor(v - rz[i] / cell), k1 = Math.floor(v + rz[i] / cell);
    const a = ia[i] * cell * cell, b = ib[i] * cell * cell, c = ic[i] * cell * cell, crack = isCrack[i];
    for (let k = k0; k <= k1; k++) {
      const dz = k + 0.5 - v;
      for (let j = j0; j <= j1; j++) {
        const dx = j + 0.5 - u;
        const q = a * dx * dx + 2 * b * dx * dz + c * dz * dz;
        if (q > 9) continue;
        const w = alpha[i] * Math.exp(-0.5 * q), at = k * W + j;
        total[at] += w;
        if (crack) crackSum[at] += w;
        if (colourSum) for (let ch = 0; ch < 3; ch++) colourSum[ch][at] += w * rgb[i * 3 + ch];
      }
    }
  }
  let surface = 0;
  const mask = new Uint8Array(N);
  for (let k = BORDER; k < H - BORDER; k++) for (let j = BORDER; j < W - BORDER; j++) {
    const at = k * W + j;
    if (total[at] < COVERED) continue;
    surface++;
    if (crackSum[at] > 0.5 * total[at]) mask[at] = 1;
  }

  // Flip every connected region of `value` cells smaller than `area` m2 (crack regions join across corners,
  // the gaps between them do not).
  const stack = new Int32Array(N);
  const flipSmall = (grid, value, area) => {
    const minCells = area / (cellM * cellM), seen = new Uint8Array(N);
    for (let start = 0; start < N; start++) {
      if (grid[start] !== value || seen[start]) continue;
      let top = 0, size = 0;
      stack[top++] = start; seen[start] = 1;
      while (top > size) {
        const at = stack[size++], j = at % W, k = (at - j) / W;
        for (let dk = -1; dk <= 1; dk++) for (let dj = -1; dj <= 1; dj++) {
          const jj = j + dj, kk = k + dk;
          if (jj < 0 || kk < 0 || jj >= W || kk >= H || (!value && dj && dk)) continue;
          const next = kk * W + jj;
          if (grid[next] === value && !seen[next]) { seen[next] = 1; stack[top++] = next; }
        }
      }
      if (size < minCells) for (let s = 0; s < size; s++) grid[stack[s]] = 1 - value;
    }
  };
  flipSmall(mask, 1, MIN_REGION); // crack speckle
  flipSmall(mask, 0, MIN_REGION); // pinholes inside a crack
  let crackCells = 0;
  for (let at = 0; at < N; at++) crackCells += mask[at];

  // Centre lines. Close gaps (grow, then shrink, by a square of CLOSE), close small islands, thin.
  const grow = (grid, r, value) => { // every cell within r (along x, then z) of a `value` cell becomes `value`
    for (const [stride, limit, lines] of [[1, W, H], [W, H, W]]) {
      const src = grid.slice();
      for (let l = 0; l < lines; l++) for (let p = r; p < limit - r; p++) {
        const at = stride === 1 ? l * W + p : p * W + l;
        if (src[at] === value) continue;
        for (let d = 1; d <= r; d++) if (src[at - d * stride] === value || src[at + d * stride] === value) { grid[at] = value; break; }
      }
    }
  };
  const line = mask.slice();
  const closeCells = Math.min(BORDER - 1, Math.max(1, Math.round(CLOSE / cellM)));
  grow(line, closeCells, 1);
  grow(line, closeCells, 0);
  flipSmall(line, 0, ISLAND);
  const offsets = [-W, -W + 1, 1, W + 1, W, W - 1, -1, -W - 1]; // clockwise from north
  let active = [];
  for (let at = 0; at < N; at++) if (line[at]) active.push(at);
  const p = new Uint8Array(8);
  for (let changed = true; changed;) {
    changed = false;
    for (const pass of [0, 1]) {
      const gone = [];
      for (const at of active) {
        let sum = 0, turns = 0;
        for (let t = 0; t < 8; t++) sum += p[t] = line[at + offsets[t]];
        if (sum < 2 || sum > 6) continue;
        for (let t = 0; t < 8; t++) if (!p[t] && p[(t + 1) % 8]) turns++;
        if (turns !== 1) continue;
        if (pass === 0 ? (p[0] * p[2] * p[4] || p[2] * p[4] * p[6]) : (p[0] * p[2] * p[6] || p[0] * p[4] * p[6])) continue;
        gone.push(at);
      }
      for (const at of gone) line[at] = 0;
      if (gone.length) changed = true;
      active = active.filter((at) => line[at]);
    }
  }

  // The lines as a graph. Two line cells are linked when they share a side, or a corner that no line cell
  // fills in; a cell with other than two links is a node, and a branch runs from node to node.
  const links = (at) => {
    const found = [];
    for (let t = 0; t < 8; t++) {
      const next = at + offsets[t];
      if (!line[next]) continue;
      if (t % 2 && (line[at + offsets[t - 1]] || line[at + offsets[(t + 1) % 8]])) continue;
      found.push(next);
    }
    return found;
  };
  const degree = new Uint8Array(N);
  for (const at of active) degree[at] = links(at).length;
  const walked = new Set(); // node * 8 + direction, for branch ends already walked
  const direction = (from, to) => offsets.indexOf(to - from);
  const branches = [];
  const walk = (from, first) => {
    const cells = [from, first];
    let previous = from, at = first;
    while (degree[at] === 2 && at !== from) {
      const next = links(at).find((q) => q !== previous);
      previous = at; at = next;
      cells.push(at);
    }
    return cells;
  };
  for (const node of active) {
    if (degree[node] === 2) continue;
    for (const first of links(node)) {
      if (walked.has(node * 8 + direction(node, first))) continue;
      const cells = walk(node, first), end = cells[cells.length - 1];
      walked.add(node * 8 + direction(node, first));
      walked.add(end * 8 + direction(end, cells[cells.length - 2]));
      branches.push(cells);
    }
  }
  const onBranch = new Uint8Array(N);
  for (const cells of branches) for (const at of cells) onBranch[at] = 1;
  for (const at of active) { // rings with no node
    if (degree[at] !== 2 || onBranch[at]) continue;
    const cells = walk(at, links(at)[0]);
    for (const q of cells) onBranch[q] = 1;
    branches.push(cells);
  }
  const stride = Math.max(1, Math.round(STEP / cellM));
  let length = 0, kept = 0, pruned = 0;
  for (const cells of branches) {
    let metresLong = 0;
    for (let s = 0; s < cells.length - 1; s += stride) {
      const a = cells[s], b = cells[Math.min(s + stride, cells.length - 1)];
      metresLong += Math.hypot((a % W) - (b % W), Math.floor(a / W) - Math.floor(b / W)) * cellM;
    }
    const first = cells[0], last = cells[cells.length - 1];
    const spur = metresLong < SPUR && first !== last &&
      ((degree[first] === 1 && degree[last] > 2) || (degree[last] === 1 && degree[first] > 2));
    if (spur) {
      pruned++;
      for (const at of cells) if (degree[at] <= 2) line[at] = 0;
    } else {
      kept++;
      length += metresLong;
    }
  }

  const area = cellM * cellM;
  const result = {
    surface_m2: +(surface * area).toFixed(2),
    crack_m2: +(crackCells * area).toFixed(3),
    crack_share: +(crackCells / surface).toFixed(4),
    length_m: +length.toFixed(1),
    width_mm: Math.round((crackCells * area) / length * 1000),
  };
  if (METRES_PER_UNIT[n]) result.metres_per_unit = metres;

  if (mapsDir) { // a look straight down: the road's colours, cracks in red, centre lines in black
    const png = Buffer.alloc(H * (1 + W * 3));
    for (let k = 0; k < H; k++) {
      const row = k * (1 + W * 3) + 1;
      for (let j = 0; j < W; j++) {
        const at = k * W + j, o = row + j * 3;
        if (total[at] < COVERED) { png[o] = png[o + 1] = png[o + 2] = 24; continue; }
        for (let ch = 0; ch < 3; ch++) png[o + ch] = colourSum[ch][at] / total[at];
        if (mask[at]) { png[o] = png[o] * 0.35 + 255 * 0.65; png[o + 1] *= 0.35; png[o + 2] *= 0.35; }
        if (line[at]) png[o] = png[o + 1] = png[o + 2] = 0;
      }
    }
    const bar = Math.round(0.5 / cellM); // a 0.5 m scale bar
    for (let k = H - 20; k < H - 14; k++) png.fill(255, k * (1 + W * 3) + 1 + 16 * 3, k * (1 + W * 3) + 1 + (16 + bar) * 3);
    const chunk = (type, data) => {
      const body = Buffer.concat([Buffer.from(type), data]), head = Buffer.alloc(4), tail = Buffer.alloc(4);
      head.writeUInt32BE(data.length); tail.writeUInt32BE(zlib.crc32(body));
      return Buffer.concat([head, body, tail]);
    };
    const ihdr = Buffer.alloc(13);
    ihdr.writeUInt32BE(W, 0); ihdr.writeUInt32BE(H, 4); ihdr.set([8, 2, 0, 0, 0], 8);
    fs.mkdirSync(mapsDir, { recursive: true });
    fs.writeFileSync(path.join(mapsDir, `hole_${n}_map.png`), Buffer.concat([
      Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', zlib.deflateSync(png)), chunk('IEND', Buffer.alloc(0))]));
  }
  console.log(`section ${n}: ${count.toLocaleString('en-US')} Gaussians on ${W} x ${H} cells; surface ${result.surface_m2} m2, ` +
              `crack ${result.crack_m2} m2 (${(100 * result.crack_share).toFixed(1)}%), length ${result.length_m} m ` +
              `(${kept} branches, ${pruned} spurs pruned), mean width ${result.width_mm} mm`);
  return result;
}

const previous = sections.length < 8 && fs.existsSync(out) ? JSON.parse(fs.readFileSync(out, 'utf8')).sections : {};
const measured = { ...previous };
for (const n of sections) measured[n] = measure(n);
if (!noWrite) {
  fs.writeFileSync(out, JSON.stringify({
    note: 'Made by tools/crack_metrics.mjs. Metres assume the phone was held 1.5 m above the road.',
    cell_m: cellM,
    sections: measured,
  }, null, 1) + '\n');
  console.log(`-> ${out}`);
}
