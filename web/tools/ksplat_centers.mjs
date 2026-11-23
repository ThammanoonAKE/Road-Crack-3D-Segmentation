// Write a .ksplat's splat centres, in the order the viewer indexes them, as float32 x y z triples.
//
//   node --max-old-space-size=24000 ksplat_centers.mjs IN.ksplat OUT.f32
//
// make_ksplat.mjs regroups splats into spatial buckets, so this order is not the PLY's; crack_labels.py matches
// the two by position.
import * as fs from 'fs';
import * as THREE from 'three';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';

const [input, output] = process.argv.slice(2);
if (!input || !output) {
  console.error('usage: node ksplat_centers.mjs IN.ksplat OUT.f32');
  process.exit(1);
}
const file = fs.readFileSync(input);
const splatBuffer = new GaussianSplats3D.SplatBuffer(file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength));
const count = splatBuffer.getSplatCount();
const centers = new Float32Array(count * 3);
const center = new THREE.Vector3();
for (let i = 0; i < count; i++) {
  splatBuffer.getSplatCenter(i, center);
  centers[i * 3] = center.x;
  centers[i * 3 + 1] = center.y;
  centers[i * 3 + 2] = center.z;
}
fs.writeFileSync(output, Buffer.from(centers.buffer));
console.log(`${count} centres -> ${output}`);
