// Convert a 3DGS .ply to GaussianSplats3D's .ksplat.
//
//   node --max-old-space-size=24000 make_ksplat.mjs IN.ply OUT.ksplat [compressionLevel=1] [shDegree=0]
//
// Level 1 stores centres, scales and rotations as 16-bit values within 5-unit blocks: centres to
// about 0.08 mm at this export's scale. Colours are 8-bit, as on screen anyway. Splats below 1/255
// opacity are dropped; they draw nothing. shDegree keeps view-dependent colour up to degree 2 (the
// format's limit) as 16-bit values at level 1; the PLY must carry it.
import * as fs from 'fs';
import * as THREE from 'three';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';

const [input, output, level = '1', shDegree = '0'] = process.argv.slice(2);
if (!input || !output) {
  console.error('usage: node make_ksplat.mjs IN.ply OUT.ksplat [compressionLevel] [shDegree]');
  process.exit(1);
}
const file = fs.readFileSync(input);
const plyBuffer = file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength);
const splats = GaussianSplats3D.PlyParser.parseToUncompressedSplatArray(plyBuffer, Number(shDegree));
const generator = GaussianSplats3D.SplatBufferGenerator.getStandardGenerator(
    1, Number(level), 0, new THREE.Vector3(), 5.0, 256);
const splatBuffer = generator.generateFromUncompressedSplatArray(splats);
fs.writeFileSync(output, Buffer.from(splatBuffer.bufferData));
console.log(`${splats.splatCount} splats, SH degree ${splats.sphericalHarmonicsDegree} -> ${output} ` +
            `(${(fs.statSync(output).size / 1e6).toFixed(0)} MB)`);
