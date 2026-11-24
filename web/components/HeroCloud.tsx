"use client";

import { PerformanceMonitor, Sparkles } from "@react-three/drei";
import { Canvas, useFrame } from "@react-three/fiber";
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import styles from "./HeroCloud.module.css";

export type CloudMode = "model" | "trace" | "highlight";

/**
 * Section 2 as a live point cloud: about 180 000 points reduced from the reconstruction's own Gaussians
 * (tools/hero_points.mjs), drawn with react-three-fiber.
 *   - The points condense out of dust into the road, from the middle outwards.
 *   - A scan line sweeps the surface, the camera drifts and leans towards the pointer.
 *   - "trace" lets the red spread along the labelled cracks from the middle; "highlight" shows them all and
 *     dims the rest of the road.
 * Loaded only on the client, and not at all under reduced motion: Hero shows the photograph then.
 */

interface Cloud {
  geometry: THREE.BufferGeometry;
  reach: number; // distance from the origin to the farthest point, on the road plane
}

const VERTEX = /* glsl */ `
uniform float uTime;
uniform float uReveal; // 0..1: the road condensing out of dust
uniform float uFront;  // how far the red has spread along the cracks, in units of uReach
uniform float uCrack;  // 0..1: how red the labelled cracks are
uniform float uDim;    // 0..1: how far the rest of the road is dimmed
uniform float uSize;   // point diameter on the road, in scene units
uniform float uScale;  // pixels per scene unit at distance 1
uniform float uReach;
attribute vec4 aColour; // rgb, and the crack share in alpha
varying vec3 vColour;
varying float vAlpha;

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
  float r = length(position.xz) / uReach;
  float h1 = hash(position.xz * 91.7);
  float h2 = hash(position.zx * 57.3 + 3.1);
  float h3 = hash(position.xz * 23.9 + 7.7);

  // each point leaves its place in the dust a little later the farther it is from the middle
  float e = clamp(uReveal * 1.9 - r * 0.6 - h1 * 0.3, 0.0, 1.0);
  e = 1.0 - pow(1.0 - e, 3.0);
  vec3 dust = position + vec3((h2 - 0.5) * 0.9, 0.2 + h3 * 0.9, (h1 - 0.5) * 0.9);
  vec3 p = mix(dust, position, e);

  // the scan line: a bright band that crosses the section every nine seconds and lifts the points it passes
  float band = exp(-pow((position.z - mix(2.1, -2.3, fract(uTime / 9.0))) / 0.05, 2.0)) * e;
  p.y += band * 0.012;

  float crack = smoothstep(0.35, 0.65, aColour.a);
  float lit = crack * uCrack * smoothstep(uFront, uFront - 0.06, r);
  float front = crack * uCrack * exp(-pow((r - uFront) / 0.045, 2.0));
  float pulse = 0.5 + 0.5 * sin(r * 26.0 - uTime * 2.2);

  vec3 base = aColour.rgb * (1.0 - uDim * 0.55 * (1.0 - crack)) + band * 0.3;
  vec3 red = vec3(1.0, 0.271, 0.227) * (0.82 + 0.3 * pulse);
  vColour = mix(base, red, lit) + front * vec3(1.0, 0.62, 0.5);
  vAlpha = e;

  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  gl_Position = projectionMatrix * mv;
  gl_PointSize = uSize * uScale / -mv.z * (0.5 + 0.5 * e) * (1.0 + 0.3 * lit + 0.6 * front + 0.5 * band);
}
`;

const FRAGMENT = /* glsl */ `
varying vec3 vColour;
varying float vAlpha;

void main() {
  float a = smoothstep(0.5, 0.36, length(gl_PointCoord - 0.5)) * vAlpha;
  if (a < 0.01) discard;
  gl_FragColor = vec4(vColour, a);
}
`;

async function loadCloud(signal: AbortSignal): Promise<Cloud> {
  const response = await fetch("/media/hero_points.bin", { signal });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  const data = await response.arrayBuffer();
  const view = new DataView(data);
  if (view.getUint32(0, true) !== 0x31504352) throw new Error("hero_points.bin: not an RCP1 file"); // "RCP1"
  const count = view.getUint32(4, true);
  const min = [0, 1, 2].map((a) => view.getFloat32(8 + a * 4, true));
  const size = [0, 1, 2].map((a) => view.getFloat32(20 + a * 4, true));
  const packed = new Uint16Array(data, 32, count * 3);
  const positions = new Float32Array(count * 3);
  let reach = 0;
  for (let i = 0; i < count; i++) {
    for (let a = 0; a < 3; a++) positions[i * 3 + a] = min[a] + (packed[i * 3 + a] / 65535) * size[a];
    reach = Math.max(reach, Math.hypot(positions[i * 3], positions[i * 3 + 2]));
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("aColour", new THREE.BufferAttribute(new Uint8Array(data, 32 + count * 6, count * 4), 4, true));
  return { geometry, reach };
}

// Where the camera looks from: a slow orbit round a point on the road, leaning towards the pointer.
const TARGET = new THREE.Vector3(0.27, 0, 0.1);
const DISTANCE = 3.05;
const POLAR = 0.86; // radians from straight above
const FOV = 36;

function Road({ cloud, mode, run, onDrawn }: { cloud: Cloud; mode: CloudMode; run: number; onDrawn: () => void }) {
  const points = useRef<THREE.Points<THREE.BufferGeometry, THREE.ShaderMaterial>>(null);
  const [shader] = useState(() => ({
    vertexShader: VERTEX,
    fragmentShader: FRAGMENT,
    transparent: true,
    // the road is one layer of points seen from above: drawn in file order, the crack points land on top
    depthTest: false,
    depthWrite: false,
    uniforms: {
      uTime: { value: 0 },
      uReveal: { value: 0 },
      uFront: { value: 0 },
      uCrack: { value: 0 },
      uDim: { value: 0 },
      uSize: { value: 0.0092 },
      uScale: { value: 1 },
      uReach: { value: cloud.reach },
    },
  }));
  const state = useRef({ mode, front: 0, drawn: false, pointer: new THREE.Vector2(), lean: new THREE.Vector2(), aim: new THREE.Vector3() });

  // "trace" replays the spread each time it is chosen; "highlight" shows every crack at once
  useEffect(() => {
    state.current.mode = mode;
    state.current.front = mode === "trace" ? 0 : 2;
  }, [mode, run]);

  useEffect(() => {
    const move = (e: PointerEvent) => {
      if (e.pointerType === "touch") return;
      state.current.pointer.set((e.clientX / innerWidth) * 2 - 1, (e.clientY / innerHeight) * 2 - 1);
    };
    window.addEventListener("pointermove", move, { passive: true });
    return () => window.removeEventListener("pointermove", move);
  }, []);

  useFrame(({ camera, size, viewport, clock }, delta) => {
    if (!points.current) return;
    const s = state.current;
    const u = points.current.material.uniforms;
    const dt = Math.min(delta, 0.1);
    u.uTime.value = clock.elapsedTime;
    u.uReveal.value = Math.min(1, u.uReveal.value + dt / 2.6);
    // the red starts to spread once the road has mostly formed
    if (u.uReveal.value > 0.7) s.front = Math.min(2, s.front + dt * 0.5);
    u.uFront.value = s.front;
    u.uCrack.value = THREE.MathUtils.damp(u.uCrack.value, s.mode === "model" ? 0 : 1, 5, dt);
    u.uDim.value = THREE.MathUtils.damp(u.uDim.value, s.mode === "highlight" ? 1 : 0, 5, dt);
    u.uScale.value = (size.height * viewport.dpr) / (2 * Math.tan((FOV * Math.PI) / 360));

    // On a tall screen the card covers the lower half: step back a little and look at a point nearer the camera,
    // so the middle of the section sits in the upper part of the frame.
    const tall = THREE.MathUtils.clamp((1.25 - size.width / size.height) / 0.75, 0, 1);
    const distance = DISTANCE * (1 + 0.85 * tall);
    s.lean.x = THREE.MathUtils.damp(s.lean.x, s.pointer.x, 2.5, dt);
    s.lean.y = THREE.MathUtils.damp(s.lean.y, s.pointer.y, 2.5, dt);
    const theta = Math.sin(clock.elapsedTime * 0.11) * 0.2 + s.lean.x * 0.14;
    const phi = POLAR + s.lean.y * 0.05;
    camera.position.set(
      TARGET.x + distance * Math.sin(phi) * Math.sin(theta),
      TARGET.y + distance * Math.cos(phi),
      TARGET.z + distance * Math.sin(phi) * Math.cos(theta),
    );
    camera.lookAt(s.aim.set(TARGET.x, 0, TARGET.z + 0.85 * tall));

    if (!s.drawn) {
      s.drawn = true;
      onDrawn();
    }
  });

  return (
    <>
      <points ref={points} geometry={cloud.geometry} frustumCulled={false}>
        <shaderMaterial args={[shader]} />
      </points>
      {/* dust that keeps drifting over the road after the points have settled */}
      <Sparkles count={70} scale={[3.4, 0.7, 4]} position={[0.27, 0.4, -0.1]} size={1.6} speed={0.22} opacity={0.42} color="#ffffff" />
    </>
  );
}

export default function HeroCloud({
  mode,
  run,
  onError,
}: {
  mode: CloudMode;
  run: number;
  onError: () => void;
}) {
  const root = useRef<HTMLDivElement>(null);
  const [cloud, setCloud] = useState<Cloud | null>(null);
  const [drawn, setDrawn] = useState(false);
  const [visible, setVisible] = useState(true);
  const [dpr, setDpr] = useState(() => Math.min(window.devicePixelRatio || 1, 1.75));

  // the points are fetched once; the latest onError is reached through a ref, so a new callback does not refetch
  const fail = useRef(onError);
  useEffect(() => {
    fail.current = onError;
  }, [onError]);

  useEffect(() => {
    const abort = new AbortController();
    let loaded: Cloud | null = null;
    loadCloud(abort.signal)
      .then((c) => setCloud((loaded = c)))
      .catch(() => {
        if (!abort.signal.aborted) fail.current(); // without the points, Hero shows its photograph
      });
    return () => {
      abort.abort();
      loaded?.geometry.dispose();
    };
  }, []);

  // nothing is drawn while the hero is scrolled out of view
  useEffect(() => {
    if (!root.current) return;
    const seen = new IntersectionObserver(([entry]) => setVisible(entry.isIntersecting));
    seen.observe(root.current);
    return () => seen.disconnect();
  }, []);

  return (
    <div ref={root} className={styles.cloud} data-drawn={drawn} aria-hidden>
      {cloud && (
        <Canvas
          dpr={dpr}
          frameloop={visible ? "always" : "never"}
          camera={{ fov: FOV, near: 0.1, far: 40, position: [0, 2, 2] }}
          gl={{ antialias: false, alpha: true, powerPreference: "high-performance" }}
          style={{ pointerEvents: "none" }}
        >
          <PerformanceMonitor onDecline={() => setDpr(1)} />
          <Road
            cloud={cloud}
            mode={mode}
            run={run}
            onDrawn={() => setDrawn(true)}
          />
        </Canvas>
      )}
    </div>
  );
}
