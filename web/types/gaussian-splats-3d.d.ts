// Minimal typing for the parts of GaussianSplats3D 0.4.7 this app uses.
declare module "@mkkellogg/gaussian-splats-3d" {
  export const SceneRevealMode: { Default: 0; Gradual: 1; Instant: 2 };

  interface Vec3 {
    x: number;
    y: number;
    z: number;
    set(x: number, y: number, z: number): Vec3;
    fromArray(a: ArrayLike<number>): Vec3;
  }

  interface OrbitControls {
    target: Vec3;
    autoRotate: boolean;
    autoRotateSpeed: number;
    enableDamping: boolean;
    update(): void;
  }

  interface DataTexture {
    data: Uint32Array;
    texture: { needsUpdate: boolean };
  }

  export class Viewer {
    constructor(options: Record<string, unknown>);
    camera: { position: Vec3; up: Vec3 };
    controls: OrbitControls | null;
    splatMesh: {
      getSplatCount(): number;
      getSplatDataTextures(): { baseData: { colors: Uint8Array }; centerColors: DataTexture };
    };
    keyDownListener: ((e: KeyboardEvent) => void) | null;
    addSplatScene(
      path: string,
      options: {
        showLoadingUI?: boolean;
        progressiveLoad?: boolean;
        onProgress?: (percent: number, label: string, status: number) => void;
      },
    ): Promise<void>;
    start(): void;
    dispose(): Promise<void>;
  }
}
