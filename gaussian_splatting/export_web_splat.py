"""Export a road splat for a web viewer: level, centred on the crack, colour without view dependence.

The trainer's frame has no meaningful up. This moves the splat so the road plane (RANSAC on the SfM
points) is y = 0 with +y up, the spot the phone kept aiming at is the origin, and the phone's
opening view looks along -z, as a three.js camera does. Units are scaled so the phone was held
1.5 high, roughly metres. Only the base colour (degree-0 SH) is kept: on this road, renders with
degree 3 and degree 0 matched, and it is 45 of the 59 values per Gaussian. Everything else is
written at full precision, so a later .ksplat conversion decides the only loss. A JSON beside the
PLY holds the opening camera.

    python -I gaussian_splatting/export_web_splat.py --scene C:/3D-segmentation-data/road_crack/151605/scene \
        --ply runs/road_crack_151605_v2/sharp.ply --output web/public/splats/hole_1.ply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_clip_tour import fit_plane  # noqa: E402

PHONE_HEIGHT = 1.5
KEEP = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3"]


def web_frame(scene):
    """The web frame of a scene (gsplat's colmap Parser, as main() builds it): web = (p - middle) @ rot.T * scale.
    Returns (middle, rot, scale, aims, opening eye), aims being where each photo's optical axis met the road."""
    c2ws = scene.camtoworlds[np.argsort(scene.image_names)]
    centre, up = fit_plane(scene.points, np.random.default_rng(0))
    if (c2ws[:, :3, 3].mean(0) - centre) @ up < 0:
        up = -up
    eyes, fwds = c2ws[:, :3, 3], c2ws[:, :3, 2]
    flat = lambda v: v - (v @ up)[..., None] * up  # noqa: E731
    dist = np.where(fwds @ up < -0.05, ((centre - eyes) @ up) / np.minimum(fwds @ up, -0.05), 3.0)
    aims = eyes + fwds * np.clip(dist, 0.3, 6.0)[:, None]
    phone_height = float(np.median((eyes - centre) @ up))
    # The crack: the densest spot of the aim points (as render_clip_tour.py), on the road plane.
    gap = np.linalg.norm(flat(aims[:, None] - aims[None]), axis=2)
    near = gap[np.argmax((gap < 0.5 * phone_height).sum(1))] < 0.5 * phone_height
    middle = aims[near].mean(0)
    middle -= ((middle - centre) @ up) * up
    eyes_s = gaussian_filter1d(eyes, 3.6, axis=0, mode="nearest")
    aims_s = gaussian_filter1d(aims, 3.6, axis=0, mode="nearest")
    start = int(np.argmin(np.linalg.norm(flat(aims_s - middle), axis=1)))
    heading = flat((aims_s[start] - eyes_s[start])[None])[0]
    heading /= np.linalg.norm(heading)

    # New axes as rows: x right of the heading, y up, z back towards the opening camera.
    rot = np.stack([np.cross(heading, up), up, -heading])
    return middle, rot, PHONE_HEIGHT / phone_height, aims, eyes_s[start]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, required=True, help="The scene the splat trained on.")
    parser.add_argument("--ply", type=Path, required=True, help="The splat, in that scene's training frame.")
    parser.add_argument("--output", type=Path, required=True, help="Output .ply; the camera goes beside it as .json.")
    parser.add_argument("--crop", type=float, default=0.0,
                        help="Keep only road within this many phone heights (across the road plane) of where some "
                             "photo's optical axis met it, and within --max-height of the road (0: no crop). The "
                             "edge of what the phone aimed at is where the splat blurs.")
    parser.add_argument("--max-height", type=float, default=0.5,
                        help="With --crop: how far above or below the road to keep, in phone heights.")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "third_party" / "gsplat" / "examples"))
    from datasets.colmap import Parser  # gsplat example code, not a package

    scene = Parser(data_dir=str(args.scene), factor=1, normalize=True, test_every=8)
    middle, rot, scale, aims, opening = web_frame(scene)
    to_web = lambda p: (np.asarray(p) - middle) @ rot.T * scale  # noqa: E731

    with args.ply.open("rb") as handle:
        header = []
        while (line := handle.readline().decode("ascii").rstrip("\n")) != "end_header":
            header.append(line)
        names = [h.split()[2] for h in header if h.startswith("property")]
        data = np.fromfile(handle, dtype=np.dtype([(n, "<f4") for n in names]))
    xyz = to_web(np.stack([data[k] for k in "xyz"], 1).astype(np.float64))
    if args.crop:
        from scipy.spatial import cKDTree

        aim_web = to_web(aims)
        reach, _ = cKDTree(aim_web[:, [0, 2]]).query(xyz[:, [0, 2]], distance_upper_bound=args.crop * PHONE_HEIGHT)
        keep = np.isfinite(reach) & (np.abs(xyz[:, 1]) <= args.max_height * PHONE_HEIGHT)
        print(f"crop: kept {keep.sum():,} of {len(keep):,} within {args.crop} phone heights of the aim points")
        data, xyz = data[keep], xyz[keep]
    out = np.empty(len(data), dtype=np.dtype([(n, "<f4") for n in KEEP]))
    for i, k in enumerate("xyz"):
        out[k] = xyz[:, i]
    for k in ("f_dc_0", "f_dc_1", "f_dc_2", "opacity"):
        out[k] = data[k]
    for i in range(3):
        out[f"scale_{i}"] = data[f"scale_{i}"] + np.float32(np.log(scale))
    quats = np.stack([data[f"rot_{i}"] for i in range(4)], 1).astype(np.float64)  # (w, x, y, z)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    turned = (Rotation.from_matrix(rot) * Rotation.from_quat(quats[:, [1, 2, 3, 0]])).as_quat()[:, [3, 0, 1, 2]]
    for i in range(4):
        out[f"rot_{i}"] = turned[:, i]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["ply", "format binary_little_endian 1.0", f"element vertex {len(out)}"]
    lines += [f"property float {n}" for n in KEEP] + ["end_header"]
    with args.output.open("wb") as handle:
        handle.write(("\n".join(lines) + "\n").encode("ascii"))
        out.tofile(handle)
    camera = {"position": to_web(opening).round(4).tolist(), "target": [0.0, 0.0, 0.0], "up": [0.0, 1.0, 0.0],
              "gaussians": int(len(out))}
    args.output.with_suffix(".json").write_text(json.dumps(camera, indent=1), encoding="utf-8")
    print(f"{len(out):,} Gaussians -> {args.output} ({args.output.stat().st_size / 1e6:.0f} MB); "
          f"opening camera at {camera['position']}")


if __name__ == "__main__":
    main()
