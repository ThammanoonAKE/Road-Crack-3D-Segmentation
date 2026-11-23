"""Render an orbit video of one clip's splat: a full circle round the crack, landscape.

The camera circles the spot the phone kept aiming at, at the phone's median distance from it, a
little steeper than the phone was held. The road plane is a RANSAC fit to the SfM points, "up"
points towards the cameras, and the horizon stays level on it. Several clips went only part way round
their crack; a steeper look keeps the unfilmed side mostly the road surface, which renders cleanly
from above, rather than the curbs and plants beyond it. The orbit starts from the photo that looks
most squarely at the crack. Frames use the photos' own focal length and landscape shape.

A clip that walks along the road (``--area all``) gets a stadium-shaped loop instead. The camera goes
down one side of the filmed strip, round its end and back up the other, always looking at the nearest
stretch from the phone's viewing distance. Framing the whole strip at once would mean looking from
ten times the phone's height, where cracks vanish and the roadside floaters dominate.

    python -I gaussian_splatting/render_clip_tour.py --scene <data>/road_crack/151605/scene \
        --ply runs/road_crack_151605/ply/point_cloud_19999.ply --output hole_1.mp4 --label "Hole 1"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
W, H, FPS = 1920, 1080, 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, required=True, help="The scene the splat trained on.")
    parser.add_argument("--ply", type=Path, required=True, help="The splat, in that scene's training frame.")
    parser.add_argument("--output", type=Path, required=True, help=".mp4, or .jpg for a sheet of stills")
    parser.add_argument("--label", default="", help="Text drawn in the top-left corner.")
    parser.add_argument("--orbit-seconds", type=float, default=20.0, help="Time for one full circle.")
    parser.add_argument("--pitch", type=float, default=50.0,
                        help="Degrees below the horizon the camera looks; the phones were held at about 38.")
    parser.add_argument("--area", choices=("crack", "all"), default="crack",
                        help="crack: a circle round the spot the phone kept aiming at. all, for a clip that walks "
                             "along the road: a stadium-shaped loop round everything the phone aimed at, looking "
                             "at the nearest stretch from the phone's viewing distance.")
    parser.add_argument("--back", type=float, default=1.0,
                        help="With --area all: viewing distance as a multiple of the phone's.")
    parser.add_argument("--stills", type=int, default=0, help="With a .jpg output: how many stills.")
    return parser.parse_args()


def fit_plane(points: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """RANSAC plane (point, unit normal) through most of the points."""
    span = np.percentile(np.linalg.norm(points - np.median(points, 0), axis=1), 50)
    best, best_count = None, -1
    for _ in range(500):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        count = int((np.abs((points - a) @ n) < 0.02 * span).sum())
        if count > best_count:
            best, best_count = (a, n), count
    a, n = best
    inliers = points[np.abs((points - a) @ n) < 0.02 * span]
    centre = inliers.mean(0)
    n = np.linalg.svd(inliers - centre, full_matrices=False)[2][2]  # refit on the inliers
    return centre, n


def look_rotation(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """OpenCV camera-to-world rotation (x right, y down, z forward), level on ``up``."""
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(fwd, right), fwd], 1)


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(ROOT / "third_party" / "gsplat" / "examples"))
    from datasets.colmap import Parser  # gsplat example code, not a package
    from gsplat import rasterization

    parser = Parser(data_dir=str(args.scene), factor=1, normalize=True, test_every=8)
    order = np.argsort(parser.image_names)  # frame names carry the time
    c2ws = parser.camtoworlds[order]
    K = parser.Ks_dict[parser.camera_ids[0]]
    focal = float(K[0, 0])
    centre, up = fit_plane(parser.points, np.random.default_rng(0))
    if (c2ws[:, :3, 3].mean(0) - centre) @ up < 0:
        up = -up
    eyes, fwds = c2ws[:, :3, 3], c2ws[:, :3, 2]
    flat = lambda v: v - (v @ up)[..., None] * up  # noqa: E731
    # Each camera aims at where its optical axis meets the road (or 3 units out if it looks up).
    dist = np.where(fwds @ up < -0.05, ((centre - eyes) @ up) / np.minimum(fwds @ up, -0.05), 3.0)
    aims = eyes + fwds * np.clip(dist, 0.3, 6.0)[:, None]

    # The crack is where the phone kept looking: the densest spot of its aim points, which on a long
    # walk along the road is not their median.
    phone_height = float(np.median((eyes - centre) @ up))
    radius = 0.5 * phone_height
    gap = np.linalg.norm(flat(aims[:, None] - aims[None]), axis=2)
    near = gap[np.argmax((gap < radius).sum(1))] < radius
    middle = aims[near].mean(0)
    middle -= ((middle - centre) @ up) * up
    # Start where the phone looked most squarely at it (path smoothed over ~0.6 s of frames).
    aims_s = gaussian_filter1d(aims, 3.6, axis=0, mode="nearest")
    eyes_s = gaussian_filter1d(eyes, 3.6, axis=0, mode="nearest")
    start = int(np.argmin(np.linalg.norm(flat(aims_s - middle), axis=1)))
    e1 = flat((eyes_s[start] - middle)[None])[0]
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    distance = float(np.median(np.linalg.norm(eyes - middle, axis=1)))
    pitch = np.radians(args.pitch)
    turn = int(args.orbit_seconds * FPS)
    ease = 0.5 - 0.5 * np.cos(np.pi * np.arange(turn) / (turn - 1))

    if args.area == "crack":
        def orbit(angle: float) -> tuple[np.ndarray, np.ndarray]:
            out = np.cos(angle) * e1 + np.sin(angle) * e2
            return middle + distance * (np.cos(pitch) * out + np.sin(pitch) * up), middle

        loop = [orbit(a) for a in 2 * np.pi * ease]
    else:
        # The aim points' 5-95% extent along their own axes; the camera keeps the phone's viewing distance
        # from the nearest point of the strip's centre line, so it sees the whole length on the way round.
        spot = flat(aims - centre)
        axis, across = np.linalg.svd(spot - spot.mean(0), full_matrices=False)[2][:2]
        along = spot @ axis
        lo, hi = np.percentile(along, 5), np.percentile(along, 95)
        middle = centre + spot.mean(0) - (spot.mean(0) @ axis) * axis + 0.5 * (lo + hi) * axis
        half = 0.5 * (hi - lo)
        view = args.back * float(np.median(np.linalg.norm(eyes - aims, axis=1)))
        reach, lift = view * np.cos(pitch), view * np.sin(pitch)
        t = np.linspace(0, 2 * np.pi, 4000)
        start_t = np.arctan2((eyes_s[start] - middle) @ across, (eyes_s[start] - middle) @ axis)
        t = t + start_t
        cams = (middle + np.outer((half + reach) * np.cos(t), axis) + np.outer(reach * np.sin(t), across)
                + lift * up)
        aimed = middle + np.outer(half * np.cos(t), axis)
        arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(cams, axis=0), axis=1))])
        pick = np.interp(ease * arc[-1], arc, np.arange(len(t))).round().astype(int)
        loop = [(cams[i], aimed[i]) for i in pick]
        distance = view
    plan = [loop[0]] * FPS + loop + [loop[-1]] * FPS

    dev = "cuda"
    with args.ply.open("rb") as handle:
        props = []
        while (line := handle.readline().decode("ascii").strip()) != "end_header":
            if line.startswith("property"):
                props.append(line.split()[2])
        data = np.fromfile(handle, dtype=np.dtype([(n, "<f4") for n in props]))
    col = lambda names: torch.as_tensor(np.stack([data[n] for n in names], 1), device=dev)  # noqa: E731
    means, quats = col(["x", "y", "z"]), col([f"rot_{i}" for i in range(4)])
    scales = torch.exp(col([f"scale_{i}" for i in range(3)]))
    opacities = torch.sigmoid(torch.as_tensor(data["opacity"], device=dev))
    rest = col([f"f_rest_{i}" for i in range(45)]).reshape(-1, 3, 15).transpose(1, 2)
    colors = torch.cat([col(["f_dc_0", "f_dc_1", "f_dc_2"])[:, None], rest], 1)
    del data
    Kt = torch.tensor([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]], device=dev, dtype=torch.float32)[None]
    background = torch.tensor([[0.55, 0.56, 0.58]], device=dev)
    font = ImageFont.truetype("tahoma.ttf", 48)  # Pillow looks a bare file name up in the Windows fonts folder

    def render(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
        c2w = np.eye(4)
        c2w[:3, :3] = look_rotation(eye, target, up)
        c2w[:3, 3] = eye
        viewmat = torch.as_tensor(np.linalg.inv(c2w), device=dev, dtype=torch.float32)[None]
        with torch.no_grad():
            img, _, _ = rasterization(means, quats, scales, opacities, colors, viewmat, Kt, W, H,
                                      sh_degree=3, backgrounds=background, near_plane=0.05, packed=True)
        frame = (img[0].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
        if args.label:
            pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil)
            box = draw.textbbox((36, 30), args.label, font=font)
            draw.rectangle((box[0] - 16, box[1] - 12, box[2] + 16, box[3] + 12), fill=(0, 0, 0))
            draw.text((36, 30), args.label, font=font, fill=(255, 255, 255))
            frame = np.asarray(pil)
        return frame

    print(f"{len(plan)} frames ({len(plan) / FPS:.1f} s): orbit at {distance / phone_height:.2f}x the phone's "
          f"height, {args.pitch:.0f} deg down, starting at photo {start} of {len(eyes)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".jpg":
        pick = np.linspace(FPS, FPS + turn - 1, args.stills or 8).round().astype(int)
        shots = [cv2.resize(render(*plan[i]), (W // 4, H // 4), interpolation=cv2.INTER_AREA) for i in pick]
        shots += [np.zeros_like(shots[0])] * (-len(shots) % 4)
        sheet = np.vstack([np.hstack(shots[i:i + 4]) for i in range(0, len(shots), 4)])
        cv2.imwrite(str(args.output), sheet[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
        return
    writer = imageio.get_writer(str(args.output), fps=FPS, codec="libx264", quality=8, macro_block_size=8)
    for eye, target in plan:
        writer.append_data(render(eye, target))
    writer.close()
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
