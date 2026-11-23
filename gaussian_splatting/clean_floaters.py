"""Remove floaters from a trained splat, and optionally what the photos saw too poorly to be sharp.

For every N-th training view the splat's own expected depth is rendered at reduced size. A
Gaussian whose centre projects in front of IN_FRONT times the nearest depth round its pixel lies in
front of what that camera saw; most of the views that see it saying so marks it as floating in empty
air. Real surfaces are what the views see, so they never vote against themselves.

With --min-views / --max-distance it also removes surfaces that came out blurred: a Gaussian is kept
only if at least MIN_VIEWS of the used views see it unoccluded (centre no deeper than 5% behind the
depth there) and the nearest of them is within MAX_DISTANCE times the photos' typical viewing distance
(the median depth at the image centre). Roadside plants and kerbs the phone only glimpsed from afar
fail this; the ground it walked round passes.

--max-footprint judges sharpness directly. In each view that sees a Gaussian unoccluded, one pixel
covers depth / focal / cos(incidence) of its surface; the surface normal is the Gaussian's shortest
axis. A Gaussian is kept only if its best view resolves it within MAX_FOOTPRINT times the median best
footprint. Distance alone missed road the phone looked along: near enough, but so slanted that the
splat smeared it.

The PLY stays in the trainer's frame, so render_clip_tour.py and gsplat tools read it unchanged.

    python -I gaussian_splatting/clean_floaters.py --scene C:/3D-segmentation-data/road_crack/152410/scene \
        --ply runs/road_crack_152410_v2/ply/point_cloud_39999.ply --output runs/road_crack_152410_v2/clean.ply
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, required=True, help="The scene the splat trained on.")
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--every", type=int, default=2, help="Use every N-th training view.")
    parser.add_argument("--scale-down", type=int, default=4, help="Render depth at 1/N size.")
    parser.add_argument("--in-front", type=float, default=0.85,
                        help="A view votes 'free' when the centre is nearer than this times the depth seen there.")
    parser.add_argument("--min-free", type=int, default=3, help="Votes needed, and more than half the views seeing it.")
    parser.add_argument("--min-views", type=int, default=0,
                        help="Also remove Gaussians fewer than this many used views see unoccluded (0: off).")
    parser.add_argument("--max-distance", type=float, default=0.0,
                        help="Also remove Gaussians no view sees from within this multiple of the typical "
                             "viewing distance (0: off).")
    parser.add_argument("--max-footprint", type=float, default=0.0,
                        help="Also remove Gaussians whose sharpest view resolves them more coarsely than this "
                             "multiple of the median (0: off).")
    parser.add_argument("--removed", action="store_true", help="Also write the removed Gaussians, for a look.")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    return args


def read_ply(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("rb") as handle:
        header = []
        while (line := handle.readline().decode("ascii").rstrip("\n")) != "end_header":
            header.append(line)
        names = [h.split()[2] for h in header if h.startswith("property")]
        return header, np.fromfile(handle, dtype=np.dtype([(n, "<f4") for n in names]))


def write_ply(path: Path, header: list[str], data: np.ndarray) -> None:
    lines = [f"element vertex {len(data)}" if h.startswith("element vertex") else h for h in header]
    with path.open("wb") as handle:
        handle.write(("\n".join(lines + ["end_header"]) + "\n").encode("ascii"))
        data.tofile(handle)


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(ROOT / "third_party" / "gsplat" / "examples"))
    from datasets.colmap import Parser  # gsplat example code, not a package
    from gsplat import rasterization

    parser = Parser(data_dir=str(args.scene), factor=1, normalize=True, test_every=8)
    header, data = read_ply(args.ply)
    dev = "cuda"
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=dev)  # noqa: E731
    means = t(np.stack([data[k] for k in "xyz"], 1))
    quats = t(np.stack([data[f"rot_{i}"] for i in range(4)], 1))
    scales = torch.exp(t(np.stack([data[f"scale_{i}"] for i in range(3)], 1)))
    opacities = torch.sigmoid(t(data["opacity"]))
    colors = torch.zeros_like(means)
    free = torch.zeros(len(data), dtype=torch.int32, device=dev)
    seen = torch.zeros(len(data), dtype=torch.int32, device=dev)
    visible = torch.zeros(len(data), dtype=torch.int32, device=dev)
    closest = torch.full((len(data),), float("inf"), device=dev)
    footprint = torch.full((len(data),), float("inf"), device=dev)
    centre_depths = []
    # Surface normal of each Gaussian: the rotation's column along its shortest axis.
    w_, x_, y_, z_ = (quats / quats.norm(dim=1, keepdim=True)).unbind(1)
    rot = torch.stack([
        torch.stack([1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ - w_ * z_), 2 * (x_ * z_ + w_ * y_)], 1),
        torch.stack([2 * (x_ * y_ + w_ * z_), 1 - 2 * (x_ * x_ + z_ * z_), 2 * (y_ * z_ - w_ * x_)], 1),
        torch.stack([2 * (x_ * z_ - w_ * y_), 2 * (y_ * z_ + w_ * x_), 1 - 2 * (x_ * x_ + y_ * y_)], 1)], 1)
    normals = rot.gather(2, scales.argmin(1)[:, None, None].expand(-1, 3, 1))[..., 0]
    del rot, w_, x_, y_, z_
    full_focal = float(parser.Ks_dict[parser.camera_ids[0]][0, 0])

    camera = parser.camera_ids[0]
    w, h = (v // args.scale_down for v in parser.imsize_dict[camera])
    K = parser.Ks_dict[camera].copy()
    K[:2] /= args.scale_down
    Kt = t(K.astype(np.float32))[None]
    views = parser.camtoworlds[::args.every]
    for n, c2w in enumerate(views):
        w2c = np.linalg.inv(c2w).astype(np.float32)
        with torch.no_grad():
            ed, alpha, _ = rasterization(means, quats, scales, opacities, colors, t(w2c)[None], Kt, w, h,
                                         render_mode="ED", near_plane=0.01, packed=True)
            depth = torch.where(alpha[0, ..., 0] > 0.3, ed[0, ..., 0], torch.full_like(ed[0, ..., 0], float("inf")))
            nearest = -F.max_pool2d(-depth[None, None], 5, 1, 2)[0, 0]
            farthest = F.max_pool2d(torch.nan_to_num(depth, posinf=0)[None, None], 5, 1, 2)[0, 0]
            cam = means @ t(w2c[:3, :3]).T + t(w2c[:3, 3])
            z = cam[:, 2]
            u = K[0, 0] * cam[:, 0] / z + K[0, 2]
            v = K[1, 1] * cam[:, 1] / z + K[1, 2]
            idx = torch.nonzero((z > 0.01) & (u >= 0) & (u < w) & (v >= 0) & (v < h))[:, 0]
            ui, vi, zi = u[idx].long(), v[idx].long(), z[idx]
            is_free = zi < args.in_front * nearest[vi, ui]
            free[idx] += is_free.int()
            seen[idx] += (is_free | (zi < 1.1 * farthest[vi, ui])).int()
            unoccluded = zi <= 1.05 * depth[vi, ui]
            visible[idx] += unoccluded.int()
            closest[idx[unoccluded]] = torch.minimum(closest[idx[unoccluded]], zi[unoccluded])
            if args.max_footprint:
                seen_idx = idx[unoccluded]
                ray = cam[seen_idx] / cam[seen_idx].norm(dim=1, keepdim=True)
                incidence = (ray * (normals[seen_idx] @ t(w2c[:3, :3]).T)).sum(1).abs().clamp(min=0.05)
                footprint[seen_idx] = torch.minimum(footprint[seen_idx], zi[unoccluded] / full_focal / incidence)
            middle = depth[h // 3:2 * h // 3, w // 3:2 * w // 3]
            middle = middle[torch.isfinite(middle)]
            if len(middle):
                centre_depths.append(float(middle.median()))
        if n % 50 == 0:
            print(f"  view {n}/{len(views)}", flush=True)
    free, seen = free.cpu().numpy(), seen.cpu().numpy()
    visible, closest, footprint = visible.cpu().numpy(), closest.cpu().numpy(), footprint.cpu().numpy()
    floating = (free >= args.min_free) & (free > 0.5 * np.maximum(seen, 1))
    poor = np.zeros(len(data), dtype=bool)
    if args.min_views:
        poor |= visible < args.min_views
    if args.max_distance:
        typical = float(np.median(centre_depths))
        poor |= closest > args.max_distance * typical
        print(f"typical viewing distance {typical:.3f}; keeping what some view saw within "
              f"{args.max_distance * typical:.3f}", flush=True)
    if args.max_footprint:
        median = float(np.median(footprint[np.isfinite(footprint)]))
        poor |= footprint > args.max_footprint * median
        print(f"median best footprint {median:.5f} per pixel; keeping what some view resolved within "
              f"{args.max_footprint * median:.5f}", flush=True)
    poor &= ~floating
    drop = floating | poor
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_ply(args.output, header, data[~drop])
    if args.removed:
        write_ply(args.output.with_suffix(".removed.ply"), header, data[drop])
    print(f"removed {floating.sum():,} floating and {poor.sum():,} poorly seen of {len(data):,} Gaussians "
          f"({drop.sum() / len(data):.1%}) -> {args.output}")


if __name__ == "__main__":
    main()
