"""Fit each AI depth map to the COLMAP sparse points of the image that made it.

A monocular network predicts depth only up to an unknown per-image affine
transform, so a raw map cannot be compared with COLMAP at all. COLMAP's
``points3D.bin`` records, for every sparse point, the exact pixel in every image
that observed it. Those observations are the only exact depth ground truth
available here, and they are what the scale and shift are solved against.

The fit space matters. Networks of the Depth Anything family emit
affine-invariant *inverse* depth, so the correct model is linear in inverse
depth::

    1 / z = scale * raw + shift

Fitting in depth space instead would bend the far field badly. ``--raw-kind``
selects the space and defaults to whatever the inference manifest recorded.

Robustness matters too: sparse tracks contain mismatches, and a single bad
correspondence moves a two-parameter least-squares fit a long way. Each image is
therefore fitted with RANSAC followed by Huber-weighted refinement, and the
reported error comes from correspondences held out of the fit.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "asset_pipeline"))

from colmap_model import read_cameras, read_frames  # noqa: E402
from depth_io import (  # noqa: E402
    RAW_KIND_INVERSE,
    RAW_KINDS,
    AlignmentRecord,
    DepthManifest,
    read_depth_bundle,
    write_aligned,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-dir", type=Path, required=True,
                        help="Output of run_ai_depth.py, containing depth_manifest.json.")
    parser.add_argument("--sparse-dir", type=Path, required=True,
                        help="COLMAP sparse model with cameras.bin, images.bin, points3D.bin.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-kind", choices=RAW_KINDS, default=None,
                        help="Override the manifest's declared prediction space.")
    parser.add_argument("--min-points", type=int, default=40,
                        help="Images with fewer usable observations are rejected.")
    parser.add_argument("--inlier-rel-tol", type=float, default=0.05,
                        help="Relative depth error still counted as an inlier.")
    parser.add_argument("--ransac-iters", type=int, default=400)
    parser.add_argument("--irls-iters", type=int, default=12)
    parser.add_argument("--holdout-fraction", type=float, default=0.3,
                        help="Share of observations kept out of the fit for scoring.")
    parser.add_argument("--max-median-abs-rel", type=float, default=0.12,
                        help="Reject an image whose held-out median error exceeds this.")
    parser.add_argument("--min-inlier-fraction", type=float, default=0.5)
    parser.add_argument("--depth-percentile-clip", type=float, default=0.1,
                        help="Clip the fitted depth to this percentile band of the "
                             "sparse depths, expanded by --clip-expand.")
    parser.add_argument("--clip-expand", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--write-maps", action="store_true",
                        help="Write aligned metric depth maps, not just the report.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only the first N images; 0 means all.")
    args = parser.parse_args()
    if not 0.0 <= args.holdout_fraction < 0.9:
        parser.error("--holdout-fraction must be in [0, 0.9)")
    if args.inlier_rel_tol <= 0:
        parser.error("--inlier-rel-tol must be positive")
    return args


IMAGE_HEADER = "<idddddddi"
POINT_HEADER = "<QdddBBBdQ"


def _read_struct(handle, fmt: str):
    size = struct.calcsize(fmt)
    raw = handle.read(size)
    if len(raw) != size:
        raise EOFError("Unexpected end of a COLMAP binary file")
    return struct.unpack(fmt, raw)


def read_observations(path: Path) -> dict[int, np.ndarray]:
    """image id -> structured array of (x, y, point3D_id) for its 2D features."""
    result: dict[int, np.ndarray] = {}
    dtype = np.dtype([("x", "<f8"), ("y", "<f8"), ("point_id", "<i8")])
    with path.open("rb") as handle:
        for _ in range(_read_struct(handle, "<Q")[0]):
            image_id = _read_struct(handle, IMAGE_HEADER)[0]
            while handle.read(1) != b"\0":
                pass
            points = _read_struct(handle, "<Q")[0]
            result[image_id] = np.frombuffer(
                handle.read(points * dtype.itemsize), dtype=dtype, count=points
            )
    return result


def read_point_xyz(path: Path) -> dict[int, tuple[float, float, float]]:
    points: dict[int, tuple[float, float, float]] = {}
    with path.open("rb") as handle:
        for _ in range(_read_struct(handle, "<Q")[0]):
            values = _read_struct(handle, POINT_HEADER)
            handle.seek(values[8] * 8, 1)
            points[values[0]] = (values[1], values[2], values[3])
    return points


def sample_bilinear(depth: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample a depth map at floating-point pixel coordinates."""
    height, width = depth.shape
    xc = np.clip(x, 0.0, width - 1.0)
    yc = np.clip(y, 0.0, height - 1.0)
    x0 = np.floor(xc).astype(np.int64)
    y0 = np.floor(yc).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = (xc - x0).astype(np.float64)
    fy = (yc - y0).astype(np.float64)
    top = depth[y0, x0] * (1.0 - fx) + depth[y0, x1] * fx
    bottom = depth[y1, x0] * (1.0 - fx) + depth[y1, x1] * fx
    return top * (1.0 - fy) + bottom * fy


def target_values(z: np.ndarray, raw_kind: str) -> np.ndarray:
    """The quantity the linear model predicts, given the network's output space."""
    if raw_kind == RAW_KIND_INVERSE:
        return 1.0 / z
    return z


def depth_from_fit(raw: np.ndarray, scale: float, shift: float, raw_kind: str) -> np.ndarray:
    """Invert the fitted linear model back to metric depth."""
    fitted = scale * raw + shift
    if raw_kind == RAW_KIND_INVERSE:
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = 1.0 / fitted
        depth[~np.isfinite(depth)] = 0.0
        depth[fitted <= 0.0] = 0.0
        return depth
    fitted[~np.isfinite(fitted)] = 0.0
    return np.maximum(fitted, 0.0)


def solve_linear(raw: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None):
    """Weighted least squares for target ~= scale * raw + shift."""
    if weights is None:
        weights = np.ones_like(raw)
    total = weights.sum()
    if total <= 0:
        return None
    mean_raw = (weights * raw).sum() / total
    mean_target = (weights * target).sum() / total
    centred = raw - mean_raw
    variance = (weights * centred * centred).sum()
    if variance <= 1e-18:
        return None
    scale = (weights * centred * (target - mean_target)).sum() / variance
    return float(scale), float(mean_target - scale * mean_raw)


def fit_scale_shift(
    raw: np.ndarray,
    z: np.ndarray,
    raw_kind: str,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[float, float, np.ndarray] | None:
    """RANSAC, then Huber-weighted refinement, in the network's own space."""
    target = target_values(z, raw_kind)
    count = len(raw)

    best_inliers: np.ndarray | None = None
    best_count = -1
    if count >= 2:
        for _ in range(max(1, args.ransac_iters)):
            i, j = rng.choice(count, size=2, replace=False)
            span = raw[j] - raw[i]
            if abs(span) < 1e-9:
                continue
            scale = (target[j] - target[i]) / span
            shift = target[i] - scale * raw[i]
            fitted_depth = depth_from_fit(raw, scale, shift, raw_kind)
            with np.errstate(divide="ignore", invalid="ignore"):
                error = np.abs(fitted_depth - z) / z
            inliers = np.isfinite(error) & (fitted_depth > 0) & (error < args.inlier_rel_tol)
            hits = int(inliers.sum())
            if hits > best_count:
                best_count, best_inliers = hits, inliers

    if best_inliers is None or best_count < 2:
        best_inliers = np.ones(count, dtype=bool)

    selected = best_inliers if best_inliers.sum() >= 2 else np.ones(count, dtype=bool)
    solution = solve_linear(raw[selected], target[selected])
    if solution is None:
        solution = solve_linear(raw, target)
    if solution is None:
        return None
    scale, shift = solution

    # Huber-weighted refinement over all correspondences, so a point rejected by
    # a single RANSAC sample can still come back if the refined fit supports it.
    for _ in range(max(0, args.irls_iters)):
        residual = (scale * raw + shift) - target
        spread = np.median(np.abs(residual - np.median(residual)))
        sigma = max(1.4826 * float(spread), 1e-12)
        delta = 1.345 * sigma
        magnitude = np.abs(residual)
        weights = np.where(magnitude <= delta, 1.0, delta / np.maximum(magnitude, 1e-18))
        refined = solve_linear(raw, target, weights)
        if refined is None:
            break
        if abs(refined[0] - scale) < 1e-12 and abs(refined[1] - shift) < 1e-12:
            scale, shift = refined
            break
        scale, shift = refined

    fitted_depth = depth_from_fit(raw, scale, shift, raw_kind)
    with np.errstate(divide="ignore", invalid="ignore"):
        error = np.abs(fitted_depth - z) / z
    inliers = np.isfinite(error) & (fitted_depth > 0) & (error < args.inlier_rel_tol)
    return scale, shift, inliers


def main() -> None:
    args = parse_args()
    depth_dir = args.depth_dir.resolve()
    sparse_dir = args.sparse_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = DepthManifest.load(depth_dir)
    raw_kind = args.raw_kind or manifest.raw_kind
    print(f"Depth model     : {manifest.model}")
    print(f"Prediction space: {raw_kind}")

    cameras = read_cameras(sparse_dir / "cameras.bin")
    frames = read_frames(sparse_dir / "images.bin")
    observations = read_observations(sparse_dir / "images.bin")
    points = read_point_xyz(sparse_dir / "points3D.bin")
    print(f"COLMAP model    : {len(frames)} frames, {len(points):,} sparse points")

    records = manifest.by_name()
    rng = np.random.default_rng(args.seed)
    results: list[AlignmentRecord] = []
    written: dict[str, str] = {}
    frames_to_do = frames[: args.limit] if args.limit else frames

    for index, frame in enumerate(frames_to_do, start=1):
        record = records.get(frame.name)
        if record is None:
            results.append(AlignmentRecord(
                frame.name, 0.0, 0.0, 0, 0, float("nan"), float("nan"), float("nan"),
                False, "no depth map for this image",
            ))
            continue

        camera = cameras[frame.camera_id]
        entries = observations[frame.id]
        entries = entries[entries["point_id"] >= 0]
        if not len(entries):
            results.append(AlignmentRecord(
                frame.name, 0.0, 0.0, 0, 0, float("nan"), float("nan"), float("nan"),
                False, "no sparse observations",
            ))
            continue

        known = np.array([pid in points for pid in entries["point_id"]])
        entries = entries[known]
        xyz = np.array([points[pid] for pid in entries["point_id"]], dtype=np.float64)
        if not len(xyz):
            results.append(AlignmentRecord(
                frame.name, 0.0, 0.0, 0, 0, float("nan"), float("nan"), float("nan"),
                False, "observations reference missing points",
            ))
            continue

        # Camera-space depth is exactly what a depth map should contain.
        local = xyz @ frame.rotation.T + frame.translation
        z = local[:, 2]

        depth, conf, sky = read_depth_bundle(depth_dir, record)
        # The network may predict at its own resolution; map image pixels across.
        sx = record.width / camera.width
        sy = record.height / camera.height
        raw = sample_bilinear(depth, entries["x"] * sx, entries["y"] * sy)

        usable = (
            (z > 1e-6)
            & np.isfinite(raw)
            & (entries["x"] >= 0) & (entries["x"] < camera.width)
            & (entries["y"] >= 0) & (entries["y"] < camera.height)
        )
        raw, z = raw[usable], z[usable]
        if len(raw) < args.min_points:
            results.append(AlignmentRecord(
                frame.name, 0.0, 0.0, int(len(raw)), 0,
                float("nan"), float("nan"), float("nan"), False,
                f"only {len(raw)} usable observations, need {args.min_points}",
            ))
            continue

        # Hold part of the track out so the reported error is not the fit's own.
        order = rng.permutation(len(raw))
        holdout = int(round(len(raw) * args.holdout_fraction))
        holdout = min(holdout, len(raw) - args.min_points // 2)
        test_index = order[:holdout]
        fit_index = order[holdout:]
        if len(fit_index) < 2:
            fit_index, test_index = order, order

        fitted = fit_scale_shift(raw[fit_index], z[fit_index], raw_kind, args, rng)
        if fitted is None:
            results.append(AlignmentRecord(
                frame.name, 0.0, 0.0, int(len(raw)), 0,
                float("nan"), float("nan"), float("nan"), False,
                "degenerate fit: the raw depth has no usable variation",
            ))
            continue
        scale, shift, inliers = fitted

        score_index = test_index if len(test_index) else fit_index
        scored = depth_from_fit(raw[score_index], scale, shift, raw_kind)
        with np.errstate(divide="ignore", invalid="ignore"):
            error = np.abs(scored - z[score_index]) / z[score_index]
        error = error[np.isfinite(error)]
        if not len(error):
            results.append(AlignmentRecord(
                frame.name, scale, shift, int(len(raw)), int(inliers.sum()),
                float("nan"), float("nan"), float("nan"), False,
                "no finite held-out residuals",
            ))
            continue

        median_abs_rel = float(np.median(error))
        p95 = float(np.percentile(error, 95))
        rmse = float(np.sqrt(np.mean((scored - z[score_index]) ** 2)))
        inlier_fraction = float(inliers.mean())

        accepted = True
        reason = ""
        if median_abs_rel > args.max_median_abs_rel:
            accepted, reason = False, (
                f"held-out median abs-rel {median_abs_rel:.3f} exceeds "
                f"{args.max_median_abs_rel:.3f}"
            )
        elif inlier_fraction < args.min_inlier_fraction:
            accepted, reason = False, (
                f"inlier fraction {inlier_fraction:.3f} below "
                f"{args.min_inlier_fraction:.3f}"
            )

        results.append(AlignmentRecord(
            image_name=frame.name,
            scale=float(scale),
            shift=float(shift),
            sparse_points=int(len(raw)),
            inliers=int(inliers.sum()),
            median_abs_rel=median_abs_rel,
            p95_abs_rel=p95,
            rmse=rmse,
            accepted=accepted,
            reason=reason,
        ))

        if args.write_maps and accepted:
            metric = depth_from_fit(depth.astype(np.float64), scale, shift, raw_kind)
            low = np.percentile(z, args.depth_percentile_clip)
            high = np.percentile(z, 100.0 - args.depth_percentile_clip)
            span = max(high - low, 1e-6)
            near = max(low - args.clip_expand * span, 1e-6)
            far = high + args.clip_expand * span
            valid = np.isfinite(metric) & (metric > near) & (metric < far)
            if sky is not None:
                valid &= ~sky
            metric = np.where(valid, metric, 0.0)
            written[frame.name] = write_aligned(
                output_dir / "aligned", frame.name, metric, valid, conf
            )

        if index % 50 == 0 or index == len(frames_to_do):
            done = [r for r in results if r.accepted]
            print(
                f"  {index}/{len(frames_to_do)} images | accepted {len(done)} | "
                f"median abs-rel so far "
                f"{np.median([r.median_abs_rel for r in done]) if done else float('nan'):.4f}",
                flush=True,
            )

    accepted = [r for r in results if r.accepted]
    summary = {
        "depth_dir": str(depth_dir),
        "sparse_dir": str(sparse_dir),
        "model": manifest.model,
        "raw_kind": raw_kind,
        "fit_space": (
            "inverse depth (1/z = scale * raw + shift)"
            if raw_kind == RAW_KIND_INVERSE
            else "depth (z = scale * raw + shift)"
        ),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "images_total": len(results),
        "images_accepted": len(accepted),
        "images_rejected": len(results) - len(accepted),
        "holdout_fraction": args.holdout_fraction,
        "inlier_rel_tol": args.inlier_rel_tol,
        "aligned_maps_written": len(written),
        "aligned_dir": str(output_dir / "aligned") if written else None,
    }
    if accepted:
        medians = np.array([r.median_abs_rel for r in accepted])
        scales = np.array([r.scale for r in accepted])
        shifts = np.array([r.shift for r in accepted])
        summary["heldout_median_abs_rel"] = {
            "p50": float(np.median(medians)),
            "p90": float(np.percentile(medians, 90)),
            "max": float(medians.max()),
        }
        summary["mean_inliers_per_image"] = float(
            np.mean([r.inliers for r in accepted])
        )
        summary["mean_points_per_image"] = float(
            np.mean([r.sparse_points for r in accepted])
        )
        # Per-image affine parameters should vary; a constant scale would mean
        # the network already agreed with COLMAP up to one global factor.
        summary["scale_spread"] = {
            "p50": float(np.median(scales)),
            "relative_iqr": float(
                (np.percentile(scales, 75) - np.percentile(scales, 25))
                / max(abs(np.median(scales)), 1e-12)
            ),
        }
        summary["shift_spread"] = {
            "p50": float(np.median(shifts)),
            "iqr": float(np.percentile(shifts, 75) - np.percentile(shifts, 25)),
        }

    report = {
        "summary": summary,
        "images": [asdict(record) for record in results],
        "aligned_files": written,
    }
    path = output_dir / "alignment_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
