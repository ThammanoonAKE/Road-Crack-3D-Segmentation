"""Run Depth Anything 3 over a COLMAP image set and store raw depth maps.

This is stage 1 of the depth-prior pipeline. It deliberately does no alignment:
the network's output space is recorded in the manifest and
``align_depth_to_sparse.py`` solves the scale and shift against the COLMAP
sparse tracks afterwards.

DA3 is preferred over a per-image monocular model because it predicts a window
of frames jointly, so neighbouring views agree with each other. Inconsistent
neighbours are exactly what produces a lumpy fused mesh.

Camera conditioning is available on top of that, and the convention matters more
than it looks. Measured on 32 frames of this scene against held-out sparse
observations:

===================  ==========================
extrinsics passed    held-out median abs-rel
===================  ==========================
world-to-camera      0.92%
none                 1.01%
camera-to-world      2.89%
===================  ==========================

The camera-to-world rows are identical for both axis conventions, which means
the network rejected those poses and fell back to its own geometry. So DA3 wants
**world-to-camera**, and passing it the wrong way costs three times the error
while looking like it worked.

Preset and resolution were measured the same way, on the same 32 frames:

=====================  =========  =========  ==============
preset                 p50        p90        32 frames
=====================  =========  =========  ==============
da3-base               1.22%      1.75%      19 s
da3-large              0.92%      1.17%      11 s
DA3-LARGE-1.1          1.11%      1.30%      28 s
da3-giant              0.78%      0.92%      18 s
DA3-GIANT-1.1          0.78%      0.92%      18 s
=====================  =========  =========  ==============

=====================  =========  =========  ==============
process_res            p50        p90        map size
=====================  =========  =========  ==============
504                    0.89%      1.28%      504x280
756                    0.78%      0.92%      756x420
1036                   0.75%      0.98%      1036x588
=====================  =========  =========  ==============

``da3-giant`` at 1036 is the default. Note what the metric can and cannot say:
it scores depth accuracy at sparse feature points, which sit on texture rather
than on edges, so it saturates while the map keeps getting sharper. Source
images here are 1280x719, so 1036 is close to native.

This script needs its own environment. Depth Anything 3 pins ``numpy<2`` while
the training and meshing environment uses NumPy 2, so it runs from
``.venv-depth`` and communicates only through the files it writes.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from depth_io import (  # noqa: E402
    RAW_KIND_DEPTH,
    RAW_KIND_METRIC,
    DepthManifest,
    write_depth,
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sparse-dir", type=Path, default=None,
        help="COLMAP sparse model. When given, its poses and intrinsics "
             "condition the network and fix the frame order.",
    )
    parser.add_argument(
        "--model", default="depth-anything/DA3-GIANT-1.1",
        help="DA3 preset (da3-small/base/large/giant, da3metric-large, "
             "da3mono-large) or a Hugging Face repo id. DA3-GIANT-1.1 and "
             "DA3-GIANT ship byte-identical weights (same SHA256).",
    )
    parser.add_argument(
        "--process-res", type=int, default=1036,
        help="Network working resolution, and therefore the depth map size. "
             "504 is the upstream default. Accuracy at sparse points "
             "saturates around 756, but a larger map still carries sharper "
             "edges, which the sparse-point metric cannot see.",
    )
    parser.add_argument(
        "--view-window", type=int, default=8,
        help="Frames predicted jointly. Larger windows agree over a longer "
             "baseline but cost VRAM quadratically in attention.",
    )
    parser.add_argument(
        "--view-stride", type=int, default=0,
        help="Frames advanced per window; 0 means no overlap. With overlap, "
             "each frame keeps the window where it sits most centrally.",
    )
    parser.add_argument("--no-poses", action="store_true",
                        help="Ignore COLMAP poses even when --sparse-dir is given.")
    parser.add_argument(
        "--pose-convention",
        choices=("c2w_opencv", "w2c_opencv", "c2w_opengl", "w2c_opengl"),
        default="w2c_opencv",
        help="Extrinsic convention handed to the network. w2c_opencv is "
             "correct for DA3 and COLMAP; the others are kept so the choice "
             "stays measurable rather than assumed.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--start", type=int, default=0,
        help="First frame index in COLMAP name order. Keep the selection "
             "contiguous: the multi-view windows only help when the frames "
             "in one window actually see the same place.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true",
                        help="Skip images that already have a depth file.")
    args = parser.parse_args()
    if args.view_window < 1:
        parser.error("--view-window must be at least 1")
    if args.view_stride < 0:
        parser.error("--view-stride cannot be negative")
    if args.process_res < 112:
        parser.error("--process-res is too small to be useful")
    if args.start < 0:
        parser.error("--start cannot be negative")
    return args


def as_metric_flag(value) -> int:
    """Normalise DA3's ``is_metric``, which can arrive as an empty addict Dict."""
    if isinstance(value, (bool, int, float, np.integer, np.floating)):
        return int(bool(value))
    return 0


# OpenCV cameras look down +z with +y down; OpenGL look down -z with +y up.
CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def colmap_order(sparse_dir: Path, images_dir: Path, convention: str = "w2c_opencv"):
    """Frame names, extrinsics and K matrices, in COLMAP name order."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "asset_pipeline"))
    from colmap_model import read_cameras, read_frames

    cameras = read_cameras(sparse_dir / "cameras.bin")
    frames = read_frames(sparse_dir / "images.bin")
    frames.sort(key=lambda frame: frame.name)

    names: list[str] = []
    poses: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    for frame in frames:
        if not (images_dir / frame.name).exists():
            continue
        camera = cameras[frame.camera_id]
        # COLMAP already stores world-to-camera, which is what DA3 wants.
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = frame.rotation
        world_to_camera[:3, 3] = frame.translation
        fx, fy, cx, cy = camera.pinhole()
        pose = np.linalg.inv(world_to_camera)
        if convention.endswith("opengl"):
            # Rotate the camera basis, not the world: right-multiply.
            pose = pose @ CV_TO_GL
        if convention.startswith("w2c"):
            pose = np.linalg.inv(pose)
        names.append(frame.name)
        poses.append(pose)
        intrinsics.append(
            np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        )
    return names, np.asarray(poses), np.asarray(intrinsics)


def plan_windows(
    count: int, window: int, stride: int
) -> list[tuple[int, list[int], list[int]]]:
    """Split frame indices into windows and assign each frame to exactly one.

    With overlap, a frame is claimed by the window whose centre it is nearest,
    because a view predicted at the edge of a window has the least context.
    """
    step = stride if stride > 0 else window
    starts = list(range(0, max(1, count), step))
    spans = []
    for start in starts:
        stop = min(start + window, count)
        if stop > start:
            # A short final window reaches back for context. With one view, DA3's pose alignment
            # (Umeyama) is degenerate and raises; 513 frames in windows of 8 left exactly one.
            spans.append((max(0, stop - window), stop))
        if stop >= count:
            break

    owner: dict[int, int] = {}
    best: dict[int, float] = {}
    for index, (start, stop) in enumerate(spans):
        centre = (start + stop - 1) / 2.0
        for frame in range(start, stop):
            distance = abs(frame - centre)
            if frame not in best or distance < best[frame]:
                best[frame] = distance
                owner[frame] = index

    plan = []
    for index, (start, stop) in enumerate(spans):
        members = list(range(start, stop))
        keep = [frame for frame in members if owner[frame] == index]
        if keep:
            plan.append((index, members, keep))
    # Every frame must be produced exactly once.
    produced = sorted(frame for _, _, keep in plan for frame in keep)
    if produced != list(range(count)):
        raise RuntimeError("Window plan does not cover every frame exactly once")
    return plan


def main() -> None:
    args = parse_args()
    images_dir = args.images.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    maps_subdir = "raw"

    poses = intrinsics = None
    if args.sparse_dir is not None:
        names, poses, intrinsics = colmap_order(
            args.sparse_dir.resolve(), images_dir, args.pose_convention
        )
        if args.no_poses:
            poses = intrinsics = None
        print(f"Frames from COLMAP model: {len(names)}")
    else:
        names = sorted(
            path.name for path in images_dir.iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        )
        print(f"Frames from directory listing: {len(names)}")
    if not names:
        raise SystemExit(f"No images found under {images_dir}")

    if args.start or args.limit:
        stop = args.start + args.limit if args.limit else len(names)
        names = names[args.start : stop]
        if poses is not None:
            poses = poses[args.start : stop]
            intrinsics = intrinsics[args.start : stop]
        if not names:
            raise SystemExit(
                f"--start {args.start} is past the end of the {len(names)} frames"
            )
        print(f"Frame window: [{args.start}, {args.start + len(names)})")

    import torch
    from depth_anything_3.api import DepthAnything3

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    if device != args.device:
        print(f"CUDA unavailable; falling back to {device}")
    print(f"Loading {args.model}")
    if "/" in args.model:
        model = DepthAnything3.from_pretrained(args.model)
    else:
        model = DepthAnything3.from_pretrained(f"depth-anything/{args.model.upper()}")
    model = model.to(device)
    model.device = device
    model.eval()

    plan = plan_windows(len(names), args.view_window, args.view_stride)
    print(
        f"Inference plan: {len(plan)} windows of up to {args.view_window} views, "
        f"process_res={args.process_res}, "
        f"poses={'COLMAP' if poses is not None else 'predicted by the network'}"
    )

    manifest = DepthManifest(
        model=args.model,
        raw_kind=RAW_KIND_DEPTH,
        images_dir=str(images_dir),
        image_count=len(names),
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        extra={
            "process_res": args.process_res,
            "view_window": args.view_window,
            "view_stride": args.view_stride or args.view_window,
            "camera_conditioned": poses is not None,
            "pose_convention": args.pose_convention if poses is not None else None,
            "sparse_dir": str(args.sparse_dir) if args.sparse_dir else None,
        },
    )
    if args.resume and (output_dir / "depth_manifest.json").exists():
        existing = DepthManifest.load(output_dir).by_name()
    else:
        existing = {}

    metric_flags: set[int] = set()
    for window_index, members, keep in plan:
        pending = [names[i] for i in keep]
        if args.resume and all(
            name in existing and (output_dir / existing[name].file).exists()
            for name in pending
        ):
            for name in pending:
                manifest.records.append(existing[name])
            print(f"  window {window_index}: reused {len(pending)} maps")
            continue

        paths = [str(images_dir / names[i]) for i in members]
        kwargs = {}
        if poses is not None:
            kwargs["extrinsics"] = poses[members]
            kwargs["intrinsics"] = intrinsics[members]
        prediction = model.inference(
            paths,
            process_res=args.process_res,
            export_dir=None,
            **kwargs,
        )
        metric_flags.add(as_metric_flag(prediction.is_metric))

        depth = np.asarray(prediction.depth)
        conf = np.asarray(prediction.conf) if prediction.conf is not None else None
        sky = np.asarray(prediction.sky) if prediction.sky is not None else None
        if len(depth) != len(members):
            raise RuntimeError(
                f"Window {window_index}: model returned {len(depth)} maps for "
                f"{len(members)} inputs"
            )

        for slot, frame in enumerate(members):
            if frame not in keep:
                continue
            manifest.records.append(
                write_depth(
                    output_dir,
                    names[frame],
                    depth[slot],
                    conf=conf[slot] if conf is not None else None,
                    sky=sky[slot].astype(bool) if sky is not None else None,
                    window=window_index,
                    subdir=maps_subdir,
                )
            )
        print(
            f"  window {window_index}: {len(members)} views -> wrote "
            f"{len(keep)} maps ({len(manifest.records)}/{len(names)} done)",
            flush=True,
        )
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if metric_flags == {1}:
        manifest.raw_kind = RAW_KIND_METRIC
    manifest.extra["is_metric_flags"] = sorted(metric_flags)
    manifest.records.sort(key=lambda record: record.image_name)
    path = manifest.save(output_dir)

    summary = {
        "model": manifest.model,
        "raw_kind": manifest.raw_kind,
        "images": len(manifest.records),
        "windows": len(plan),
        "camera_conditioned": poses is not None,
        "process_res": args.process_res,
        "map_resolution": [manifest.records[0].width, manifest.records[0].height],
        "stored_confidence": manifest.records[0].has_conf,
        "stored_sky_mask": manifest.records[0].has_sky,
        "manifest": str(path),
    }
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
