"""Prepare a deterministic dense COLMAP initialization for gsplat.

The source and labelled PLYs must have identical point order.  Every semantic
point is retained when it fits in the requested budget; background points are
sampled evenly over the full fused-cloud order.  Arrays are written as .npy
files so the trainer can memory-map and validate them independently.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SOURCE_DTYPE = np.dtype(
    [
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
)
LABEL_DTYPE = np.dtype(SOURCE_DTYPE.descr + [("label", "<u2"), ("votes", "<u2")])


def read_binary_ply(path: Path, dtype: np.dtype) -> tuple[np.memmap, int]:
    header = bytearray()
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"Missing end_header in {path}")
            header.extend(line)
            if line.rstrip() == b"end_header":
                break
            if len(header) > 1024 * 1024:
                raise ValueError(f"Unexpectedly large PLY header in {path}")
    lines = header.decode("ascii").splitlines()
    if "format binary_little_endian 1.0" not in lines:
        raise ValueError(f"Only binary little-endian PLY is supported: {path}")
    vertex_line = next(line for line in lines if line.startswith("element vertex "))
    count = int(vertex_line.rsplit(" ", 1)[1])
    expected_size = len(header) + count * dtype.itemsize
    if path.stat().st_size != expected_size:
        raise ValueError(
            f"PLY layout mismatch for {path}: expected {expected_size} bytes, "
            f"found {path.stat().st_size}"
        )
    return np.memmap(path, mode="r", dtype=dtype, offset=len(header), shape=(count,)), count


def semantic_indices(labels: np.ndarray, count: int, chunk_size: int) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        local = np.flatnonzero(labels[start:end] > 0)
        if local.size:
            chunks.append(local.astype(np.int64) + start)
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)


def evenly_sample_background(
    labels: np.ndarray, count: int, requested: int
) -> np.ndarray:
    if requested == 0:
        return np.empty(0, dtype=np.int64)
    background_fraction = float(np.count_nonzero(labels == 0)) / count
    candidate_count = min(count, int(np.ceil(requested / background_fraction * 1.01)))
    while True:
        candidates = np.linspace(0, count - 1, candidate_count, dtype=np.int64)
        background = candidates[labels[candidates] == 0]
        if background.size >= requested:
            return background[:requested]
        if candidate_count == count:
            raise RuntimeError("Not enough background points for requested sample")
        candidate_count = min(count, int(candidate_count * 1.1) + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ply", type=Path, required=True)
    parser.add_argument("--labelled-ply", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--points", type=int, default=5_000_000)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    args = parser.parse_args()
    if args.points < 1 or args.chunk_size < 1:
        raise ValueError("--points and --chunk-size must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")

    source, source_count = read_binary_ply(args.source_ply.resolve(), SOURCE_DTYPE)
    labelled, labelled_count = read_binary_ply(args.labelled_ply.resolve(), LABEL_DTYPE)
    if source_count != labelled_count:
        raise ValueError(f"Point counts differ: source={source_count}, labelled={labelled_count}")
    if args.points > source_count:
        raise ValueError(f"Requested {args.points:,} points from only {source_count:,}")

    # Verify that label fusion preserved exact point order before relying on
    # cross-file RGB and semantic fields.
    probes = np.linspace(0, source_count - 1, 4096, dtype=np.int64)
    for field in ("x", "y", "z"):
        if not np.array_equal(source[field][probes], labelled[field][probes]):
            raise ValueError(f"Source/labelled PLY point order differs in field {field}")

    semantic = semantic_indices(labelled["label"], source_count, args.chunk_size)
    if semantic.size > args.points:
        # This branch is intended only for small smoke datasets.
        take = np.linspace(0, semantic.size - 1, args.points, dtype=np.int64)
        selected = semantic[take]
    else:
        background = evenly_sample_background(
            labelled["label"], source_count, args.points - semantic.size
        )
        selected = np.sort(np.concatenate((semantic, background)))
    if selected.size != args.points or np.unique(selected).size != args.points:
        raise RuntimeError("Sampling did not produce the requested number of unique points")

    args.output_dir.mkdir(parents=True)
    xyz_out = np.lib.format.open_memmap(
        args.output_dir / "xyz.npy", mode="w+", dtype="<f4", shape=(args.points, 3)
    )
    normals_out = np.lib.format.open_memmap(
        args.output_dir / "normals.npy", mode="w+", dtype="<f4", shape=(args.points, 3)
    )
    rgb_out = np.lib.format.open_memmap(
        args.output_dir / "rgb.npy", mode="w+", dtype="u1", shape=(args.points, 3)
    )
    labels_out = np.lib.format.open_memmap(
        args.output_dir / "labels.npy", mode="w+", dtype="<u2", shape=(args.points,)
    )
    votes_out = np.lib.format.open_memmap(
        args.output_dir / "votes.npy", mode="w+", dtype="<u2", shape=(args.points,)
    )
    indices_out = np.lib.format.open_memmap(
        args.output_dir / "source_indices.npy", mode="w+", dtype="<i8", shape=(args.points,)
    )

    for start in range(0, args.points, args.chunk_size):
        end = min(start + args.chunk_size, args.points)
        idx = selected[start:end]
        xyz_out[start:end] = np.column_stack((source["x"][idx], source["y"][idx], source["z"][idx]))
        normals_out[start:end] = np.column_stack((source["nx"][idx], source["ny"][idx], source["nz"][idx]))
        rgb_out[start:end] = np.column_stack((source["red"][idx], source["green"][idx], source["blue"][idx]))
        labels_out[start:end] = labelled["label"][idx]
        votes_out[start:end] = labelled["votes"][idx]
        indices_out[start:end] = idx
        print(f"prepared {end:,}/{args.points:,} dense init points", flush=True)

    label_counts = np.bincount(np.asarray(labels_out), minlength=4)
    summary = {
        "source_ply": str(args.source_ply.resolve()),
        "labelled_ply": str(args.labelled_ply.resolve()),
        "source_points": source_count,
        "selected_points": args.points,
        "selection": "all semantic points plus evenly sampled background",
        "label_counts": {str(i): int(value) for i, value in enumerate(label_counts)},
        "rgb": "original COLMAP fused RGB",
        "coordinates": "original COLMAP world coordinates; trainer applies parser.transform",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
