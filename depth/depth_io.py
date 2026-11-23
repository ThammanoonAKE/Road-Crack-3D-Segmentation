"""Shared storage format for AI depth maps and their alignment metadata.

A depth stage writes one ``.npz`` per source image plus a single manifest, so a
run can be inspected, resumed, and verified without loading every map.

Raw AI depth is whatever the network emitted (usually affine-invariant inverse
depth). Aligned depth is metric depth in COLMAP world units, directly
comparable with ``points3D.bin``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

MANIFEST_NAME = "depth_manifest.json"
RAW_KIND_INVERSE = "affine_invariant_inverse_depth"
RAW_KIND_DEPTH = "affine_invariant_depth"
RAW_KIND_METRIC = "metric_depth"
RAW_KINDS = (RAW_KIND_INVERSE, RAW_KIND_DEPTH, RAW_KIND_METRIC)


@dataclass
class DepthRecord:
    """One image's depth map on disk."""

    image_name: str
    file: str
    width: int
    height: int
    raw_min: float
    raw_max: float
    has_conf: bool = False
    has_sky: bool = False
    # Index of the inference window this map came from, for reproducibility.
    window: int = -1

    def path(self, root: Path) -> Path:
        return Path(root) / self.file


@dataclass
class AlignmentRecord:
    """Result of fitting one image's AI depth to the COLMAP sparse track."""

    image_name: str
    scale: float
    shift: float
    sparse_points: int
    inliers: int
    # Absolute relative error between fitted and COLMAP depth, over inliers.
    median_abs_rel: float
    p95_abs_rel: float
    rmse: float
    accepted: bool
    reason: str = ""


@dataclass
class DepthManifest:
    model: str
    raw_kind: str
    images_dir: str
    image_count: int
    created: str
    extra: dict[str, Any] = field(default_factory=dict)
    records: list[DepthRecord] = field(default_factory=list)

    def to_json(self) -> str:
        payload = {
            "model": self.model,
            "raw_kind": self.raw_kind,
            "images_dir": self.images_dir,
            "image_count": self.image_count,
            "created": self.created,
            "extra": self.extra,
            "records": [asdict(record) for record in self.records],
        }
        return json.dumps(payload, indent=2)

    @classmethod
    def load(cls, root: Path) -> "DepthManifest":
        path = Path(root) / MANIFEST_NAME
        if not path.exists():
            raise FileNotFoundError(f"No depth manifest at {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifest = cls(
            model=payload["model"],
            raw_kind=payload["raw_kind"],
            images_dir=payload["images_dir"],
            image_count=payload["image_count"],
            created=payload["created"],
            extra=payload.get("extra", {}),
        )
        manifest.records = [DepthRecord(**record) for record in payload["records"]]
        if manifest.raw_kind not in RAW_KINDS:
            raise ValueError(f"Unknown raw_kind in manifest: {manifest.raw_kind}")
        return manifest

    def save(self, root: Path) -> Path:
        path = Path(root) / MANIFEST_NAME
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    def by_name(self) -> dict[str, DepthRecord]:
        return {record.image_name: record for record in self.records}


def relative_file_name(image_name: str) -> str:
    """Flatten an image name into a single safe file name."""
    stem = image_name.replace("\\", "/").replace("/", "__")
    return f"{stem}.npz"


def write_depth(
    root: Path,
    image_name: str,
    depth: np.ndarray,
    conf: np.ndarray | None = None,
    sky: np.ndarray | None = None,
    window: int = -1,
    subdir: str = "",
) -> DepthRecord:
    """Store one raw depth map as float16, which is ample for a network output.

    ``conf`` is the network's own per-pixel confidence and ``sky`` its sky mask.
    Both are optional but worth keeping: they let a later stage down-weight or
    drop the pixels the network itself is unsure about.

    ``root`` is the manifest directory. ``subdir`` places the file beneath it and
    is recorded in the returned path, so a reader only ever needs the manifest
    directory to find every map.
    """
    root = Path(root)
    (root / subdir).mkdir(parents=True, exist_ok=True)
    if depth.ndim != 2:
        raise ValueError(f"{image_name}: expected a 2D depth map, got {depth.shape}")
    finite = np.isfinite(depth)
    if not finite.any():
        raise ValueError(f"{image_name}: depth map is entirely non-finite")
    payload: dict[str, np.ndarray] = {"depth": depth.astype(np.float16)}
    if conf is not None:
        if conf.shape != depth.shape:
            raise ValueError(f"{image_name}: conf shape {conf.shape} != {depth.shape}")
        payload["conf"] = conf.astype(np.float16)
    if sky is not None:
        if sky.shape != depth.shape:
            raise ValueError(f"{image_name}: sky shape {sky.shape} != {depth.shape}")
        payload["sky"] = sky.astype(np.bool_)
    name = f"{subdir}/{relative_file_name(image_name)}" if subdir else relative_file_name(image_name)
    np.savez_compressed(root / name, **payload)
    return DepthRecord(
        image_name=image_name,
        file=name,
        width=int(depth.shape[1]),
        height=int(depth.shape[0]),
        raw_min=float(depth[finite].min()),
        raw_max=float(depth[finite].max()),
        has_conf=conf is not None,
        has_sky=sky is not None,
        window=int(window),
    )


def read_depth(root: Path, record: DepthRecord) -> np.ndarray:
    with np.load(record.path(root)) as handle:
        return handle["depth"].astype(np.float32)


def read_depth_bundle(
    root: Path, record: DepthRecord
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Return depth plus confidence and sky mask when the record has them."""
    with np.load(record.path(root)) as handle:
        depth = handle["depth"].astype(np.float32)
        conf = handle["conf"].astype(np.float32) if "conf" in handle else None
        sky = handle["sky"].astype(bool) if "sky" in handle else None
    return depth, conf, sky


def iter_depths(root: Path, manifest: DepthManifest) -> Iterator[tuple[DepthRecord, np.ndarray]]:
    for record in manifest.records:
        yield record, read_depth(root, record)


def write_aligned(
    root: Path,
    image_name: str,
    depth: np.ndarray,
    valid: np.ndarray,
    conf: np.ndarray | None = None,
) -> str:
    """Store one metric depth map, its validity mask, and any confidence."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    name = relative_file_name(image_name)
    payload: dict[str, np.ndarray] = {
        "depth": depth.astype(np.float32),
        "valid": valid.astype(np.bool_),
    }
    if conf is not None:
        payload["conf"] = conf.astype(np.float16)
    np.savez_compressed(root / name, **payload)
    return name


def read_aligned(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as handle:
        return handle["depth"].astype(np.float32), handle["valid"].astype(bool)
