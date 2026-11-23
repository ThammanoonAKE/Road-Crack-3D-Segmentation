"""Minimal COLMAP sparse-model reader and pinhole projection helpers.

This module is imported both by ordinary CPython scripts and by scripts running
inside Blender, so it depends on nothing beyond NumPy.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# COLMAP camera model id -> (name, parameter count).
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}

# Models this projector implements exactly. Fisheye models are rejected rather
# than silently approximated, because a wrong projection would smear texture.
SUPPORTED_MODELS = {
    "SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "FULL_OPENCV",
}


@dataclass(frozen=True)
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]

    def pinhole(self) -> tuple[float, float, float, float]:
        """Return fx, fy, cx, cy for the supported models."""
        p = self.params
        if self.model == "SIMPLE_PINHOLE":
            return p[0], p[0], p[1], p[2]
        if self.model == "PINHOLE":
            return p[0], p[1], p[2], p[3]
        if self.model in ("SIMPLE_RADIAL", "RADIAL"):
            return p[0], p[0], p[1], p[2]
        if self.model in ("OPENCV", "FULL_OPENCV"):
            return p[0], p[1], p[2], p[3]
        raise ValueError(f"Unsupported camera model for projection: {self.model}")

    def distortion(self) -> tuple[float, ...]:
        """Return (k1, k2, p1, p2) radial/tangential terms; zeros when undistorted."""
        p = self.params
        if self.model in ("SIMPLE_PINHOLE", "PINHOLE"):
            return (0.0, 0.0, 0.0, 0.0)
        if self.model == "SIMPLE_RADIAL":
            return (p[3], 0.0, 0.0, 0.0)
        if self.model == "RADIAL":
            return (p[3], p[4], 0.0, 0.0)
        if self.model in ("OPENCV", "FULL_OPENCV"):
            return (p[4], p[5], p[6], p[7])
        raise ValueError(f"Unsupported camera model for distortion: {self.model}")


@dataclass(frozen=True)
class Frame:
    """One registered image: its file name and world-to-camera pose."""

    id: int
    name: str
    camera_id: int
    rotation: np.ndarray  # 3x3 world-to-camera
    translation: np.ndarray  # 3 world-to-camera

    @property
    def center(self) -> np.ndarray:
        """Camera centre in world coordinates."""
        return -self.rotation.T @ self.translation

    @property
    def forward(self) -> np.ndarray:
        """Unit viewing direction in world coordinates."""
        return self.rotation[2]


def _unpack(handle, fmt: str):
    size = struct.calcsize("<" + fmt)
    raw = handle.read(size)
    if len(raw) != size:
        raise EOFError("Unexpected end of a COLMAP binary file")
    return struct.unpack("<" + fmt, raw)


def quaternion_to_rotation(qvec) -> np.ndarray:
    """COLMAP stores w, x, y, z."""
    w, x, y, z = (float(v) for v in qvec)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("Degenerate quaternion in COLMAP model")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def read_cameras(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    with Path(path).open("rb") as handle:
        for _ in range(_unpack(handle, "Q")[0]):
            camera_id, model_id, width, height = _unpack(handle, "iiQQ")
            if model_id not in CAMERA_MODELS:
                raise ValueError(f"Unknown COLMAP camera model id {model_id}")
            name, count = CAMERA_MODELS[model_id]
            params = _unpack(handle, "d" * count)
            cameras[camera_id] = Camera(camera_id, name, int(width), int(height), params)
    return cameras


def read_frames(path: Path) -> list[Frame]:
    """Read images.bin, skipping the per-image 2D observation block."""
    frames: list[Frame] = []
    with Path(path).open("rb") as handle:
        for _ in range(_unpack(handle, "Q")[0]):
            values = _unpack(handle, "idddddddi")
            image_id = values[0]
            qvec = values[1:5]
            tvec = values[5:8]
            camera_id = values[8]
            name = bytearray()
            while (char := handle.read(1)) != b"\0":
                if not char:
                    raise EOFError("Unexpected end while reading a COLMAP image name")
                name.extend(char)
            observation_count = _unpack(handle, "Q")[0]
            handle.seek(observation_count * 24, 1)
            frames.append(
                Frame(
                    id=int(image_id),
                    name=name.decode("utf-8"),
                    camera_id=int(camera_id),
                    rotation=quaternion_to_rotation(qvec),
                    translation=np.asarray(tvec, dtype=np.float64),
                )
            )
    return frames


def project(points: np.ndarray, frame: Frame, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    """Project world-space points into one image.

    Returns pixel coordinates (N, 2) as float64 and camera-space depth (N,).
    Depth is negative or zero for points behind the camera; the caller must
    filter on it before trusting the pixel coordinates.
    """
    points = np.asarray(points, dtype=np.float64)
    local = points @ frame.rotation.T + frame.translation
    depth = local[:, 2]
    safe = np.where(np.abs(depth) < 1e-12, 1e-12, depth)
    x = local[:, 0] / safe
    y = local[:, 1] / safe

    k1, k2, p1, p2 = camera.distortion()
    if k1 or k2 or p1 or p2:
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x, y = x_d, y_d

    fx, fy, cx, cy = camera.pinhole()
    return np.column_stack((fx * x + cx, fy * y + cy)), depth
