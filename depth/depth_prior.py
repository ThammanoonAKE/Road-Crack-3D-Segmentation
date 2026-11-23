"""Serve aligned AI depth to a Gaussian Splatting trainer as a depth prior.

gsplat's depth loss already accepts a set of pixel coordinates and their depths
per image, and supervises the rendered depth at exactly those pixels. It is
normally fed the few hundred COLMAP sparse points that fall in the frame. This
module feeds it tens of thousands of samples drawn from the aligned AI depth
map instead, which is dense supervision through an interface that already works,
with no change to the loss.

Two conversions have to be right or the prior fights the reconstruction:

scene normalisation
    The trainer normalises the world, so its camera-space depths are scaled by
    the similarity transform the COLMAP parser applied. Aligned maps are in raw
    COLMAP units and are multiplied by that scale here.

image resolution
    The network predicts at its own resolution and the trainer may downscale the
    images. Sample coordinates are mapped into whatever resolution the trainer
    actually loaded.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ENV_DIR = "GSPLAT_DEPTH_PRIOR_DIR"
ENV_SAMPLES = "GSPLAT_DEPTH_PRIOR_SAMPLES"
ENV_MIN_CONF = "GSPLAT_DEPTH_PRIOR_MIN_CONF"
ENV_SEED = "GSPLAT_DEPTH_PRIOR_SEED"
ENV_KEEP_SPARSE = "GSPLAT_DEPTH_PRIOR_KEEP_SPARSE"


class DepthPrior:
    """Random dense samples from the accepted aligned depth maps."""

    def __init__(
        self,
        root: Path,
        samples: int = 40000,
        min_conf: float = 0.0,
        scene_scale: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.samples = int(samples)
        self.min_conf = float(min_conf)
        self.scene_scale = float(scene_scale)
        self._rng = np.random.default_rng(seed)

        report_path = self.root / "alignment_report.json"
        if not report_path.exists():
            raise FileNotFoundError(
                f"No alignment_report.json in {self.root}. Run "
                "depth/align_depth_to_sparse.py with --write-maps first."
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.files: dict[str, str] = report.get("aligned_files", {})
        if not self.files:
            raise ValueError(
                f"{report_path} lists no aligned maps. Re-run the alignment "
                "stage with --write-maps."
            )
        self.summary = report.get("summary", {})
        self._cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}

    @classmethod
    def from_env(cls, scene_scale: float = 1.0) -> "DepthPrior | None":
        """Build from environment variables, or return None when unset.

        Using the environment keeps the change to gsplat down to one file and
        makes the prior work with both the 3DGS and 2DGS trainers unchanged.
        """
        root = os.environ.get(ENV_DIR, "").strip()
        if not root:
            return None
        seed = os.environ.get(ENV_SEED, "").strip()
        prior = cls(
            root=Path(root),
            samples=int(os.environ.get(ENV_SAMPLES, "40000")),
            min_conf=float(os.environ.get(ENV_MIN_CONF, "0.0")),
            scene_scale=scene_scale,
            seed=int(seed) if seed else None,
        )
        print(
            f"[depth prior] {len(prior.files)} aligned maps from {prior.root}\n"
            f"[depth prior] {prior.samples} samples per image, "
            f"scene scale {prior.scene_scale:.6f}, min conf {prior.min_conf}",
            flush=True,
        )
        return prior

    @staticmethod
    def keep_sparse() -> bool:
        """Whether COLMAP's own sparse depths stay in the supervision set."""
        return os.environ.get(ENV_KEEP_SPARSE, "1").strip() not in ("0", "false", "False")

    def _load(self, image_name: str):
        cached = self._cache.get(image_name)
        if cached is not None:
            return cached
        relative = self.files.get(image_name)
        if relative is None:
            return None
        path = self.root / "aligned" / Path(relative).name
        if not path.exists():
            path = self.root / relative
        if not path.exists():
            return None
        with np.load(path) as handle:
            depth = handle["depth"].astype(np.float32)
            valid = handle["valid"].astype(bool)
            conf = handle["conf"].astype(np.float32) if "conf" in handle else None
        flat_valid = np.flatnonzero(valid.reshape(-1))
        entry = (depth, flat_valid, conf)
        # One scene's maps are far too large to hold at once; keep a small window.
        if len(self._cache) > 24:
            self._cache.pop(next(iter(self._cache)))
        self._cache[image_name] = entry
        return entry

    def sample(
        self, image_name: str, width: int, height: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (pixels (M, 2), depths (M,)) for one image, in trainer units."""
        empty = (np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32))
        loaded = self._load(image_name)
        if loaded is None:
            return empty
        depth, flat_valid, conf = loaded
        if not len(flat_valid):
            return empty

        candidates = flat_valid
        if conf is not None and self.min_conf > 0.0:
            keep = conf.reshape(-1)[candidates] >= self.min_conf
            if keep.any():
                candidates = candidates[keep]

        count = min(self.samples, len(candidates))
        chosen = (
            candidates
            if count == len(candidates)
            else self._rng.choice(candidates, size=count, replace=False)
        )

        map_height, map_width = depth.shape
        rows, cols = np.divmod(chosen, map_width)
        # Sample at pixel centres, then rescale into the trainer's resolution.
        x = (cols + 0.5) * (width / map_width)
        y = (rows + 0.5) * (height / map_height)
        pixels = np.stack((x, y), axis=1).astype(np.float32)
        depths = depth.reshape(-1)[chosen].astype(np.float32) * self.scene_scale

        finite = np.isfinite(depths) & (depths > 0)
        finite &= (
            (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
            & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        )
        return pixels[finite], depths[finite]


def similarity_scale(transform: np.ndarray) -> float:
    """Uniform scale of the COLMAP parser's normalisation transform."""
    matrix = np.asarray(transform, dtype=np.float64)[:3, :3]
    # The transform is a similarity, so every column shares one scale.
    return float(np.cbrt(abs(np.linalg.det(matrix))))
