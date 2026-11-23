"""Patch the gsplat clone for dense depth-prior training, reversibly.

``third_party/gsplat`` is an untracked clone, so changes to it live here as a
patch rather than as edits nobody can review. Two files are touched.

1. ``examples/datasets/colmap.py`` -- serve a dense AI depth prior

   gsplat's ``--depth-loss`` already supervises rendered depth at a supplied set
   of pixels, and is normally handed the few hundred COLMAP sparse points
   visible in the frame. The patch appends tens of thousands of samples from the
   aligned AI depth maps to that same set, so the loss itself is untouched.
   Configuration is by environment variable, which keeps the change to one file
   and lets both ``simple_trainer.py`` and ``simple_trainer_2dgs.py`` pick the
   prior up unchanged, since both build the same ``Dataset``.

2. ``gsplat/cuda/_wrapper.py`` -- make CUDA op arguments contiguous

   The 2DGS autograd hooks pass incoming gradients straight to their CUDA op,
   which requires contiguous memory. With ``--depth-loss`` on, the trainer
   slices the depth channel out of the render, so its gradient arrives strided
   and training dies in the backward pass with::

       RuntimeError: grad.depths must be contiguous

   Three 2DGS classes have the same hole, so the fix goes in the single place
   every op call passes through instead of in each of them. Normalising is safe:
   these ops require contiguous input regardless, so this cannot change a
   result, and it copies only where the alternative was a crash.

Usage::

    python gaussian_splatting/patch_gsplat_depth_prior.py --apply
    python gaussian_splatting/patch_gsplat_depth_prior.py --status
    python gaussian_splatting/patch_gsplat_depth_prior.py --revert
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

BEGIN = "# >>> road-project depth prior"
END = "# <<< road-project depth prior"

DATASET_ANCHOR_INIT = """        self.load_depths = load_depths
"""

DATASET_ANCHOR_GETITEM = """            data["points"] = torch.from_numpy(points).float()
            data["depths"] = torch.from_numpy(depths).float()
"""

DATASET_INIT_PATCH = """        self.load_depths = load_depths
{begin}
        self._depth_prior = None
        self._depth_prior_keep_sparse = True
        if load_depths:
            import sys as _road_sys

            if r"{depth_dir}" not in _road_sys.path:
                _road_sys.path.insert(0, r"{depth_dir}")
            from depth_prior import DepthPrior as _RoadDepthPrior
            from depth_prior import similarity_scale as _road_similarity_scale

            self._depth_prior = _RoadDepthPrior.from_env(
                _road_similarity_scale(parser.transform)
            )
            self._depth_prior_keep_sparse = _RoadDepthPrior.keep_sparse()
{end}
"""

DATASET_GETITEM_PATCH = """{begin}
            _road_prior = getattr(self, "_depth_prior", None)
            if _road_prior is not None:
                if not getattr(self, "_depth_prior_keep_sparse", True):
                    points = points[:0]
                    depths = depths[:0]
                _road_points, _road_depths = _road_prior.sample(
                    image_name, image.shape[1], image.shape[0]
                )
                if len(_road_points):
                    points = np.concatenate([points, _road_points], axis=0)
                    depths = np.concatenate([depths, _road_depths], axis=0)
{end}
            data["points"] = torch.from_numpy(points).float()
            data["depths"] = torch.from_numpy(depths).float()
"""

WRAPPER_ANCHOR = """        if _C is not None:
            _ensure_autograd_registrations()
        return getattr(torch.ops.gsplat, name)(*args, **kwargs)
"""

WRAPPER_PATCH = """        if _C is not None:
            _ensure_autograd_registrations()
{begin}
        # The 2DGS autograd hooks hand incoming gradients straight to their CUDA
        # op, and a sliced depth channel arrives strided, which those kernels
        # reject ("grad.depths must be contiguous"). Every op here requires
        # contiguous input anyway, so normalising cannot change a result.
        args = tuple(
            arg.contiguous()
            if isinstance(arg, torch.Tensor) and not arg.is_contiguous()
            else arg
            for arg in args
        )
        kwargs = {{
            key: (
                value.contiguous()
                if isinstance(value, torch.Tensor) and not value.is_contiguous()
                else value
            )
            for key, value in kwargs.items()
        }}
{end}
        return getattr(torch.ops.gsplat, name)(*args, **kwargs)
"""


@dataclass
class PatchSite:
    relative_path: str
    label: str

    def resolve(self, gsplat_root: Path) -> Path:
        return (gsplat_root / self.relative_path).resolve()


SITES = (
    PatchSite("examples/datasets/colmap.py", "dense depth prior"),
    PatchSite("gsplat/cuda/_wrapper.py", "contiguous CUDA arguments"),
)


def is_applied(text: str) -> bool:
    return BEGIN in text


def require_single(text: str, anchor: str, path: Path, label: str) -> None:
    found = text.count(anchor)
    if found != 1:
        raise RuntimeError(
            f"Expected exactly one {label} anchor in {path.name}, found {found}. "
            "Upstream has changed; update this patch."
        )


def backup_once(path: Path) -> Path:
    backup = path.with_suffix(path.suffix + ".road-backup")
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def patch_dataset(path: Path, depth_dir: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if is_applied(text):
        return "already applied"
    require_single(text, DATASET_ANCHOR_INIT, path, "Dataset.__init__")
    require_single(text, DATASET_ANCHOR_GETITEM, path, "__getitem__")
    backup_once(path)
    text = text.replace(
        DATASET_ANCHOR_INIT,
        DATASET_INIT_PATCH.format(begin=BEGIN, end=END, depth_dir=str(depth_dir)),
        1,
    )
    text = text.replace(
        DATASET_ANCHOR_GETITEM,
        DATASET_GETITEM_PATCH.format(begin=BEGIN, end=END),
        1,
    )
    path.write_text(text, encoding="utf-8")
    return "applied"


def patch_wrapper(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if is_applied(text):
        return "already applied"
    require_single(text, WRAPPER_ANCHOR, path, "_make_lazy_cuda_func")
    backup_once(path)
    text = text.replace(
        WRAPPER_ANCHOR, WRAPPER_PATCH.format(begin=BEGIN, end=END), 1
    )
    path.write_text(text, encoding="utf-8")
    return "applied"


def revert(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if not is_applied(text):
        return "not applied"
    backup = path.with_suffix(path.suffix + ".road-backup")
    if backup.exists():
        shutil.copy2(backup, path)
        return f"reverted from {backup.name}"
    kept: list[str] = []
    inside = False
    for line in text.splitlines(keepends=True):
        if BEGIN in line:
            inside = True
            continue
        if END in line:
            inside = False
            continue
        if not inside:
            kept.append(line)
    path.write_text("".join(kept), encoding="utf-8")
    return "reverted by removing the marked blocks"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--apply", action="store_true")
    group.add_argument("--revert", action="store_true")
    group.add_argument("--status", action="store_true")
    parser.add_argument("--gsplat-dir", type=Path, default=None)
    parser.add_argument("--depth-dir", type=Path, default=None,
                        help="Directory holding depth_prior.py.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    gsplat_root = (args.gsplat_dir or (root / "third_party" / "gsplat")).resolve()
    depth_dir = (args.depth_dir or (root / "depth")).resolve()
    if not gsplat_root.is_dir():
        raise SystemExit(f"gsplat clone not found: {gsplat_root}")
    if not (depth_dir / "depth_prior.py").exists():
        raise SystemExit(f"depth_prior.py not found in {depth_dir}")

    paths = {site.label: site.resolve(gsplat_root) for site in SITES}
    for label, path in paths.items():
        if not path.exists():
            raise SystemExit(f"{label}: file not found: {path}")

    if args.status:
        states = []
        for site in SITES:
            path = paths[site.label]
            applied = is_applied(path.read_text(encoding="utf-8"))
            states.append(applied)
            print(f"  {'APPLIED    ' if applied else 'NOT APPLIED'} {site.label}: {path}")
        print("APPLIED" if all(states) else "NOT APPLIED")
        return

    if args.apply:
        print(f"dense depth prior          : {patch_dataset(paths['dense depth prior'], depth_dir)}")
        print(f"contiguous CUDA arguments  : {patch_wrapper(paths['contiguous CUDA arguments'])}")
    else:
        for site in SITES:
            print(f"{site.label:27}: {revert(paths[site.label])}")


if __name__ == "__main__":
    main()
