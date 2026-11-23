"""Mark a road hole's cracks on its web splat, for the page's crack highlight (index.html: button or key C).

DeepCrack (models/deepcrack/crack_detector) gives every photo of the clip a crack probability per pixel: the
mean of its maps at full and half size. Full size alone also finds hairline cracks but turns dark, rough patches
into blobs; half size keeps clean lines but misses hairlines.

Each Gaussian of the web PLY is scored by where its weight in the renders lands: the share of crack probability
under it, weighted and summed over every photo. The weights are the gradient of a render with respect to a
per-Gaussian value, so a Gaussian hidden behind others counts for as little as it draws. Gaussians scoring over
--threshold are cracks. The renders use the web PLY itself, in its own frame (export_web_splat.web_frame), with
the photos' cameras moved into it.

make_ksplat.mjs regroups the splats, so the labels are carried over by position: ksplat_centers.mjs lists the
.ksplat's centres in the order the viewer indexes them. The output holds the crack splats' indices in that order,
uint32 little-endian. A sheet compares the detector on photos with the marked splat seen from the same poses.

    python -I web/tools/crack_labels.py --scene C:/3D-segmentation-data/road_crack/151720/scene \
        --ply C:/3D-segmentation-data/difix/road_151720/hole_2_4k_cut.ply --ksplat web/public/splats/hole_2.ksplat
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]  # the repository (this file is web/tools/)
sys.path.insert(0, str(ROOT / "gaussian_splatting"))
sys.path.insert(0, str(ROOT / "third_party" / "gsplat" / "examples"))
from export_web_splat import web_frame  # noqa: E402

SH_C0 = 0.28209479177387814
HIGHLIGHT = np.array([255, 40, 40]) / 255  # index.html's crack colour and strength
STRENGTH = 0.8


def read_ply(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = []
        while (line := handle.readline().decode("ascii").rstrip("\n")) != "end_header":
            header.append(line)
        names = [h.split()[2] for h in header if h.startswith("property")]
        return np.fromfile(handle, dtype=np.dtype([(n, "<f4") for n in names]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", type=Path, required=True, help="The scene the splat trained on (1080p photos).")
    parser.add_argument("--ply", type=Path, required=True, help="The web PLY the .ksplat was made from.")
    parser.add_argument("--ksplat", type=Path, required=True, help="The web .ksplat; its .json must sit beside it.")
    parser.add_argument("--output", type=Path, help="Default: the .ksplat's path with .cracks.bin.")
    parser.add_argument("--sheet", type=Path, help="Default: data/road_crack/previews/<name>_cracks.jpg.")
    parser.add_argument("--detector", type=Path, default=ROOT / "models" / "deepcrack" / "crack_detector")
    parser.add_argument("--threshold", type=float, default=0.35, help="Crack share above which a Gaussian is crack.")
    parser.add_argument("--min-weight", type=float, default=1.0,
                        help="Ignore Gaussians whose summed weight over all photos is below this (barely seen).")
    args = parser.parse_args()
    output = args.output or args.ksplat.with_suffix(".cracks.bin")
    sheet = args.sheet or ROOT / "data" / "road_crack" / "previews" / f"{args.ksplat.stem}_cracks.jpg"
    sys.path.insert(0, str(args.detector))
    from crack_detector import CrackDetector  # noqa: E402
    from datasets.colmap import Parser  # gsplat example code, not a package
    from gsplat.rendering import rasterization

    dev = torch.device("cuda")
    scene = Parser(data_dir=str(args.scene), factor=1, normalize=True, test_every=8)
    middle, rot, scale, _, opening = web_frame(scene)
    to_web = lambda p: (np.asarray(p) - middle) @ rot.T * scale  # noqa: E731
    camera = json.loads(args.ksplat.with_suffix(".json").read_text(encoding="utf-8"))
    assert np.allclose(to_web(opening), camera["position"], atol=2e-4), "the scene's web frame is not the PLY's"

    ply = read_ply(args.ply)
    xyz = np.stack([ply[k] for k in "xyz"], 1)
    means = torch.as_tensor(xyz, device=dev)
    quats = torch.as_tensor(np.stack([ply[f"rot_{i}"] for i in range(4)], 1), device=dev)
    quats = quats / quats.norm(dim=1, keepdim=True)
    scales = torch.exp(torch.as_tensor(np.stack([ply[f"scale_{i}"] for i in range(3)], 1), device=dev))
    opacities = torch.sigmoid(torch.as_tensor(ply["opacity"], device=dev))
    rgb = np.clip(0.5 + SH_C0 * np.stack([ply[f"f_dc_{i}"] for i in range(3)], 1), 0, 1)

    def view(i):
        c2w = scene.camtoworlds[i].astype(np.float64).copy()
        c2w[:3, :3] = rot @ c2w[:3, :3]
        c2w[:3, 3] = to_web(c2w[:3, 3])
        cam = scene.camera_ids[i]
        w, h = scene.imsize_dict[cam]
        return (torch.as_tensor(np.linalg.inv(c2w), dtype=torch.float32, device=dev)[None],
                torch.as_tensor(scene.Ks_dict[cam], dtype=torch.float32, device=dev)[None], w, h)

    det = CrackDetector(args.detector / "deepcrack_convnext_small_aug.pt")
    feat = torch.zeros(len(xyz), 2, device=dev, requires_grad=True)
    crack = torch.zeros(len(xyz), device=dev)
    total = torch.zeros(len(xyz), device=dev)
    n = len(scene.image_paths)
    picks = sorted({n // 6, n // 2, 5 * n // 6})
    probs = {}
    for i, path in enumerate(scene.image_paths):
        photo = cv2.imread(path)
        prob = 0.5 * (det.predict(photo) + det.predict(photo, scale=0.5))
        if i in picks:
            probs[i] = (photo, prob)
        viewmat, K, w, h = view(i)
        assert prob.shape == (h, w), (path, prob.shape, (h, w))
        img, _, _ = rasterization(means, quats, scales, opacities, feat, viewmat, K, w, h, packed=False)
        ((img[0, ..., 0] * torch.as_tensor(prob, device=dev)).sum() + img[0, ..., 1].sum()).backward()
        crack += feat.grad[:, 0]
        total += feat.grad[:, 1]
        feat.grad = None
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n} photos", flush=True)
    share = (crack / total.clamp(min=1e-6)).cpu().numpy()
    seen = (total >= args.min_weight).cpu().numpy()
    label = seen & (share > args.threshold)
    print(f"{len(xyz):,} Gaussians, {seen.mean():.1%} seen enough; crack share over "
          + ", ".join(f"{t}: {(seen & (share > t)).mean():.1%}" for t in (0.2, 0.35, 0.5, 0.65))
          + f"; marked {label.sum():,} ({label.mean():.1%}) at {args.threshold}")

    # The detector on three photos beside the marked splat from the same poses, half size.
    marked = torch.as_tensor(np.where(label[:, None], rgb * (1 - STRENGTH) + HIGHLIGHT * STRENGTH, rgb),
                             dtype=torch.float32, device=dev)
    rows = []
    for i in picks:
        photo, prob = probs[i]
        viewmat, K, w, h = view(i)
        with torch.no_grad():
            img, _, _ = rasterization(means, quats, scales, opacities, marked, viewmat, K, w, h, packed=False,
                                      backgrounds=torch.zeros(1, 3, device=dev))
        render = (img[0].clamp(0, 1).cpu().numpy()[..., ::-1] * 255).astype(np.uint8)
        left = CrackDetector.overlay(photo, (prob > det.threshold).astype(np.uint8) * 255, color=(40, 40, 255),
                                     alpha=STRENGTH)
        pair = [cv2.resize(t, (w // 2, h // 2), interpolation=cv2.INTER_AREA) for t in (left, render)]
        cv2.putText(pair[0], f"DeepCrack on photo {Path(scene.image_paths[i]).name[:28]}", (8, 26), 0, 0.7, (0, 255, 255), 2)
        cv2.putText(pair[1], f"web splat, cracks marked (share > {args.threshold})", (8, 26), 0, 0.7, (0, 255, 255), 2)
        rows.append(np.hstack(pair))
    sheet.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(sheet), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])

    # Carry the labels over to the .ksplat's order.
    with tempfile.TemporaryDirectory() as tmp:
        centres_path = Path(tmp) / "centres.f32"
        subprocess.run(["node", "--max-old-space-size=24000", "ksplat_centers.mjs", str(args.ksplat.resolve()),
                        str(centres_path)], cwd=Path(__file__).parent, check=True)
        centres = np.fromfile(centres_path, dtype="<f4").reshape(-1, 3)
    dist, nearest = cKDTree(xyz).query(centres)
    print(f".ksplat: {len(centres):,} splats; farthest from its PLY match {dist.max():.1e} units "
          f"(the phone was 1.5 up), 99.9% within {np.percentile(dist, 99.9):.1e}")
    indices = np.nonzero(label[nearest])[0].astype("<u4")
    indices.tofile(output)
    print(f"{len(indices):,} crack splats -> {output} ({output.stat().st_size / 1e6:.1f} MB); sheet -> {sheet}")


if __name__ == "__main__":
    main()
