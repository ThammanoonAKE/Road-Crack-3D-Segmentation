"""Orbit sheet of what the web keeps: each checkpoint's Gaussians that survived clean_floaters (matched by centre
in its *_clean.ply) and the web crop, side by side. Usage: CLIP CKPT_A CLEAN_A LABEL_A CKPT_B CLEAN_B LABEL_B OUT"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from difix_road import PHONE_HEIGHT, crop_splats, dev, draw, in_crop, load, make_runner, orbit_views, photo_K, web_frame  # noqa: E402


def ply_xyz(path):
    with open(path, "rb") as f:
        n, props = 0, []
        while (line := f.readline().decode().strip()) != "end_header":
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                assert line.split()[1] == "float", line
                props.append(line.split()[-1])
        data = np.fromfile(f, dtype=np.float32, count=n * len(props)).reshape(n, len(props))
    return data[:, [props.index("x"), props.index("y"), props.index("z")]]


clip, out = sys.argv[1], sys.argv[8]
runner = make_runner(clip, 100000)
models = []
for ckpt, clean, label in (sys.argv[2:5], sys.argv[5:8]):
    load(runner, ckpt)
    means = runner.splats["means"].detach().cpu().numpy()
    finite = np.isfinite(means).all(1)
    d = np.full(len(means), np.inf)
    d[finite] = cKDTree(ply_xyz(clean)).query(means[finite], distance_upper_bound=1e-5)[0]
    keep = torch.as_tensor(np.isfinite(d), device=dev) & in_crop(runner)
    if os.environ.get("SPIKE_PH") and label == sys.argv[7]:  # drop B's large opaque Gaussians, as drop_spikes.py
        _, _, _, scale, _, _, _ = web_frame(runner.parser)
        big = torch.exp(runner.splats["scales"]).max(1).values * scale / PHONE_HEIGHT > float(os.environ["SPIKE_PH"])
        keep &= ~(big & (torch.sigmoid(runner.splats["opacities"]) > 0.3))
    if os.environ.get("NEAR_PH") and label == sys.argv[7]:  # and those far from every Gaussian A kept
        _, _, _, scale, _, _, _ = web_frame(runner.parser)
        dist = np.full(len(means), np.inf)
        dist[finite] = cKDTree(kept_a).query(means[finite])[0]
        far = torch.as_tensor(dist * scale / PHONE_HEIGHT > float(os.environ["NEAR_PH"]), device=dev)
        print(f"  {int((far & keep).sum()):,} kept ones lie over {os.environ['NEAR_PH']} phone heights from A's")
        keep &= ~far
    if label == sys.argv[4]:
        kept_a = means[keep.cpu().numpy()]
    print(f"{label}: {int(keep.sum()):,} of {len(means):,} Gaussians kept")
    models.append(({k: v.detach().clone() for k, v in crop_splats(runner, keep).items()}, label))
rows = []
for name, c2w, K in orbit_views(runner, photo_K(runner.trainset), els=(20, 40), back=1.6)[::3]:
    t = []
    for s, label in models:
        im = (draw(runner, s, c2w, K)[0].clamp(0, 1).cpu().numpy()[..., ::-1] * 255).astype(np.uint8).copy()
        cv2.putText(im, f"{name} {label}", (8, 28), 0, 0.8, (0, 0, 255), 2)
        t.append(im)
    rows.append(np.hstack(t))
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
