"""Size of the Gaussians the web keeps, the web's top_r3 against the 4K pilot: are the 4K build's edge spikes its
largest Gaussians? Usage: CLIP CKPT_A CLEAN_A CKPT_B CLEAN_B"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from difix_road import PHONE_HEIGHT, in_crop, load, make_runner, web_frame  # noqa: E402


def ply_xyz(path):  # web_orbit.py's (that one runs on import)
    with open(path, "rb") as f:
        n, props = 0, []
        while (line := f.readline().decode().strip()) != "end_header":
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                props.append(line.split()[-1])
        data = np.fromfile(f, dtype=np.float32, count=n * len(props)).reshape(n, len(props))
    return data[:, [props.index("x"), props.index("y"), props.index("z")]]


clip = sys.argv[1]
runner = make_runner(clip, 100000)
_, _, _, scale, _, _, _ = web_frame(runner.parser)  # world -> web units, where the phone is PHONE_HEIGHT up
for ckpt, clean in (sys.argv[2:4], sys.argv[4:6]):
    load(runner, ckpt)
    means = runner.splats["means"].detach().cpu().numpy()
    finite = np.isfinite(means).all(1)
    d = np.full(len(means), np.inf)
    d[finite] = cKDTree(ply_xyz(clean)).query(means[finite], distance_upper_bound=1e-5)[0]
    keep = (torch.as_tensor(np.isfinite(d), device="cuda") & in_crop(runner)).cpu().numpy()
    s = torch.exp(runner.splats["scales"]).detach().cpu().numpy()[keep] * scale / PHONE_HEIGHT
    big = s.max(1)
    op = torch.sigmoid(runner.splats["opacities"]).detach().cpu().numpy()[keep]
    print(f"{os.path.basename(ckpt)}: {keep.sum():,} kept; largest axis in phone heights, percentiles "
          + ", ".join(f"{q}%={np.percentile(big, q):.4f}" for q in (50, 99, 99.9, 99.99, 100)))
    for t in (0.02, 0.05, 0.1, 0.2):
        print(f"  > {t}: {(big > t).sum():,} (mean opacity {op[big > t].mean() if (big > t).any() else 0:.2f})")
