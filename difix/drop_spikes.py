"""Drop the edge spikes and strays of a 4K-tile build from its web ply (export_web_splat.py's frame, the phone
PHONE_HEIGHT units up): Gaussians whose longest axis passes SPIKE phone heights while opacity > 0.3, and
Gaussians farther than NEAR phone heights from every Gaussian of a reference web ply (the build it grew from).

    python -I drop_spikes.py IN.ply REF.ply OUT.ply [SPIKE=0.03] [NEAR=0.01]

The 4K tiles leave the crop's outer band unscored (difix_road_4k.ERODE), so a few hundred Gaussians there
grew into large opaque spikes and a few thousand drifted off the road into empty space; seen from low angles
they read as white and green streaks round the patch. Neither clean_floaters.py nor the web crop catches them.
"""
import sys

import numpy as np
from scipy.spatial import cKDTree

PHONE_HEIGHT = 1.5


def read(path):
    with open(path, "rb") as f:
        header, n, props = [], 0, []
        while True:
            line = f.readline()
            header.append(line)
            text = line.decode().strip()
            if text.startswith("element vertex"):
                n = int(text.split()[-1])
            elif text.startswith("property"):
                assert text.split()[1] == "float", text
                props.append(text.split()[-1])
            elif text == "end_header":
                break
        return header, props, np.fromfile(f, dtype=np.float32, count=n * len(props)).reshape(n, len(props))


src, ref, out = sys.argv[1:4]
spike = float(sys.argv[4]) if len(sys.argv) > 4 else 0.03
near = float(sys.argv[5]) if len(sys.argv) > 5 else 0.01
header, props, v = read(src)
col = {p: i for i, p in enumerate(props)}
xyz = v[:, [col["x"], col["y"], col["z"]]]
longest = np.exp(v[:, [col["scale_0"], col["scale_1"], col["scale_2"]]]).max(1) / PHONE_HEIGHT
opaque = 1 / (1 + np.exp(-v[:, col["opacity"]])) > 0.3
big = (longest > spike) & opaque
_, _, r = read(ref)
far = cKDTree(r[:, :3]).query(xyz)[0] / PHONE_HEIGHT > near
keep = ~(big | far)
print(f"{int(big.sum()):,} large opaque, {int(far.sum()):,} far from {ref}; kept {int(keep.sum()):,} of {len(v):,}")
with open(out, "wb") as f:
    for line in header:
        f.write(f"element vertex {int(keep.sum())}\n".encode() if line.startswith(b"element vertex") else line)
    v[keep].tofile(f)
