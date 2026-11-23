"""Repair one round of novel views with Difix (reference-guided): DIR/render + DIR/ref -> DIR/difix_ref.

Usage: python -I difix_fix.py DIFIX_SRC DIR
"""
import glob
import os
import sys
import time

import torch
from PIL import Image

src, d = sys.argv[1], sys.argv[2]
sys.path.insert(0, src)
from pipeline_difix import DifixPipeline  # noqa: E402

pipe = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
pipe.set_progress_bar_config(disable=True)
pipe.to("cuda")
os.makedirs(f"{d}/difix_ref", exist_ok=True)
paths = sorted(glob.glob(f"{d}/render/*.png"))
t0 = time.time()
for p in paths:
    n = os.path.basename(p)
    image = Image.open(p).convert("RGB")
    ref = Image.open(f"{d}/ref/{n}").convert("RGB")
    with torch.no_grad():
        out = pipe("remove degradation", image=image, ref_image=ref, num_inference_steps=1, timesteps=[199],
                   guidance_scale=0.0).images[0]
    out.resize(image.size, Image.LANCZOS).save(f"{d}/difix_ref/{n}")
print(f"{len(paths)} views repaired in {time.time() - t0:.0f} s")
