"""Blur every vehicle in a video, so that number plates cannot be read (used on the sample videos).

    python -I blur_vehicles.py IN.mp4 OUT.mp4 [--ss SEC] [--t SEC] [--every 2] [--sigma 7] [--grow 9] [--crf 28]
                                             [--keep-top 56]
    python -I blur_vehicles.py IN.mp4 --scan [--every 15]

Vehicles (car, truck, bus, motorcycle) are found with Mask2Former (Swin-T, COCO instance segmentation,
facebook/mask2former-swin-tiny-coco-instance) on every --every-th frame. A frame is blurred where any vehicle mask
of the detected frames on either side of it lies (so a vehicle missed in one detection, or moved since, is still
covered), grown by --grow px and feathered. The whole vehicle is blurred, not only its plate: no plate detector is
used, and a light blur over the whole vehicle (Gaussian sigma --sigma px) makes plate characters unreadable while
the vehicle stays recognisable. Runs on the CPU unless --device cuda.

--scan only reports, for every --every-th frame, how much of the frame is vehicle.
"""
import argparse
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the model is used from the local cache only

import cv2  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor  # noqa: E402

MODEL = "facebook/mask2former-swin-tiny-coco-instance"
VEHICLES = {"car", "truck", "bus", "motorcycle"}

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("out", nargs="?")
ap.add_argument("--ss", type=float, default=0.0, help="start, seconds")
ap.add_argument("--t", type=float, default=0.0, help="duration, seconds (0 = to the end)")
ap.add_argument("--every", type=int, default=2)
ap.add_argument("--sigma", type=float, default=7.0)
ap.add_argument("--grow", type=int, default=9)
ap.add_argument("--score", type=float, default=0.3, help="instance score threshold")
ap.add_argument("--crf", type=int, default=28)
ap.add_argument("--device", default="cpu")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--scan", action="store_true")
ap.add_argument("--keep-top", type=int, default=0,
                help="leave the top N rows unblurred (the text HUD of the edge-AI videos, which shows no imagery)")
args = ap.parse_args()
torch.set_num_threads(args.threads)

# Built here rather than loaded: the cached snapshot has no preprocessor_config.json. ImageNet mean/std (the
# processor's defaults); short side 544, the sample videos' own size, instead of upsampling to 800.
proc = Mask2FormerImageProcessor(size={"shortest_edge": 544, "longest_edge": 1333})
model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL).to(args.device).eval()
vehicle_ids = {i for i, name in model.config.id2label.items() if name in VEHICLES}


def vehicles(bgr):
    """Union of the vehicle masks in a BGR frame, bool HxW."""
    h, w = bgr.shape[:2]
    inputs = proc(images=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), return_tensors="pt").to(args.device)
    with torch.no_grad():
        out = model(**inputs)
    res = proc.post_process_instance_segmentation(out, threshold=args.score, target_sizes=[(h, w)])[0]
    seg = res["segmentation"].cpu().numpy()
    mask = np.zeros((h, w), bool)
    for s in res["segments_info"]:
        if s["label_id"] in vehicle_ids:
            mask |= seg == s["id"]
    return mask


def frames():
    cap = cv2.VideoCapture(args.src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    first = int(round(args.ss * fps))
    last = int(round((args.ss + args.t) * fps)) if args.t else 1 << 30
    k = 0
    while k < last:
        ok, f = cap.read()
        if not ok:
            break
        if k >= first:
            yield k - first, f, fps
        k += 1


if args.scan:
    hits = []
    for i, f, fps in frames():
        if i % args.every == 0:
            share = vehicles(f).mean()
            if share > 0:
                hits.append((i / fps, share))
    name = os.path.basename(args.src)
    print(f"{name}: vehicles in {len(hits)} sampled frames" +
          (f", up to {100 * max(s for _, s in hits):.1f}% of the frame, at {', '.join(f'{t:.0f}s' for t, _ in hits[:12])}"
           if hits else ""), flush=True)
    raise SystemExit

# Pass 1: masks on every --every-th frame.
masks = {}
for i, f, fps in frames():
    if i % args.every == 0:
        masks[i] = np.packbits(vehicles(f))
shape = f.shape[:2]
n = i + 1
print(f"detected on {len(masks)} of {n} frames; vehicles in {sum(1 for m in masks.values() if m.any())}", flush=True)

# Pass 2: blur where the neighbouring detections saw a vehicle.
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * args.grow + 1, 2 * args.grow + 1))
writer = imageio.get_writer(args.out, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=8,
                            ffmpeg_params=["-crf", str(args.crf), "-preset", "slow", "-movflags", "+faststart"])
blurred = 0
for i, f, _ in frames():
    lo = (i // args.every) * args.every
    near = [masks[j] for j in (lo - args.every, lo, lo + args.every) if j in masks]
    m = np.unpackbits(np.bitwise_or.reduce(near), count=shape[0] * shape[1]).reshape(shape).astype(bool)
    if m.any():
        alpha = cv2.GaussianBlur(cv2.dilate(m.astype(np.uint8) * 255, kernel), (0, 0), args.grow / 2)[..., None] / 255.0
        if args.keep_top:
            alpha[: args.keep_top] = 0
        soft = cv2.GaussianBlur(f, (0, 0), args.sigma)
        f = (alpha * soft + (1 - alpha) * f).astype(np.uint8)
        blurred += 1
    writer.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
writer.close()
print(f"{os.path.basename(args.src)} -> {args.out}: {n} frames, {blurred} with vehicles blurred", flush=True)
