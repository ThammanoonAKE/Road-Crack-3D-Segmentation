"""Edge-style crack detection on a phone clip: decode, shrink, detect frame by frame, steady it, draw, and time it.

    python -I edge_crack_video.py CLIP.mp4 OUT.mp4 [--long 960] [--device cuda|cpu] [--every N]
                                  [--alpha 0.25] [--low 0.13] [--raw] [--compare]

Each frame is shrunk to --long on its long side, keeping its shape (portrait or landscape), turned upright when
the clip carries a rotation tag OpenCV does not apply (the road clips: 90 degrees), and run through
crack_detector_1class_best.

Frame by frame the mask flickered: pixels near the 0.16 threshold, and pieces near the 150 px minimum area,
came and went. So the probability map is averaged over time along the road:
- the road is a plane, so a homography (ORB features, RANSAC) carries the previous average onto this frame;
- the average takes --alpha of the new map (0.25: about four frames of memory, 0.13 s). On hole 2 the
  flicker went 26% -> 12% at 0.35 and 10% at 0.25 or 0.2; the rest is mostly 1 px misregistration of thin
  lines, which this measure counts too;
- road just come into view has no history and takes the new map as is;
- hysteresis: a pixel over --low is crack if its region also reaches the model's 0.16 somewhere and covers at
  least the model's 150 px.
--raw skips all that (the package's own predict_mask). --compare writes raw and steadied side by side.

The corner shows the model time, the rate the whole loop keeps, the crack share and the flicker: the share of
crack pixels that changed against the previous frame carried over. The video plays at the clip's 30 fps
whatever the loop managed; --every N keeps every Nth frame. A summary line goes to stdout.
"""
import argparse
import importlib.util
import os
import time

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("clip")
ap.add_argument("out")
ap.add_argument("--long", type=int, default=960)
ap.add_argument("--device", default="cuda")
ap.add_argument("--every", type=int, default=1)
ap.add_argument("--alpha", type=float, default=0.25)
ap.add_argument("--low", type=float, default=0.13)
ap.add_argument("--raw", action="store_true")
ap.add_argument("--compare", action="store_true")
ap.add_argument("--package", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "deepcrack", "crack_detector_1class_best"))
args = ap.parse_args()

spec = importlib.util.spec_from_file_location("crack_detector", rf"{args.package}\crack_detector.py")
cd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cd)
det = cd.CrackDetector(device=args.device)
cuda = det.device.type == "cuda"
gpu = torch.cuda.get_device_name(0).replace("NVIDIA ", "") if cuda else "CPU"
orb = cv2.ORB_create(1500)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def hysteresis(prob):
    weak = (prob > args.low).astype(np.uint8)
    if not weak.any():
        return weak
    n, labels, stats, _ = cv2.connectedComponentsWithStats(weak, connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(labels[prob > det.threshold])] = True
    keep &= stats[:, cv2.CC_STAT_AREA] >= det.min_area
    keep[0] = False
    return keep[labels].astype(np.uint8) * 255


FEAT_SCALE = 0.5  # features on a half-size frame; the homography is scaled back


def features(img):
    gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None, fx=FEAT_SCALE, fy=FEAT_SCALE,
                      interpolation=cv2.INTER_AREA)
    return orb.detectAndCompute(gray, None)


def homography(prev, cur):
    """prev -> cur for the road plane, at full frame size, or None."""
    if prev is None or prev[1] is None or cur[1] is None or len(prev[0]) < 30 or len(cur[0]) < 30:
        return None
    m = matcher.match(prev[1], cur[1])
    if len(m) < 30:
        return None
    a = np.float32([prev[0][x.queryIdx].pt for x in m])
    b = np.float32([cur[0][x.trainIdx].pt for x in m])
    H, inl = cv2.findHomography(a, b, cv2.RANSAC, 2.0)
    if H is None or inl.sum() < 25:
        return None
    S = np.diag([FEAT_SCALE, FEAT_SCALE, 1.0])
    return np.linalg.inv(S) @ H @ S


def flicker(prev_mask, mask, H):
    """Share of crack pixels (either frame) that changed, the previous mask carried onto this frame."""
    if prev_mask is None or H is None:
        return None
    warped = cv2.warpPerspective(prev_mask, H, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST)
    seen = cv2.warpPerspective(np.full(mask.shape, 255, np.uint8), H, (mask.shape[1], mask.shape[0])) > 0
    a, b = (warped > 0) & seen, (mask > 0) & seen
    union = (a | b).sum()
    return float((a ^ b).sum() / union) if union else None


def hud(img, lines):
    cv2.rectangle(img, (0, 0), (img.shape[1], 24 * len(lines) + 6), (0, 0, 0), -1)
    for k, line in enumerate(lines):
        cv2.putText(img, line, (8, 20 + 24 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


cap = cv2.VideoCapture(args.clip)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
turn = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(
    int(cap.get(cv2.CAP_PROP_ORIENTATION_META)) % 360)
s = args.long / max(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w, h = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * s), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * s)  # before turning
writer =imageio.get_writer(args.out, fps=fps / args.every, codec="libx264", quality=8, macro_block_size=8)
times = {"decode": [], "model": [], "steady": [], "loop": []}
flick = {"raw": [], "steady": []}
shares = []
prev_feat = avg = None
prev_masks = {"raw": None, "steady": None}
i = kept = 0
warm = False
while True:
    t0 = time.perf_counter()
    ok, frame = cap.read()
    if not ok:
        break
    i += 1
    if (i - 1) % args.every:
        continue
    t1 = time.perf_counter()
    small = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    if turn is not None:
        small = cv2.rotate(small, turn)
    if not warm:  # the first call pays for cuDNN's algorithm search; keep it out of the numbers
        det.predict(small)
        warm = True
    if cuda:
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    prob = det.predict(small)
    if cuda:
        torch.cuda.synchronize()
    t3 = time.perf_counter()
    raw = det.mask_from_prob(prob)
    feat = features(small)
    H = homography(prev_feat, feat)
    if not args.raw:
        if avg is None or H is None:
            avg = prob
        else:
            size = (prob.shape[1], prob.shape[0])
            # Outside the previous frame, carry the new map itself, so the blend leaves it unchanged there.
            carried = cv2.warpPerspective(avg, H, size, dst=prob.copy(), flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_TRANSPARENT)
            avg = cv2.addWeighted(prob, args.alpha, carried, 1 - args.alpha, 0)
        mask = hysteresis(avg)
    else:
        mask = raw
    t4 = time.perf_counter()
    for key, m in (("raw", raw), ("steady", mask)):
        f = flicker(prev_masks[key], m, H)
        if f is not None:
            flick[key].append(f)
        prev_masks[key] = m
    prev_feat = feat
    share = (mask > 0).mean()
    shares.append(share)
    times["decode"].append(t1 - t0)
    times["model"].append(t3 - t2)
    times["steady"].append(t4 - t3)
    loop = np.mean(times["loop"][-30:]) if times["loop"] else t4 - t0
    key = "raw" if args.raw else "steady"
    recent = np.mean(flick[key][-30:]) if flick[key] else 0.0
    out = hud(det.overlay(small, mask, alpha=0.6), [
        f"DeepCrack 1-class {small.shape[1]}x{small.shape[0]} {gpu}{' bf16' if cuda else ''}"
        + ("" if args.raw else " + steadied"),
        f"model {1000 * (t3 - t2):.0f} ms  loop {1 / loop:.1f} fps  crack {100 * share:.1f}%  flicker {100 * recent:.0f}%"])
    if args.compare and not args.raw:
        left = hud(det.overlay(small, raw, alpha=0.6), [
            "frame by frame (as before)",
            f"crack {100 * (raw > 0).mean():.1f}%  flicker {100 * (np.mean(flick['raw'][-30:]) if flick['raw'] else 0):.0f}%"])
        out = np.hstack([left, np.zeros((out.shape[0], 6, 3), np.uint8), out])
    writer.append_data(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    kept += 1
    times["loop"].append(time.perf_counter() - t0)
writer.close()
ms = {k: 1000 * float(np.median(v)) for k, v in times.items()}
print(f"{args.clip.split(chr(92))[-1]}: {kept} frames at {small.shape[1]}x{small.shape[0]} on {gpu}; median decode {ms['decode']:.0f} ms, "
      f"model {ms['model']:.0f} ms, steadying {ms['steady']:.0f} ms, loop {ms['loop']:.0f} ms = "
      f"{1000 / ms['loop']:.1f} fps; crack {100 * np.mean(shares):.1f}%; flicker raw {100 * np.mean(flick['raw']):.0f}%"
      + ("" if args.raw else f" -> steadied {100 * np.mean(flick['steady']):.0f}%") + f" -> {args.out}")
