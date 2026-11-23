"""Live-camera simulation for the crack detector: a clip plays as a 30 fps stream and the detector keeps up as it can.

    python -I stream_sim.py STREAM.mp4 OUT.mp4 [--long 960] [--engine ENGINE] [--alpha 0.25] [--low 0.13]
                            [--seconds N]

STREAM.mp4 stands for what a camera sends: the clips turned upright and brought to 1080p (stream_1080p\\*_1080p.mp4)
so nothing decodes 4K. Portrait or landscape: frames are shrunk to --long on the long side, keeping their shape,
and a rotation tag OpenCV leaves unapplied is applied. --engine runs a TensorRT engine from trt_crack.py (in
.venv-trt) instead of the PyTorch package; its profiles cover both orientations.

Three threads, as on a device:
- camera: decodes the stream and releases frame k at t0 + k/fps on the wall clock, whether or not anyone took
  the previous one;
- detector: always takes the newest released frame (older ones are dropped), shrinks it to --size, runs
  crack_detector_1class_best, and keeps the time average of edge_crack_video.py (homography-warped EMA plus
  hysteresis);
- display: shows every camera frame at 30 fps with the detector's newest result, carried onto the frame by a
  road-plane homography, so the overlay follows the camera between detections.

The output video is what the screen showed, one frame per camera frame. The corner shows the camera rate, the
detector rate, the age of the overlay (capture of the frame it came from to display) and the frames the detector
skipped. A summary line goes to stdout.
"""
import argparse
import importlib.util
import os
import threading
import time

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("stream")
ap.add_argument("out")
ap.add_argument("--long", type=int, default=960, help="frames are shrunk to this long side, portrait or landscape")
ap.add_argument("--alpha", type=float, default=0.25)
ap.add_argument("--low", type=float, default=0.13)
ap.add_argument("--seconds", type=float, default=0)
ap.add_argument("--engine", help="a TensorRT engine from trt_crack.py instead of the PyTorch package")
ap.add_argument("--package", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "deepcrack", "crack_detector_1class_best"))
args = ap.parse_args()

spec = importlib.util.spec_from_file_location("crack_detector", rf"{args.package}\crack_detector.py")
cd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cd)
det = cd.CrackDetector(device="cpu" if args.engine else None)  # threshold, min_area, overlay; the model too without --engine
if args.engine:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from trt_crack import Engine
    predict = Engine(args.engine).predict
else:
    predict = det.predict

cap = cv2.VideoCapture(args.stream)
src_w, src_h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
turn = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(
    int(cap.get(cv2.CAP_PROP_ORIENTATION_META)) % 360)  # a tag OpenCV leaves unapplied, as on the raw clips
s = args.long / max(src_w, src_h)
size = (round(src_w * s), round(src_h * s))  # before turning
w, h = (size[1], size[0]) if turn in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE) else size
FEAT_SCALE = 0.5


class Features:
    """ORB on a half-size frame; homographies come back at full --size."""

    def __init__(self):
        self.orb = cv2.ORB_create(1500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def __call__(self, img):
        gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None, fx=FEAT_SCALE, fy=FEAT_SCALE,
                          interpolation=cv2.INTER_AREA)
        return self.orb.detectAndCompute(gray, None)

    def homography(self, a, b):
        if a is None or b is None or a[1] is None or b[1] is None or len(a[0]) < 30 or len(b[0]) < 30:
            return None
        m = self.matcher.match(a[1], b[1])
        if len(m) < 30:
            return None
        pa = np.float32([a[0][x.queryIdx].pt for x in m])
        pb = np.float32([b[0][x.trainIdx].pt for x in m])
        H, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 2.0)
        if H is None or inl.sum() < 25:
            return None
        S = np.diag([FEAT_SCALE, FEAT_SCALE, 1.0])
        return np.linalg.inv(S) @ H @ S


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


fps = cap.get(cv2.CAP_PROP_FPS) or 30
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if args.seconds:
    total = min(total, int(args.seconds * fps))
lock = threading.Condition()
state = {"frame": None, "k": -1, "t": 0.0, "done": False}  # newest released camera frame
result = {"mask": None, "feat": None, "k": -1, "t": 0.0}  # newest detection
shown = []  # display queue: (k, capture time, small frame)
stats = {"detected": [], "model_ms": [], "age_ms": [], "skipped": 0}
predict(np.zeros((h, w, 3), np.uint8))  # cuDNN's algorithm search before the clock starts


def camera():
    t0 = time.perf_counter() + 0.2
    for k in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        if turn is not None:
            small = cv2.rotate(small, turn)
        due = t0 + k / fps
        while time.perf_counter() < due:
            time.sleep(0.001)
        with lock:
            state.update(frame=small, k=k, t=time.perf_counter())
            shown.append((k, state["t"], small))
            lock.notify_all()
    with lock:
        state["done"] = True
        lock.notify_all()


def detector():
    feats = Features()
    avg = prev_feat = None
    last = prev_k = -1
    while True:
        with lock:
            while state["k"] == last and not state["done"]:
                lock.wait()
            if state["k"] == last and state["done"]:
                return
            frame, k, t = state["frame"], state["k"], state["t"]
        stats["skipped"] += k - last - 1
        last = k
        t1 = time.perf_counter()
        prob = predict(frame)
        torch.cuda.synchronize()
        stats["model_ms"].append(1000 * (time.perf_counter() - t1))
        feat = feats(frame)
        H = feats.homography(prev_feat, feat)
        if avg is None or H is None:
            avg = prob
        else:
            # The average fades per camera frame, not per detection: with frames skipped between two
            # detections the old map has drifted further and would otherwise leave a second, offset line.
            a = 1 - (1 - args.alpha) ** (k - prev_k)
            carried = cv2.warpPerspective(avg, H, (w, h), dst=prob.copy(), flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_TRANSPARENT)
            avg = cv2.addWeighted(prob, a, carried, 1 - a, 0)
        prev_feat, prev_k = feat, k
        mask = hysteresis(avg)
        with lock:
            result.update(mask=mask, feat=feat, k=k, t=t)
        stats["detected"].append(time.perf_counter())


threads = [threading.Thread(target=camera, daemon=True), threading.Thread(target=detector, daemon=True)]
for th in threads:
    th.start()
writer = imageio.get_writer(args.out, fps=fps, codec="libx264", quality=8, macro_block_size=8)
feats = Features()
done = 0
while done < total:
    with lock:
        while not shown and not state["done"]:
            lock.wait(0.05)
        if not shown:
            break
        k, t_cap, small = shown.pop(0)
        res = dict(result)
    feat = feats(small)
    view = small
    if res["mask"] is not None:
        H = feats.homography(res["feat"], feat) if res["k"] != k else np.eye(3)
        mask = res["mask"] if H is None else cv2.warpPerspective(res["mask"], H, (w, h), flags=cv2.INTER_NEAREST)
        view = det.overlay(small, mask, alpha=0.6)
        age = 1000 * (time.perf_counter() - res["t"])
        stats["age_ms"].append(age)
    else:
        age = float("nan")
    recent = [x for x in stats["detected"] if x > time.perf_counter() - 1]
    view = view.copy()
    cv2.rectangle(view, (0, 0), (w, 54), (0, 0, 0), -1)
    backend = "TensorRT" if args.engine else "PyTorch bf16"
    for i, line in enumerate([f"stream {w}x{h} @ {fps:.0f} fps   {backend}   frame {k}",
                              f"detector {len(recent)} fps   overlay age {age:.0f} ms   skipped {stats['skipped']}"]):
        cv2.putText(view, line, (8, 20 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    writer.append_data(cv2.cvtColor(view, cv2.COLOR_BGR2RGB))
    done += 1
    if len(shown) > 2 * fps:
        print(f"  display is {len(shown)} frames behind", flush=True)
writer.close()
for th in threads:
    th.join(timeout=5)
n_det = len(stats["model_ms"])
span = (stats["detected"][-1] - stats["detected"][0]) if n_det > 1 else 1
print(f"{args.stream.split(chr(92))[-1]}: {done} frames shown at {fps:.0f} fps; detector ran on {n_det} "
      f"({stats['skipped']} skipped), {(n_det - 1) / span:.1f} fps, model median {np.median(stats['model_ms']):.0f} ms; "
      f"overlay age median {np.nanmedian(stats['age_ms']):.0f} ms, 95% {np.nanpercentile(stats['age_ms'], 95):.0f} ms "
      f"-> {args.out}")
