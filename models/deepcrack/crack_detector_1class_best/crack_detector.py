"""Crack segmentation with DeepCrack + ConvNeXt-Small (1 class: crack), trained on public crack datasets plus
our own line-scan road scans, with heavy augmentation.

Self-contained: needs only torch, torchvision, numpy and opencv-python, plus the weights file.

Python:
    from crack_detector import CrackDetector
    det = CrackDetector()                    # weights next to this file, GPU if available
    mask = det.predict_mask("road.jpg")      # HxW uint8, 255 = crack
    prob = det.predict("road.jpg")           # HxW float32 crack probability in [0, 1]
    prob = det.predict(frame)                # numpy BGR image (cv2) also works; color="rgb" for RGB / PIL

Command line:
    python crack_detector.py photos/ --out results/ --overlay
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepcrack_1class_2026-11-20.pt")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ----------------------------------------------------------------------------------------------
# Model (same layer names as the training code, so the state dict loads strictly)
# ----------------------------------------------------------------------------------------------
def _conv_bn(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DeepCrackConvNeXt(nn.Module):
    """DeepCrack-style network: ConvNeXt-Small encoder, a full / half resolution detail branch,
    a U-Net decoder, a side output at each of 6 scales and a 1x1 conv fusing them.
    Input: RGB in [0, 1], (N, 3, H, W) with H, W multiples of 32. Output: fused logits (N, 1, H, W)."""

    ENC_CH = (96, 192, 384, 768)            # ConvNeXt-Small at strides 4, 8, 16, 32
    DETAIL = (32, 64)                       # strides 1, 2
    DECODER = (32, 64, 128, 128, 256, 256)  # strides 1 .. 32

    def __init__(self):
        super().__init__()
        from torchvision.models import convnext_small
        from torchvision.models.convnext import LayerNorm2d

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.backbone = convnext_small(weights=None).features
        self.out_norms = nn.ModuleList(LayerNorm2d(c, eps=1e-6) for c in self.ENC_CH)
        d1, d2 = self.DETAIL
        self.detail1 = nn.Sequential(_conv_bn(3, d1), _conv_bn(d1, d1))
        self.detail2 = nn.Sequential(_conv_bn(d1, d2, stride=2), _conv_bn(d2, d2))
        enc = (d1, d2) + self.ENC_CH
        dec = self.DECODER
        self.top = _conv_bn(enc[5], dec[5])
        self.decoders = nn.ModuleList(
            nn.Sequential(_conv_bn(dec[k + 1] + enc[k], dec[k]), _conv_bn(dec[k], dec[k])) for k in range(5))
        self.sides = nn.ModuleList(nn.Conv2d(e + d, 1, 1) for e, d in zip(enc, dec))
        self.fuse = nn.Conv2d(len(enc), 1, 1)

    def forward(self, x):
        x = (x - self.mean) / self.std
        size = x.shape[-2:]
        e1 = self.detail1(x)
        enc = [e1, self.detail2(e1)]
        h = x
        for i, layer in enumerate(self.backbone):
            h = layer(h)
            if i in (1, 3, 5, 7):
                enc.append(self.out_norms[len(enc) - 2](h))
        d = self.top(enc[5])
        decs = [None] * 5 + [d]
        for k in reversed(range(5)):
            d = F.interpolate(d, size=enc[k].shape[-2:], mode="bilinear", align_corners=False)
            d = self.decoders[k](torch.cat([d, enc[k]], dim=1))
            decs[k] = d
        sides = []
        for k, (e, d) in enumerate(zip(enc, decs)):
            s = self.sides[k](torch.cat([e, d], dim=1))
            sides.append(s if k == 0 else F.interpolate(s, size=size, mode="bilinear", align_corners=False))
        return self.fuse(torch.cat(sides, dim=1))


# ----------------------------------------------------------------------------------------------
# Inference wrapper
# ----------------------------------------------------------------------------------------------
def read_image(path):
    """cv2.imread that also handles non-ASCII (e.g. Thai) paths on Windows. Returns the image as stored:
    8- or 16-bit, gray, BGR or BGRA (to_rgb / to_bgr8 convert it)."""
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"cannot read image {path}")
    return img


def to_bgr8(img, input_bits=16):
    """Display version of a numpy image: BGR uint8 (16-bit: top bits, as fed to the model)."""
    if img.dtype == np.uint16:
        img = (img >> max(input_bits - 8, 0)).clip(0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.float32)
        img = np.clip(img * 255.0 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(img[..., :3])


def write_image(path, img, params=()):
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img, list(params))
    if not ok:
        raise IOError(f"cannot encode {path}")
    buf.tofile(path)


class CrackDetector:
    """
    weights:   path of deepcrack_1class_2026-11-20.pt
    device:    "cuda", "cpu", ... (default: cuda if available)
    threshold: probability above which a pixel is a crack (default: the value stored with the
               weights, 0.16, chosen on the validation split)
    min_area:  predict_mask drops crack regions smaller than this many pixels (default: stored with the
               weights, 150, the setting the in-house scan numbers were measured with; 0 = keep all)
    amp:       bf16 autocast on GPU (the setting the reported accuracy was measured with)
    tile:      images with more than tile x tile pixels are processed in overlapping tiles of this
               size. Default 2048: a 2048 x 2048 scan runs in one pass, as in the evaluation (~5.8 GB GPU
               memory). 1024 needs ~2 GB (on scans ~3% more crack pixels than one pass). 0 = never tile.
    input_bits: bit depth of uint16 numpy inputs. 16 (default) = full 0-65535 range, as cv2.imread
               with IMREAD_UNCHANGED returns 16-bit PNGs (incl. Mono12 data saved shifted to 16 bit);
               12 = raw Mono12 camera buffers with values 0-4095.
    """

    def __init__(self, weights=DEFAULT_WEIGHTS, device=None, threshold=None, amp=True, tile=2048, overlap=128,
                 input_bits=16, min_area=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(weights, map_location="cpu", weights_only=True)
        self.model = DeepCrackConvNeXt()
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        self.threshold = float(ckpt.get("threshold", 0.5) if threshold is None else threshold)
        self.min_area = int(ckpt.get("min_area", 0) if min_area is None else min_area)
        self.amp = amp and self.device.type == "cuda"
        self.tile = tile
        self.overlap = overlap
        self.input_bits = input_bits
        self.info = {k: v for k, v in ckpt.items() if k != "state_dict"}

    # -- input handling --
    def to_rgb(self, image, color="bgr"):
        """Path, PIL image or numpy array (HxW gray, HxWx3, HxWx4) -> RGB uint8 HxWx3.
        color: channel order of a numpy input ("bgr" = cv2.imread / OpenCV video frames).
        uint8 is used as is, uint16 is scaled by input_bits, float is taken as 0-1 (or 0-255 if
        its maximum is above 1). Image files are read as stored (16-bit files: top 8 bits, as in training)."""
        if isinstance(image, (str, os.PathLike)):
            image = read_image(os.fspath(image))
            color = "bgr"
        if not isinstance(image, np.ndarray):  # PIL.Image
            return np.asarray(image.convert("RGB"))
        img = image
        if img.dtype == np.uint16:
            img = (img >> max(self.input_bits - 8, 0)).clip(0, 255).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = img.astype(np.float32)
            img = np.clip(img * 255.0 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            return np.repeat(img[..., None], 3, axis=2)
        if img.ndim != 3 or img.shape[2] not in (3, 4):
            raise ValueError(f"expected an HxW, HxWx3 or HxWx4 image, got shape {img.shape}")
        img = img[..., :3]
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if color.lower() == "bgr" else np.ascontiguousarray(img)

    # -- core --
    @torch.no_grad()
    def _forward(self, x):
        """x: (N, 3, H, W) float RGB in [0, 1] on device -> crack probability (N, H, W)."""
        h, w = x.shape[-2:]
        ph, pw = (-h) % 32, (-w) % 32
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect" if ph < h and pw < w else "replicate")
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp):
            logits = self.model(x)
        return torch.sigmoid(logits.float())[:, 0, :h, :w]

    def _ramp(self, n, first, last):
        w = np.ones(n, np.float32)
        o = min(self.overlap, n // 2)
        if o > 0:
            r = np.linspace(0, 1, o + 2, dtype=np.float32)[1:-1]
            if not first:
                w[:o] = r
            if not last:
                w[-o:] = r[::-1]
        return w

    def _predict_rgb(self, rgb):
        x = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1)[None].float().div_(255.0)
        H, W = rgb.shape[:2]
        t = self.tile
        if not t or H * W <= t * t:
            return self._forward(x)[0].cpu().numpy()
        # Overlapping tiles, blended with weights that fade out towards the inner tile borders.
        th, tw = min(t, H), min(t, W)
        stride = max(t - self.overlap, 32)
        ys = sorted(set(list(range(0, H - th, stride)) + [H - th]))
        xs = sorted(set(list(range(0, W - tw, stride)) + [W - tw]))
        acc = np.zeros((H, W), np.float32)
        wsum = np.zeros((H, W), np.float32)
        for y in ys:
            wy = self._ramp(th, y == ys[0], y == ys[-1])
            for x0 in xs:
                wx = self._ramp(tw, x0 == xs[0], x0 == xs[-1])
                p = self._forward(x[..., y:y + th, x0:x0 + tw])[0].cpu().numpy()
                wt = wy[:, None] * wx[None, :]
                acc[y:y + th, x0:x0 + tw] += p * wt
                wsum[y:y + th, x0:x0 + tw] += wt
        return acc / wsum

    # -- public API --
    def predict(self, image, color="bgr", scale=1.0):
        """Crack probability map, float32 HxW in [0, 1], same size as the input.
        scale: resize the image by this factor before inference (the probability map is resized back).
        Use < 1 for photos where cracks look much wider / the image is much larger than 448x448
        crack close-ups, so that cracks appear at roughly the scale the model was trained on."""
        rgb = self.to_rgb(image, color)
        H, W = rgb.shape[:2]
        if scale != 1.0:
            size = (max(1, round(W * scale)), max(1, round(H * scale)))
            rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        prob = self._predict_rgb(rgb)
        if prob.shape != (H, W):
            prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
        return prob

    def mask_from_prob(self, prob, threshold=None, min_area=None):
        """Probability map -> binary crack mask, uint8 HxW (255 = crack), regions under min_area px removed."""
        thr = self.threshold if threshold is None else threshold
        ma = self.min_area if min_area is None else min_area
        m = (prob > thr).astype(np.uint8)
        if ma > 0 and m.any():
            n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            keep = np.concatenate([[False], st[1:, cv2.CC_STAT_AREA] >= ma])
            m = keep[lab].astype(np.uint8)
        return m * 255

    def predict_mask(self, image, color="bgr", scale=1.0, threshold=None, min_area=None):
        """Binary crack mask, uint8 HxW (255 = crack, 0 = background)."""
        return self.mask_from_prob(self.predict(image, color, scale), threshold, min_area)

    @torch.no_grad()
    def predict_batch(self, images, color="bgr", batch_size=8):
        """Probability maps for a list of images. Same-size images up to tile x tile pixels run in
        GPU batches (faster); others fall back to predict()."""
        rgbs = [self.to_rgb(im, color) for im in images]
        out = [None] * len(rgbs)
        groups = {}
        for i, r in enumerate(rgbs):
            if not self.tile or r.shape[0] * r.shape[1] <= self.tile * self.tile:
                groups.setdefault(r.shape, []).append(i)
            else:
                out[i] = self._predict_rgb(r)
        for idx in groups.values():
            for s in range(0, len(idx), batch_size):
                chunk = idx[s:s + batch_size]
                x = torch.from_numpy(np.stack([rgbs[i] for i in chunk])).to(self.device)
                probs = self._forward(x.permute(0, 3, 1, 2).float().div_(255.0)).cpu().numpy()
                for i, p in zip(chunk, probs):
                    out[i] = p
        return out

    @staticmethod
    def overlay(image_bgr, mask, color=(0, 0, 255), alpha=0.6):
        """Draw the mask on an image (red by default). Gray / 16-bit images are converted to BGR uint8 first."""
        out = to_bgr8(image_bgr).copy()
        m = mask > 0
        out[m] = (out[m] * (1 - alpha) + np.array(color, np.float32) * alpha).astype(np.uint8)
        return out


# ----------------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Detect cracks in images (DeepCrack + ConvNeXt-Small).")
    p.add_argument("inputs", nargs="+", help="image files and/or folders")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--thr", type=float, default=None, help="threshold (default: stored with the weights, 0.16)")
    p.add_argument("--scale", type=float, default=1.0, help="resize factor before inference")
    p.add_argument("--min-area", type=int, default=None,
                   help="drop crack regions smaller than this (default: stored with the weights, 150; 0 = keep all)")
    p.add_argument("--overlay", action="store_true", help="also save <name>_overlay.jpg")
    p.add_argument("--prob", action="store_true", help="also save <name>_prob.png (probability x 255)")
    p.add_argument("--tile", type=int, default=2048, help="tile size (lower it, e.g. 1024, if GPU memory is short)")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    try:  # a Windows console (cp1252 etc.) cannot print e.g. Thai file names; don't crash on them
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass

    files = []
    for inp in args.inputs:
        if os.path.isdir(inp):
            files += [os.path.join(inp, f) for f in sorted(os.listdir(inp)) if os.path.splitext(f)[1].lower() in IMG_EXTS]
        else:
            files.append(inp)
    os.makedirs(args.out, exist_ok=True)
    det = CrackDetector(args.weights, device="cpu" if args.cpu else None, threshold=args.thr, min_area=args.min_area,
                        tile=args.tile)
    print(f"device {det.device}, threshold {det.threshold}, min area {det.min_area} px, {len(files)} images")
    for f in files:
        bgr = read_image(f)
        prob = det.predict(bgr, color="bgr", scale=args.scale)
        mask = det.mask_from_prob(prob)
        stem = os.path.join(args.out, os.path.splitext(os.path.basename(f))[0])
        write_image(stem + "_mask.png", mask)
        if args.prob:
            write_image(stem + "_prob.png", (prob * 255).round().astype(np.uint8))
        if args.overlay:
            write_image(stem + "_overlay.jpg", det.overlay(bgr, mask), (cv2.IMWRITE_JPEG_QUALITY, 90))
        print(f"{os.path.basename(f)}: crack {100.0 * (mask > 0).mean():.2f}% of pixels")


if __name__ == "__main__":
    main()
