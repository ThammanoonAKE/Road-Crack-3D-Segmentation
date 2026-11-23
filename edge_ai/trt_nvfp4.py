"""crack_detector_1class_best with NVFP4 on its Linear layers (the ConvNeXt MLPs), everything else BF16, as an ONNX
graph for TensorRT (build it with trt_crack.py build ... --strongly-typed).

    python -I trt_nvfp4.py OUT.onnx [--calib 32] [--compare 12]

TensorRT runs NVFP4 only in MatMul/Gemm, not in convolutions, so only nn.Linear gets quantizers: weights and inputs
in NVFP4 (E2M1 values, one E4M3 scale per 16 along the channel axis, per-tensor amax by max calibration). Conv,
LayerNorm, the decoder and the detail branch stay unquantized and ModelOpt's export turns them to BF16.

Calibration: --calib road photos (other ones than trt_crack.py check uses), each upright 544x960 and as filmed
960x544, in float32. --compare: before exporting, PyTorch with the quantizers simulated against the plain model on
that many other photos (mean |prob diff|, mask IoU), to tell the quantization's own error from the engine's.
"""
import argparse
import glob
import os
import sys
import time
import warnings

import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trt_crack import package  # noqa: E402

import modelopt.torch.quantization as mtq  # noqa: E402
from modelopt.torch._deploy.utils.torch_onnx import OnnxBytes, get_onnx_bytes_and_metadata  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--calib", type=int, default=32)
ap.add_argument("--compare", type=int, default=12)
args = ap.parse_args()

class FlatMLP(torch.nn.Module):
    """A ConvNeXt block's Linear, GELU, Linear on (N, H*W, C) instead of (N, H, W, C): TensorRT's FP4
    DynamicQuantize takes 2D or 3D tensors only. Same layers and weights, same result."""

    def __init__(self, fc1, act, fc2):
        super().__init__()
        self.fc1, self.act, self.fc2 = fc1, act, fc2

    def forward(self, x):
        n, h, w, c = x.shape
        y = self.fc2(self.act(self.fc1(x.reshape(n, h * w, c))))
        return y.reshape(n, h, w, y.shape[-1])


det = package().CrackDetector(device="cuda")
model = det.model.float().eval()
for block in [m.block for m in model.modules() if type(m).__name__ == "CNBlock"]:
    assert isinstance(block[3], torch.nn.Linear) and isinstance(block[5], torch.nn.Linear), block
    block[3] = FlatMLP(block[3], block[4], block[5])
    block[4], block[5] = torch.nn.Identity(), torch.nn.Identity()
files = sorted(glob.glob(r"C:\3D-segmentation-data\road_crack\*\scene\images\*.png"))


def tensors(paths):
    for f in paths:
        flat = cv2.resize(cv2.imread(f), (960, 540), interpolation=cv2.INTER_AREA)
        for bgr in (cv2.rotate(flat, cv2.ROTATE_90_CLOCKWISE), flat):
            x = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).cuda().permute(2, 0, 1)[None].float() / 255
            h, w = x.shape[-2:]
            yield torch.nn.functional.pad(x, (0, (-w) % 32, 0, (-h) % 32), mode="reflect"), (h, w)


calib = files[11:: max(1, len(files) // args.calib)][: args.calib]
held = files[17:: max(1, len(files) // args.compare)][: args.compare]
reference = []  # the plain model (with FlatMLP, which changes nothing) in float32
with torch.no_grad():
    for x, (h, w) in tensors(held):
        reference.append(torch.sigmoid(model(x))[0, 0, :h, :w].cpu().numpy())

nvfp4 = next(e["cfg"] for e in mtq.NVFP4_DEFAULT_CFG["quant_cfg"] if e.get("quantizer_name") == "*weight_quantizer")
cfg = {"quant_cfg": [{"quantizer_name": "*", "enable": False},
                     {"quantizer_name": "*weight_quantizer", "parent_class": "nn.Linear", "cfg": nvfp4},
                     {"quantizer_name": "*input_quantizer", "parent_class": "nn.Linear", "cfg": nvfp4}],
       "algorithm": "max"}


def forward_loop(m):
    with torch.no_grad():
        for x, _ in tensors(calib):
            m(x)


t0 = time.time()
model = mtq.quantize(model, cfg, forward_loop)
on = [n for n, m in model.named_modules() if n.endswith("_quantizer") and getattr(m, "is_enabled", False)]
linears = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
print(f"calibrated on {2 * len(calib)} frames in {time.time() - t0:.0f} s; {len(on)} quantizers on, "
      f"{len(linears)} Linear layers, e.g. {on[:2]}", flush=True)

diffs, ious = [], []
with torch.no_grad():
    for (x, (h, w)), ref in zip(tensors(held), reference):
        prob = torch.sigmoid(model(x))[0, 0, :h, :w].cpu().numpy()
        a, b = det.mask_from_prob(ref) > 0, det.mask_from_prob(prob) > 0
        union = (a | b).sum()
        diffs.append(np.abs(prob - ref).mean())
        ious.append((a & b).sum() / union if union else 1.0)
print(f"PyTorch, NVFP4 simulated vs float32, {len(diffs)} frames: mean |prob diff| {np.mean(diffs):.4f}, "
      f"mask IoU {np.mean(ious):.3f} (min {np.min(ious):.3f})", flush=True)

# ModelOpt's NVFP4 exporter casts the FP4 MatMuls to the precision of the model's weights, taken as FP16 unless
# it finds BF16 ones; a float32 model gets FP16 MatMuls next to float32 biases, which TensorRT's strong typing
# rejects. So the calibrated model goes to BF16 as a whole before export (input too), and the graph comes out BF16.
t0 = time.time()
model = model.to(torch.bfloat16)
for m in model.modules():
    if hasattr(m, "trt_high_precision_dtype"):
        m.trt_high_precision_dtype = "BFloat16"
dummy = torch.rand(1, 3, 960, 544, device="cuda", dtype=torch.bfloat16)
blob, _ = get_onnx_bytes_and_metadata(model, (dummy,), dynamic_axes={"x": {2: "height", 3: "width"}},
                                      weights_dtype="fp32")  # "fp32": leave the dtypes as they are (BF16)
onnx_bytes = OnnxBytes.from_bytes(blob)
out_dir = os.path.dirname(os.path.abspath(args.out))
for name, data in onnx_bytes.onnx_model.items():
    path = os.path.join(out_dir, os.path.basename(args.out) if name.endswith(".onnx") else name)
    with open(path, "wb") as f:
        f.write(data)
    print(f"wrote {path} ({len(data) / 1e6:.0f} MB)")
print(f"exported in {time.time() - t0:.0f} s")
