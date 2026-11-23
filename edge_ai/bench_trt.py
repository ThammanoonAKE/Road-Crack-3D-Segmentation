"""GPU time of the crack model per frame: PyTorch (the package, bf16 autocast) against TensorRT engines, portrait
544x960 and landscape 960x544, input already on the GPU (no copies, no pre/post-processing).

    python -I bench_trt.py ENGINE [ENGINE ...] [--iters 300] [--layers]

Each case: 30 warm-up runs, then --iters runs timed with CUDA events; median and 95th percentile in ms.
--layers: per-layer TensorRT profile of each engine (one orientation), summed into the ConvNeXt MLPs (the
Linear layers, plus their FP4 quantize nodes), the rest of the backbone, and everything after it.
"""
import argparse
import os
import sys

import numpy as np
import tensorrt as trt
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trt_crack import Engine, package  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("engines", nargs="+")
ap.add_argument("--iters", type=int, default=300)
ap.add_argument("--layers", action="store_true")
args = ap.parse_args()
SHAPES = {"portrait 544x960": (960, 544), "landscape 960x544": (544, 960)}


def timed(fn):
    for _ in range(30):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(args.iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b))
    return np.median(out), np.percentile(out, 95)


rows = []
det = package().CrackDetector(device="cuda")
for key, (h, w) in SHAPES.items():
    x = torch.rand(1, 3, h, w, device="cuda")

    def run():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            det.model(x)
    rows.append(("PyTorch bf16 autocast", key, *timed(run)))
del det
torch.cuda.empty_cache()


class Profiler(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.ms = {}

    def report_layer_time(self, name, ms):
        self.ms[name] = self.ms.get(name, 0.0) + ms


for path in args.engines:
    eng = Engine(path)
    name = os.path.basename(path)
    for H, W, ctx, y in eng.profiles:
        key = "portrait 544x960" if H > W else "landscape 960x544"
        x = torch.rand(1, 3, H, W, device="cuda").to(eng.in_dtype)
        ctx.set_tensor_address(eng.inp, x.data_ptr())
        ctx.set_tensor_address(eng.out, y.data_ptr())
        stream = torch.cuda.current_stream()

        def run():
            ctx.execute_async_v3(stream.cuda_stream)
        rows.append((name, key, *timed(run)))
        if args.layers and key.startswith("portrait"):
            prof = Profiler()
            ctx.profiler = prof
            for _ in range(20):
                ctx.execute_async_v3(stream.cuda_stream)
                stream.synchronize()
            ctx.profiler = None
            groups = {"MLP (Linear + FP4 quantize)": 0.0, "rest of backbone": 0.0, "decoder, detail, sides": 0.0}
            for layer, ms in prof.ms.items():
                if "backbone" in layer and ("fc1" in layer or "fc2" in layer or "block.3" in layer
                                            or "block.5" in layer or "quantizer" in layer):
                    groups["MLP (Linear + FP4 quantize)"] += ms / 20
                elif "backbone" in layer or "out_norms" in layer:
                    groups["rest of backbone"] += ms / 20
                else:
                    groups["decoder, detail, sides"] += ms / 20
            total = sum(groups.values())
            print(f"{name} layers, portrait: " + ", ".join(f"{k} {v:.2f} ms ({100 * v / total:.0f}%)"
                                                            for k, v in groups.items()) + f"; sum {total:.2f} ms")
            top = sorted(prof.ms.items(), key=lambda kv: -kv[1])[:6]
            for layer, ms in top:
                print(f"    {ms / 20:6.3f} ms  {layer[:150]}")
    del eng
    torch.cuda.empty_cache()

print(f"GPU {torch.cuda.get_device_name(0)}, {args.iters} runs each, input on the GPU, ms per frame:")
for name, key, med, p95 in rows:
    print(f"  {name:34s} {key:18s} median {med:6.2f}   p95 {p95:6.2f}   ({1000 / med:.0f} fps)")
