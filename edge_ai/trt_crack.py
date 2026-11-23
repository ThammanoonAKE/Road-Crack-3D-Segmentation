"""crack_detector_1class_best as a TensorRT engine for portrait and landscape frames, and how far its output moves
from the PyTorch package.

    python -I trt_crack.py export OUT.onnx
    python -I trt_crack.py build IN.onnx OUT.engine [--shapes 544x960,960x544] [--bf16] [--strongly-typed]
    python -I trt_crack.py check OUT.engine [--frames N]

export: the model alone (normalisation inside, input RGB 0-1, output logits), float32 weights, opset 17, with the
        height and width left open.
build:  one TensorRT optimisation profile per --shapes entry (W x H, multiples of 32), each fixed, so each runs at
        full speed: by default portrait 544x960 and landscape 960x544, for frames shrunk to 960 on the long side
        (540 pads to 544, as the package itself pads). --bf16 lets TensorRT pick BF16 kernels per layer (weakly
        typed; gone in TensorRT 11, which builds strongly typed only). --strongly-typed takes the precisions
        written in the ONNX: a BF16 graph from `python -m modelopt.onnx.autocast -t bf16`, or a ModelOpt Q/DQ graph.
check:  runs the engine and the package (PyTorch, bf16 autocast, as shipped) on N road photos, each both
        upright (540x960) and as filmed (960x540), and prints how far apart they are: mean |prob difference|, IoU
        and F-score of the 0.16 / 150 px masks, and the engine's median time per frame for each orientation.

Engine.predict takes any BGR frame: it uses the profile of the frame's orientation, pads up to it, and shrinks a
frame that does not fit (then scales the map back), as CrackDetector.predict does with scale.
"""
import argparse
import glob
import importlib.util
import os
import time

import cv2
import numpy as np
import torch

PACKAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "deepcrack", "crack_detector_1class_best")


def package():
    spec = importlib.util.spec_from_file_location("crack_detector", rf"{PACKAGE}\crack_detector.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Engine:
    def __init__(self, path):
        import tensorrt as trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inp = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.out = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        dtype = {trt.float32: torch.float32, trt.float16: torch.float16, trt.bfloat16: torch.bfloat16}
        self.in_dtype = dtype[self.engine.get_tensor_dtype(self.inp)]
        self.out_dtype = dtype[self.engine.get_tensor_dtype(self.out)]
        self.stream = torch.cuda.Stream()
        self.profiles = []  # (H, W, context, output buffer), one per optimisation profile
        for i in range(self.engine.num_optimization_profiles):
            _, opt, _ = self.engine.get_tensor_profile_shape(self.inp, i)
            ctx = self.engine.create_execution_context()
            ctx.set_optimization_profile_async(i, self.stream.cuda_stream)
            ctx.set_input_shape(self.inp, tuple(opt))
            y = torch.empty(tuple(ctx.get_tensor_shape(self.out)), device="cuda", dtype=self.out_dtype)
            self.profiles.append((opt[2], opt[3], ctx, y))
        self.stream.synchronize()

    def profile_for(self, h, w):
        """The profile of this orientation that fits h x w, else the largest one of it (the frame is shrunk)."""
        same = [p for p in self.profiles if (p[1] >= p[0]) == (w >= h)] or self.profiles
        fits = [p for p in same if p[0] >= h and p[1] >= w]
        return min(fits, key=lambda p: p[0] * p[1]) if fits else max(same, key=lambda p: p[0] * p[1])

    def predict(self, bgr):
        """BGR uint8 frame, any size -> crack probability, float32, same size."""
        h, w = bgr.shape[:2]
        H, W, ctx, y = self.profile_for(h, w)
        scale = min(1.0, H / h, W / w)
        img = bgr if scale == 1.0 else cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        sh, sw = img.shape[:2]
        x = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).cuda().permute(2, 0, 1)[None].float() / 255
        x = torch.nn.functional.pad(x, (0, W - sw, 0, H - sh), mode="reflect" if H - sh < sh and W - sw < sw else "replicate")
        x = x.to(self.in_dtype).contiguous()
        ctx.set_tensor_address(self.inp, x.data_ptr())
        ctx.set_tensor_address(self.out, y.data_ptr())
        with torch.cuda.stream(self.stream):
            ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        prob = torch.sigmoid(y.float())[0, 0, :sh, :sw].cpu().numpy()
        return prob if scale == 1.0 else cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)


def export(args):
    det = package().CrackDetector(device="cpu")
    x = torch.rand(1, 3, 960, 544)
    torch.onnx.export(det.model.float().eval(), (x,), args.out, input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {2: "height", 3: "width"}, "logits": {2: "height", 3: "width"}},
                      opset_version=17, dynamo=False)
    print(f"exported -> {args.out}")


def build(args):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED) if args.strongly_typed else 0
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(args.onnx):
        raise SystemExit("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.builder_optimization_level = 4
    if args.bf16 and not args.strongly_typed:
        if not hasattr(trt.BuilderFlag, "BF16"):
            raise SystemExit("TensorRT %s has no BF16 builder flag (strongly typed only): convert the ONNX with "
                             "python -m modelopt.onnx.autocast -t bf16 and build it with --strongly-typed" % trt.__version__)
        config.set_flag(trt.BuilderFlag.BF16)
    name = network.get_input(0).name
    for s in args.shapes.split(","):
        W, H = map(int, s.split("x"))
        assert W % 32 == 0 and H % 32 == 0, s
        profile = builder.create_optimization_profile()
        profile.set_shape(name, (1, 3, H, W), (1, 3, H, W), (1, 3, H, W))
        config.add_optimization_profile(profile)
    t0 = time.time()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise SystemExit("build failed")
    with open(args.out, "wb") as f:
        f.write(blob)
    print(f"built {args.shapes} in {time.time() - t0:.0f} s -> {args.out} ({blob.nbytes / 1e6:.0f} MB), "
          f"TensorRT {trt.__version__}")


def check(args):
    det = package().CrackDetector()
    eng = Engine(args.engine)
    files = sorted(glob.glob(r"C:\3D-segmentation-data\road_crack\*\scene\images\*.png"))
    files = files[:: max(1, len(files) // args.frames)][: args.frames]
    rows = {"upright 540x960": [], "as filmed 960x540": []}
    for f in files:
        flat = cv2.resize(cv2.imread(f), (960, 540), interpolation=cv2.INTER_AREA)
        for key, bgr in (("upright 540x960", cv2.rotate(flat, cv2.ROTATE_90_CLOCKWISE)), ("as filmed 960x540", flat)):
            ref = det.predict(bgr)
            eng.predict(bgr)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            prob = eng.predict(bgr)
            ms = 1000 * (time.perf_counter() - t0)
            a, b = det.mask_from_prob(ref) > 0, det.mask_from_prob(prob) > 0
            union = (a | b).sum()
            inter = (a & b).sum()
            rows[key].append((np.abs(prob - ref).mean(), inter / union if union else 1.0,
                              2 * inter / (a.sum() + b.sum()) if union else 1.0, ms))
    for key, r in rows.items():
        r = np.array(r)
        print(f"{args.engine} {key}: {len(r)} frames; mean |prob diff| {r[:, 0].mean():.4f}; mask IoU "
              f"{r[:, 1].mean():.3f} (min {r[:, 1].min():.3f}), F {r[:, 2].mean():.3f}; "
              f"{np.median(r[:, 3]):.1f} ms/frame incl. copies")


if __name__ == "__main__":  # stream_sim.py and trt_nvfp4.py import Engine and package
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("export")
    p.add_argument("out")
    p = sub.add_parser("build")
    p.add_argument("onnx")
    p.add_argument("out")
    p.add_argument("--shapes", default="544x960,960x544")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--strongly-typed", action="store_true")
    p = sub.add_parser("check")
    p.add_argument("engine")
    p.add_argument("--frames", type=int, default=60)
    args = ap.parse_args()
    {"export": export, "build": build, "check": check}[args.cmd](args)
