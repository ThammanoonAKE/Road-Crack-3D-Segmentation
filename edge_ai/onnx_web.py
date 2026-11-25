"""crack_detector_1class_best as one ONNX file small enough to download into a web page (web/, the /detect page).

    python -I onnx_web.py OUT.onnx [--check PHOTO ...] [--long 960]

The graph is the one trt_crack.py exports (normalisation inside, input RGB 0-1, output logits, height and width
left open, opset 17) and still computes in float32, so it runs on every onnxruntime backend. Only the storage
changes: each large weight tensor is written as float16 and followed by a Cast back to float32, which halves the
file (208 MB -> about 105 MB). onnxruntime folds those Casts away when it loads the model.

--check runs the file with onnxruntime on the CPU and the package in PyTorch (float32, CPU) on each photo, shrunk
to --long on its long side as the page does, and prints how far apart the two are.
"""
import argparse
import importlib.util
import os
import tempfile

import cv2
import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the repository
PACKAGE = os.path.join(ROOT, "models", "deepcrack", "crack_detector_1class_best")
SMALL = 1024  # tensors with fewer values stay float32: they are biases and scales, and cost nothing to keep exact


def package():
    spec = importlib.util.spec_from_file_location("crack_detector", os.path.join(PACKAGE, "crack_detector.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def export(path):
    det = package().CrackDetector(device="cpu")
    x = torch.rand(1, 3, 960, 544)
    torch.onnx.export(det.model.float().eval(), (x,), path, input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {2: "height", 3: "width"}, "logits": {2: "height", 3: "width"}},
                      opset_version=17, dynamo=False)
    return det


def halve(src, dst):
    model = onnx.load(src)
    graph = model.graph
    casts, kept, halved = [], 0, 0
    for i, init in enumerate(graph.initializer):
        if init.data_type != TensorProto.FLOAT:
            continue
        values = numpy_helper.to_array(init)
        if values.size < SMALL:
            kept += values.size
            continue
        name = init.name
        graph.initializer[i].CopyFrom(numpy_helper.from_array(values.astype(np.float16), name + "_f16"))
        casts.append(helper.make_node("Cast", [name + "_f16"], [name], to=TensorProto.FLOAT, name=name + "_to_f32"))
        halved += values.size
    nodes = casts + list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes)
    onnx.checker.check_model(model)
    onnx.save(model, dst)
    print(f"{halved / 1e6:.1f} M values stored as float16, {kept} kept as float32, "
          f"{os.path.getsize(dst) / 1e6:.1f} MB -> {dst}")


def check(det, path, photos, long_side):
    import onnxruntime as ort
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    for photo in photos:
        bgr = cv2.imdecode(np.fromfile(photo, np.uint8), cv2.IMREAD_COLOR)
        h, w = bgr.shape[:2]
        s = min(1.0, long_side / max(h, w))
        if s < 1:
            bgr = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        h, w = bgr.shape[:2]
        x = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float() / 255
        x = torch.nn.functional.pad(x, (0, (-w) % 32, 0, (-h) % 32), mode="reflect")
        with torch.no_grad():
            ref = torch.sigmoid(det.model(x))[0, 0, :h, :w].numpy()
        out = session.run(None, {"image": x.numpy()})[0]
        got = (1 / (1 + np.exp(-out)))[0, 0, :h, :w]
        a, b = ref > det.threshold, got > det.threshold
        iou = (a & b).sum() / max(1, (a | b).sum())
        print(f"{os.path.basename(photo).encode('ascii', 'replace').decode()}: {w}x{h}, "
              f"mean |prob difference| {np.abs(ref - got).mean():.6f}, max {np.abs(ref - got).max():.4f}, "
              f"mask IoU {iou:.4f}, crack {100 * b.mean():.2f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--check", nargs="*", default=[], metavar="PHOTO")
    ap.add_argument("--long", type=int, default=960, help="long side the photos are shrunk to for --check")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        full = os.path.join(tmp, "full.onnx")
        det = export(full)
        halve(full, args.out)
    if args.check:
        check(det, args.out, args.check, args.long)


if __name__ == "__main__":
    main()
