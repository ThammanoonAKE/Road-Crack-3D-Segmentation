"""Make ModelOpt's NVFP4 ONNX (from a BF16 model) strongly typed in BF16 for TensorRT 11.

    python -I fix_fp4_types.py IN.onnx OUT.onnx

ModelOpt writes each NVFP4 weight as DQ(w_f4, DQ(scale_f8, global_f32)) and each NVFP4 activation as
DQ(x_f4, DQ(scale_f8, global_f32)) after TRT_FP4DynamicQuantize. A DequantizeLinear outputs its scale's type, so
both chains come out float32. ModelOpt then casts them to BF16 before the MatMul, but its cast clean-up drops the
weight-side Cast (stale value_info says the weight is BF16 already), and TensorRT rejects a BF16 x float32 MatMul.

Here the float32 global scale of every FP8 DequantizeLinear becomes BF16, so both chains output BF16 by themselves;
the now redundant Casts to BF16 right after them are bypassed, and the stale value_info is dropped (TensorRT infers
types itself). Only the 1-element global scales change: the FP4 values and FP8 block scales stay as they are.
"""
import sys

import ml_dtypes
import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

src, dst = sys.argv[1], sys.argv[2]
model = onnx.load(src)
g = model.graph
inits = {i.name: i for i in g.initializer}
producer = {o: n for n in g.node for o in n.output}
FP8 = TensorProto.FLOAT8E4M3FN
BF16 = TensorProto.BFLOAT16


def is_fp8(name):
    if name in inits:
        return inits[name].data_type == FP8
    n = producer.get(name)
    return n is not None and n.op_type == "TRT_FP4DynamicQuantize" and name == n.output[1]


def to_bf16(t):
    arr = numpy_helper.to_array(t).astype(np.float32).astype(ml_dtypes.bfloat16)
    new = numpy_helper.from_array(arr, t.name)
    assert new.data_type == BF16, new.data_type
    return new


changed = set()
for n in g.node:
    if n.op_type != "DequantizeLinear" or not is_fp8(n.input[0]):
        continue
    scale = n.input[1]
    if scale in changed:
        continue
    if scale in inits:
        t = inits[scale]
        assert t.data_type == TensorProto.FLOAT and int(np.prod(t.dims)) == 1, (scale, t.data_type, t.dims)
        t.CopyFrom(to_bf16(t))
    else:
        c = producer[scale]
        assert c.op_type == "Constant", (scale, c.op_type)
        users = [m for m in g.node if scale in m.input]
        assert len(users) == 1, f"{scale} shared by {len(users)} nodes"
        attr = next(a for a in c.attribute if a.name == "value")
        assert attr.t.data_type == TensorProto.FLOAT and int(np.prod(attr.t.dims)) <= 1
        attr.t.CopyFrom(to_bf16(attr.t))
    changed.add(scale)

# DQ outputs that are BF16 now: the second DQ of each chain (its scale is the first DQ's output).
bf16_dq = {n.output[0] for n in g.node if n.op_type == "DequantizeLinear" and n.input[1] in producer
           and producer[n.input[1]].op_type == "DequantizeLinear" and producer[n.input[1]].input[1] in changed}
casts = [n for n in g.node if n.op_type == "Cast" and n.input[0] in bf16_dq
         and next(a.i for a in n.attribute if a.name == "to") == BF16]
for c in casts:
    for m in g.node:
        for i, name in enumerate(m.input):
            if name == c.output[0]:
                m.input[i] = c.input[0]
    for o in g.output:
        assert o.name != c.output[0]
    g.node.remove(c)
del g.value_info[:]
onnx.save(model, dst)
print(f"{len(changed)} global scales to BF16, {len(bf16_dq)} DQ chains now BF16, {len(casts)} Casts bypassed -> {dst}")
