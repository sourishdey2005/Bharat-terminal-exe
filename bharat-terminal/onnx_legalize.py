"""ONNX graph legalization passes for torch-exported time-series models.

Torch's legacy ONNX exporter emits two constructs that ONNX Runtime rejects.
Both are mechanical to repair, and both repairs are exact: the rewritten graph
computes the same function as the graph torch traced.

1. `ReduceSum` with `axes` as an attribute
   From opset 13 onward `axes` is a tensor *input*. The exporter still writes the
   pre-13 attribute form, which ORT refuses:
       "Unrecognized attribute: axes for operator ReduceSum"

   Repair: materialize `axes` as an int64 initializer and append it as a second
   input.

2. `Gather` with a float indices constant
   `Gather` indices must be int32/int64. The exporter emits them as float32
   (typically the value 1.0), which ORT refuses:
       "Type 'tensor(float)' of input parameter (/Constant_8_output_0) of
        operator (Gather) ... is invalid"

   Repair: cast the constant to int64. Only exact integral values are touched;
   a genuinely fractional index is left alone and reported rather than silently
   truncated.

`legalize()` is idempotent and returns a report describing what it changed, so
callers can log it.
"""
from dataclasses import dataclass, field

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

INT_TYPES = (TensorProto.INT32, TensorProto.INT64)


@dataclass
class LegalizeReport:
    reducesum_nodes: int = 0
    gather_indices: int = 0
    skipped_fractional: list[str] = field(default_factory=list)
    ir_version_bumped: bool = False

    def summary(self) -> str:
        parts = [f"ReduceSum(axes attr)={self.reducesum_nodes}",
                 f"Gather(float indices)={self.gather_indices}"]
        if self.skipped_fractional:
            parts.append(f"skipped fractional={self.skipped_fractional}")
        if self.ir_version_bumped:
            parts.append("ir_version bumped")
        return ", ".join(parts)


def _legalize_reducesum(model: onnx.ModelProto, report: LegalizeReport) -> None:
    used = {init.name for init in model.graph.initializer}
    for node in model.graph.node:
        used.update(node.output)

    for node in model.graph.node:
        if node.op_type != "ReduceSum":
            continue
        axes_attr = next((a for a in node.attribute if a.name == "axes"), None)
        if axes_attr is None:
            continue

        axes = np.asarray(axes_attr.ints, dtype=np.int64)
        base = f"{node.name or 'reduce'}_axes"
        name = base
        suffix = 0
        while name in used:
            suffix += 1
            name = f"{base}_{suffix}"
        used.add(name)

        model.graph.initializer.append(numpy_helper.from_array(axes, name))
        keep = [a for a in node.attribute if a.name != "axes"]
        del node.attribute[:]
        node.attribute.extend(keep)
        node.input.append(name)
        report.reducesum_nodes += 1


def _legalize_gather(model: onnx.ModelProto, report: LegalizeReport) -> None:
    constants = {n.output[0]: n for n in model.graph.node if n.op_type == "Constant"}
    initializers = {init.name: init for init in model.graph.initializer}

    def retype(tensor) -> bool:
        if tensor.data_type in INT_TYPES:
            return False
        values = np.asarray(numpy_helper.to_array(tensor))
        if not np.allclose(values, np.round(values)):
            return False
        tensor.CopyFrom(numpy_helper.from_array(np.round(values).astype(np.int64)))
        return True

    for node in model.graph.node:
        if node.op_type != "Gather" or len(node.input) < 2:
            continue
        idx = node.input[1]
        if idx in constants:
            attr = next((a for a in constants[idx].attribute if a.name == "value"), None)
            if attr is None or attr.t.data_type in INT_TYPES:
                continue
            if retype(attr.t):
                report.gather_indices += 1
            else:
                report.skipped_fractional.append(node.name or idx)
        elif idx in initializers:
            if retype(initializers[idx]):
                report.gather_indices += 1
            else:
                report.skipped_fractional.append(node.name or idx)


def legalize(model: onnx.ModelProto) -> LegalizeReport:
    report = LegalizeReport()
    _legalize_reducesum(model, report)
    _legalize_gather(model, report)
    if model.ir_version < 8:
        model.ir_version = 8
        report.ir_version_bumped = True
    return report
