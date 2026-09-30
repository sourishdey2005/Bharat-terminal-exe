"""Export Chronos-Bolt Tiny to a valid ONNX, then int8-quantize it.

Replaces `export_chronos.py`, whose output ONNX Runtime cannot load. Three
separate defects had to be fixed, and each is verified rather than assumed.

1. `aten::nanmean` has no ONNX symbolic, so export aborted outright.
   `InstanceNorm.forward` is the only caller, and `nanmean` differs from `mean`
   solely in ignoring NaN. The exporter is fed a finite price window and the app
   only ever passes finite closes, so swapping in `mean` is exact and leaves no
   NaN-only branch to mistranslate.

2. `Patch.forward` uses `x.unfold(-1, size, step)` with `step == size`. That is
   numerically just a reshape into (..., n_patches, patch_size), but torch's ONNX
   exporter emits `unfold` transposed: (1, 16, 4) instead of (1, 4, 16). The
   downstream `Concat(context, mask)` then produced a last dimension of 8 where
   `input_patch_embedding.residual_layer` needs 32, and ORT failed with
   "Incompatible dimensions for matrix multiplication". Replaced with an explicit
   reshape.

3. Torch's legacy exporter writes `ReduceSum` `axes` as a node *attribute* (an
   opset-13+ input) and `Gather` indices as float32. Both are rejected by ORT.
   Repaired by `onnx_legalize.legalize`.

Verification chain, all of which must pass or the run aborts:
  a. patched torch model == original torch model   (patches 1 and 2 are exact)
  b. ONNX Runtime fp32    == original torch model
  c. ONNX Runtime int8    == original torch model
A graph that merely "loads" but computes something else fails (b).

Run:  python export_chronos_fixed.py
"""
import contextlib
import io
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
from chronos import ChronosBoltPipeline
from chronos.chronos_bolt import InstanceNorm, Patch

OUT_DIR = r"E:\Projects\bharat-terminal\bharat-terminal\models"
FP32 = os.path.join(OUT_DIR, "chronos_bolt_tiny.onnx")
INT8 = os.path.join(OUT_DIR, "chronos_bolt_tiny_int8.onnx")
LOG = r"E:\Projects\bharat-terminal\bharat-terminal\output\export_chronos.log"
OPSET = 18
CTX = 64


def log(msg):
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(str(msg) + "\n")


def patch_instance_norm():
    """`nanmean` -> `mean`: exact for the finite input this path always gets."""
    eps = 1e-5

    def forward(self, x, loc_scale=None):
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if loc_scale is None:
            loc = torch.nan_to_num(x.mean(dim=-1, keepdim=True), nan=0.0)
            scale = torch.nan_to_num((x - loc).square().mean(dim=-1, keepdim=True).sqrt(), nan=1.0)
            scale = torch.where(scale == 0, torch.tensor(eps, dtype=x.dtype), scale)
        else:
            loc, scale = loc_scale
        scaled_x = (x - loc) / scale
        if self.use_arcsinh:
            scaled_x = torch.arcsinh(scaled_x)
        return scaled_x.to(orig_dtype), (loc, scale)

    InstanceNorm.forward = forward


def patch_patch():
    """`unfold(step == size)` -> reshape, which the exporter lays out correctly.

    Also materializes with an explicit NaN fill so the padding branch survives
    tracing instead of depending on a data-dependent Python conditional.
    """
    def forward(self, x):
        length = x.shape[-1]
        if length % self.patch_size != 0:
            pad = self.patch_size - (length % self.patch_size)
            padding = torch.full(
                (*x.shape[:-1], pad), float("nan"), dtype=x.dtype, device=x.device
            )
            x = torch.concat((padding, x), dim=-1)
        return x.reshape(*x.shape[:-1], -1, self.patch_size)

    Patch.forward = forward


def forecast(model, context):
    """Median quantile path. `ChronosBoltOutput` carries `quantile_preds`."""
    with torch.no_grad():
        out = model(context)
    preds = getattr(out, "quantile_preds", None)
    if preds is None:
        preds = getattr(out, "forecast", out[0] if isinstance(out, (tuple, list)) else out)
    preds = preds[0] if preds.dim() == 3 else preds
    # Take the median across the quantile axis; index 4 of 9.
    return preds[preds.shape[0] // 2].numpy()


def main():
    open(LOG, "w", encoding="utf-8").close()
    torch.manual_seed(0)
    torch.set_num_threads(1)

    log("Loading amazon/chronos-bolt-tiny (real pretrained checkpoint)")
    pipeline = ChronosBoltPipeline.from_pretrained(
        "amazon/chronos-bolt-tiny", device_map="cpu", dtype=torch.float32
    )
    model = pipeline.model.eval()

    # Trending plus mild noise: a constant window could let a wrong graph pass.
    base = np.linspace(2400.0, 2600.0, CTX, dtype=np.float32)
    probe = (base + 4.0 * np.sin(np.arange(CTX) * 0.6)).astype(np.float32)
    context = torch.from_numpy(probe.reshape(1, CTX))

    # (a) Ground truth from the untouched model.
    ref = forecast(model, context)
    scale = max(float(np.abs(ref).max()), 1e-9)
    log(f"reference (untouched torch) {ref.shape}")

    patch_instance_norm()
    patch_patch()
    patched = forecast(model, context)
    rel_patch = float(np.abs(patched - ref).max()) / scale
    log(f"(a) patched torch vs untouched: rel = {rel_patch:.3e}")
    if rel_patch > 1e-6:
        log("ABORT monkey-patches changed model output")
        return 1

    log(f"exporting fp32 at opset {OPSET}")
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            torch.onnx.export(
                model,
                (context,),
                FP32,
                input_names=["context"],
                output_names=["forecast"],
                opset_version=OPSET,
                dynamo=False,
            )
    except Exception as exc:  # noqa: BLE001
        log("EXPORT FAILED: " + str(exc)[:400].replace("\n", " | "))
        return 1
    log(f"wrote {os.path.basename(FP32)} ({os.path.getsize(FP32)/1e6:.2f} MB)")

    from onnx_legalize import legalize

    graph = onnx.load(FP32, load_external_data=False)
    report = legalize(graph)
    log("legalize: " + report.summary())
    if report.skipped_fractional:
        log("ABORT non-integral Gather indices present")
        return 1
    onnx.save(graph, FP32)

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.enable_mem_pattern = False

    try:
        sess = ort.InferenceSession(FP32, so, providers=["CPUExecutionProvider"])
        got = sess.run(None, {"context": probe.reshape(1, CTX)})[0]
    except Exception as exc:  # noqa: BLE001
        log("ORT LOAD/RUN FAILED: " + str(exc)[:400].replace("\n", " | "))
        return 1

    if got.ndim != 3:
        log(f"ABORT expected [batch, quantiles, horizon], got {got.shape}")
        return 1
    q, h = got.shape[1], got.shape[2]
    log(f"quantiles={q} horizon={h}")
    if q != 9 or h != 64:
        log(f"ABORT unexpected output shape {got.shape}")
        return 1

    # Compare medians: `ref` is the torch median path, the ONNX head emits all 9.
    median = got[0, q // 2]
    if median.shape != ref.shape:
        log(f"ABORT median shape mismatch: onnx {median.shape} vs torch {ref.shape}")
        return 1
    rel = float(np.abs(median - ref).max()) / scale
    log(f"(b) onnx fp32 median vs torch: rel = {rel:.3e}")
    if rel > 1e-3:
        log("ABORT fp32 export does not match the PyTorch reference")
        return 1
    log(f"    median head={np.round(median[:5], 3).tolist()}")

    log("quantizing to int8 (dynamic, per-channel)")
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=FP32,
        model_output=INT8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        extra_options={"MatMulConstBOnly": True},
    )

    sess8 = ort.InferenceSession(INT8, so, providers=["CPUExecutionProvider"])
    got8 = sess8.run(None, {"context": probe.reshape(1, CTX)})[0]
    if got8.shape != got.shape:
        log(f"ABORT int8 shape {got8.shape} != fp32 {got.shape}")
        return 1
    if not np.isfinite(got8).all():
        log("ABORT int8 output contains non-finite values")
        return 1
    median8 = got8[0, q // 2]
    rel8 = float(np.abs(median8 - ref).max()) / scale
    log(f"(c) onnx int8 {os.path.getsize(INT8)/1e6:.2f} MB vs torch rel = {rel8:.3e}")
    if rel8 > 0.05:
        log("ABORT int8 deviates too far from the reference")
        return 1

    log("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
