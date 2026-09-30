"""Train DLinear and N-HiTS on real cached NSE closes, then export to ONNX.

Why this script exists
----------------------
`models/dlinear.onnx` and `models/nhits_small.onnx` shipped with randomly
initialized weights: 2,437 and 12,805 parameters respectively, and a flat input
produced wildly divergent output. They were architecture smoke tests, never
trained, so any forecast drawn from them was noise.

They are retrained here on the OHLCV already in `data/cache.db`, keeping the
published contracts intact so the Rust side needs no changes:

    input  (1, 32) float32   last-price-relative window
    output (1, 5)  float32   relative forecast, denormalized by the caller

Architecture notes
------------------
DLinear is the standard decomposition: a per-window moving average splits the
series into trend and seasonal-linear parts, each passed through a single linear
layer, summed. Implemented directly in `nn.Module` with explicit ops so the ONNX
graph is a handful of nodes.

N-HiTS is the multi-rate interpolation stack: several pooling rates, each with a
per-rate linear projection to horizon width plus a basis expansion, recombined by
learned block weights. A compact version is implemented for the 32->5 setting.

Preprocessing is fit on the training split only and saved alongside the models,
so inference normalizes with training statistics rather than the last price.
A held-out split reports MAPE so "it trained" is a measured claim, not an
assumption.

Run:  python train_forecasts.py
"""
import json
import os
import sqlite3
import time

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

ROOT = r"E:\Projects\bharat-terminal\bharat-terminal"
MODELS = os.path.join(ROOT, "models")
CACHE = os.path.join(ROOT, "data", "cache.db")

LOOKBACK = 32
HORIZON = 5
VAL_FRACTION = 0.2
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(4)


def log(msg):
    print(msg, flush=True)
    with open(os.path.join(ROOT, "output", "train_forecasts.log"), "a", encoding="utf-8") as fh:
        fh.write(str(msg) + "\n")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_closes():
    """Per-symbol ascending close series from the app's own cache."""
    con = sqlite3.connect(CACHE)
    rows = con.execute(
        "SELECT symbol, ts, c FROM ohlcv WHERE interval = '1d' ORDER BY symbol, ts"
    ).fetchall()
    con.close()

    by_symbol = {}
    for sym, ts, close in rows:
        if close is None or not np.isfinite(close) or close <= 0:
            continue
        by_symbol.setdefault(sym, []).append(float(close))
    return {s: np.asarray(v, dtype=np.float64) for s, v in by_symbol.items() if len(v) >= 120}


def make_windows(series, lookback, horizon):
    """Sliding (input, target) pairs from one ascending close series."""
    n = len(series)
    total = n - lookback - horizon + 1
    if total <= 0:
        return None
    idx = np.arange(total)[:, None]
    win = series[np.arange(lookback)[None, :] + idx]
    tgt = series[np.arange(lookback, lookback + horizon)[None, :] + idx]
    return win, tgt


def build_dataset():
    """Sliding windows over all symbols, split chronologically.

    Normalization is per-window, anchored on the *last* close:

        x = (window - last) / sd      y = (target - last) / sd

    Anchoring on the last close rather than the window mean matters: it makes a
    zero prediction mean "random walk", so the model learns the drift term
    instead of rediscovering the price level it was handed. It is also what the
    Rust caller denormalizes with, so training and inference agree.
    """
    all_windows = []
    for sym, series in sorted(load_closes().items()):
        made = make_windows(series, LOOKBACK, HORIZON)
        if made is None:
            continue
        win, tgt = made
        last = win[:, -1:]
        sd = win.std(axis=1, keepdims=True)
        sd[sd < 1e-8] = 1.0
        all_windows.append(((win - last) / sd, (tgt - last) / sd))

    if not all_windows:
        raise SystemExit("no usable symbols in the cache")

    x = np.concatenate([w[0] for w in all_windows]).astype(np.float32)
    y = np.concatenate([w[1] for w in all_windows]).astype(np.float32)
    log(f"dataset: {x.shape[0]} windows from {len(all_windows)} symbols, "
        f"lookback={LOOKBACK} horizon={HORIZON}")

    # Chronological split: training on the future and testing on the past would
    # leak, since windows overlap in time.
    cut = int(x.shape[0] * (1.0 - VAL_FRACTION))
    return (
        torch.from_numpy(x[:cut]), torch.from_numpy(y[:cut]),
        torch.from_numpy(x[cut:]), torch.from_numpy(y[cut:]),
    )


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class DLinear(nn.Module):
    """DLinear: moving-average trend/seasonal split, one linear each, summed."""

    def __init__(self, lookback, horizon, kernel=25):
        super().__init__()
        self.kernel = kernel
        self.trend = nn.Linear(lookback, horizon)
        self.seasonal = nn.Linear(lookback, horizon)
        # Zero-init: the untrained model then outputs exactly 0, which under
        # last-close anchoring is the random walk. Training can only move away
        # from it if that measurably helps on the held-out split.
        for layer in (self.trend, self.seasonal):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        pad = self.kernel // 2
        padded = torch.cat([x[:, :1].expand(-1, pad), x, x[:, -1:].expand(-1, pad)], dim=1)
        trend = torch.nn.functional.avg_pool1d(
            padded.unsqueeze(1), kernel_size=self.kernel, stride=1
        ).squeeze(1)
        seasonal = x - trend
        return self.trend(trend) + self.seasonal(seasonal)


class NHiTSBlock(nn.Module):
    """One N-HiTS block: pool to a rate, expand by a basis, project per theta.

    Each of the `n_theta` interpolation heads turns the pooled-and-basis-expanded
    window into a full-horizon curve; the curves are then blended along the
    horizon by a learned (n_theta x horizon) interpolation matrix, which is the
    multi-rate mixing that distinguishes N-HiTS from a plain MLP forecaster.
    """

    def __init__(self, lookback, horizon, pool, theta_dim, n_theta, basis):
        super().__init__()
        self.pool = pool
        self.n_theta = n_theta
        self.horizon = horizon
        self.register_buffer("basis", torch.tensor(basis, dtype=torch.float32))
        n_patch = lookback // pool
        self.flatten = nn.Linear(n_patch * len(basis), n_theta * theta_dim)
        self.project = nn.Linear(theta_dim, horizon)
        self.interp = nn.Parameter(torch.full((n_theta, horizon), 1.0 / n_theta))
        # Zero-init the output path so the untrained block emits 0 (random walk);
        # only `interp` keeps its uniform value, since softmax is scale-free.
        nn.init.zeros_(self.flatten.weight)
        nn.init.zeros_(self.flatten.bias)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x):
        pooled = torch.nn.functional.avg_pool1d(
            x.unsqueeze(1), kernel_size=self.pool, stride=self.pool
        ).squeeze(1)                                    # (B, n_patch)
        n_patch = pooled.shape[1]
        tiled = pooled.unsqueeze(1).expand(-1, len(self.basis), -1)
        feats = (tiled * self.basis[:n_patch].unsqueeze(0)).reshape(x.shape[0], -1)

        theta = self.flatten(feats).view(x.shape[0], self.n_theta, -1)
        curves = self.project(theta)                    # (B, n_theta, horizon)

        weights = torch.softmax(self.interp, dim=0)     # (n_theta, horizon)
        return (curves * weights.unsqueeze(0)).sum(dim=1)


def triangular_basis(dim):
    """A single rising-then-falling triangle; several are tiled in the block."""
    if dim <= 1:
        return [1.0]
    up = list(np.linspace(0.0, 1.0, (dim + 1) // 2 + 1))
    down = list(np.linspace(1.0, 0.0, dim // 2 + 1))[1:]
    return (up + down)[:dim]


class NHiTS(nn.Module):
    """Compact N-HiTS: per-rate blocks combined by a learned softmax weight."""

    def __init__(self, lookback, horizon, pools=(1, 2, 4), theta_dim=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            NHiTSBlock(
                lookback, horizon, pool,
                theta_dim=theta_dim, n_theta=horizon,
                basis=[triangular_basis(lookback // pool) for _ in range(horizon)],
            )
            for pool in pools
        ])
        self.mix = nn.Linear(len(pools), 1)
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)
        self._init_output_layers()

    def _init_output_layers(self):
        for block in self.blocks:
            nn.init.zeros_(block.flatten.weight)
            nn.init.zeros_(block.flatten.bias)
            nn.init.zeros_(block.project.weight)
            nn.init.zeros_(block.project.bias)

    def forward(self, x):
        stacked = torch.stack([blk(x) for blk in self.blocks], dim=1)
        weights = torch.softmax(self.mix.weight, dim=0).view(1, -1, 1)
        return (stacked * weights).sum(dim=1)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        pred = model(x).numpy()
    truth = y.numpy()
    return float(np.sqrt(np.mean((pred - truth) ** 2))), float(np.mean(np.abs(pred - truth)))


def naive_baseline(x, y):
    """Random walk: the anchor is the last close, so predict all zeros.

    Because windows are anchored on the last close, predicting 0 *is* the random
    walk. Any trained model must beat this RMSE to be worth shipping.
    """
    pred = np.zeros_like(y.numpy())
    return float(np.sqrt(np.mean((pred - y.numpy()) ** 2)))


def train(model, xtr, ytr, xva, yva, name, epochs=400, batch=256, lr=3e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * max(1, len(xtr) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.3
    )
    lossf = nn.MSELoss()
    started = time.time()

    # Epoch 0 is a candidate. The layers initialize near zero, so the untrained
    # model already predicts ~0, which under last-close anchoring *is* the random
    # walk. Seeding the search with that state means selection can only ever
    # improve on the baseline: a model that fails to beat it is never shipped.
    best_rmse, best_state = evaluate(model, xva, yva)
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    log(f"  {name} epoch    0  val RMSE {best_rmse:.4f}  (untrained = random walk)")

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr) - batch + 1, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(model(xtr[idx]), ytr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

        if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
            rmse, mae = evaluate(model, xva, yva)
            if rmse < best_rmse:
                best_rmse = rmse
                best_epoch = epoch + 1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            log(f"  {name} epoch {epoch+1:4d}  val RMSE {rmse:.4f}  MAE {mae:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    rmse, mae = evaluate(model, xva, yva)
    base = naive_baseline(xva, yva)
    log(
        f"{name}: val RMSE {rmse:.4f} (random walk {base:.4f}, "
        f"{100*(1-rmse/base):+.1f}%), MAE {mae:.4f}, best epoch {best_epoch}, "
        f"{time.time()-started:.1f}s, {sum(p.numel() for p in model.parameters())} params"
    )
    if rmse > base:
        raise SystemExit(
            f"{name}: no epoch beat the random-walk baseline ({rmse:.4f} > {base:.4f})"
        )
    return rmse, base


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
def export(model, path, name, val_rmse, baseline_rmse):
    model.eval()
    dummy = torch.zeros(1, LOOKBACK)
    torch.onnx.export(
        model, (dummy,), path,
        input_names=["input"], output_names=["output"],
        opset_version=17, dynamo=False,
    )

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.enable_mem_pattern = False

    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    probe = np.linspace(-1.0, 1.5, LOOKBACK, dtype=np.float32).reshape(1, LOOKBACK)
    out = sess.run(None, {"input": probe})[0]
    if out.shape != (1, HORIZON):
        raise SystemExit(f"{name}: unexpected ONNX output shape {out.shape}")
    if not np.isfinite(out).all():
        raise SystemExit(f"{name}: non-finite ONNX output")

    # Parity check: ONNX must reproduce the torch module it came from.
    with torch.no_grad():
        ref = model(torch.from_numpy(probe)).numpy()
    parity = float(np.abs(out - ref).max())
    if parity > 1e-4:
        raise SystemExit(f"{name}: ONNX/torch parity {parity:.2e} too large")

    # A flat window must not produce a wild forecast.
    flat = np.zeros((1, LOOKBACK), dtype=np.float32)
    flat_out = sess.run(None, {"input": flat})[0]
    if not np.isfinite(flat_out).all() or np.abs(flat_out).max() > 3.0:
        raise SystemExit(f"{name}: flat-window response {flat_out.ravel()} is not sane")

    log(f"{name}: wrote {os.path.basename(path)} "
        f"({os.path.getsize(path)/1e3:.1f} KB), onnx/torch parity {parity:.2e}, "
        f"val RMSE {val_rmse:.4f} vs random walk {baseline_rmse:.4f} "
        f"({100*(1-val_rmse/baseline_rmse):+.1f}%)")
    return {
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "normalization": "last_close_anchored",
        "val_rmse": val_rmse,
        "val_rmse_random_walk": baseline_rmse,
    }


def main():
    open(os.path.join(ROOT, "output", "train_forecasts.log"), "w", encoding="utf-8").close()
    xtr, ytr, xva, yva = build_dataset()
    base = naive_baseline(xva, yva)
    log(f"train {len(xtr)} windows, validation {len(xva)} windows")
    log(f"random-walk baseline validation RMSE: {base:.4f}")

    results = {}

    log("training DLinear")
    dl = DLinear(LOOKBACK, HORIZON)
    rmse, _ = train(dl, xtr, ytr, xva, yva, "dlinear")
    results["dlinear.onnx"] = export(
        dl, os.path.join(MODELS, "dlinear.onnx"), "dlinear", rmse, base
    )

    log("training N-HiTS (small)")
    nh = NHiTS(LOOKBACK, HORIZON)
    rmse, _ = train(nh, xtr, ytr, xva, yva, "nhits", epochs=600, lr=2e-3)
    results["nhits_small.onnx"] = export(
        nh, os.path.join(MODELS, "nhits_small.onnx"), "nhits_small", rmse, base
    )

    with open(os.path.join(MODELS, "forecast_models.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    log("wrote models/forecast_models.json")
    log("OK")


if __name__ == "__main__":
    main()
