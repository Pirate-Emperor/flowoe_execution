#!/usr/bin/env python3
"""Compare RK4, fixed Euler and fused Euler on the same held-out L2 windows."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from flowoe_execution.data import load_l2_csv, load_trades_csv
from flowoe_execution.execution import ExecutionSimulator, make_schedule_from_trajectory
from flowoe_execution.features import l2_features
from flowoe_execution.metrics import paired_block_bootstrap_ci, summary_stats
from flowoe_execution.model import CFMPolicy, FixedStepCryptoSampler
from flowoe_execution.ode import integrate_ode
from flowoe_execution.provenance import source_revision, utc_now, workflow_revision

ROOT = Path(__file__).resolve().parents[1]
CTX_LEN = 32
HORIZON = 8


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def compare_schedules(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    """Numerically compare normalized schedule tensors of shape (batch, horizon)."""
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("schedule tensors must have identical (batch, horizon) shapes")
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise ValueError("schedule tensors must be finite")
    p = reference.clamp_min(1e-12)
    q = candidate.clamp_min(1e-12)
    return {
        "mean_abs_error": float((p - q).abs().mean().item()),
        "max_abs_error": float((p - q).abs().max().item()),
        "mean_kl_reference_to_candidate": float((p * (p.log() - q.log())).sum(1).mean().item()),
    }


@torch.no_grad()
def rk4_from_initial_noise(model: CFMPolicy, context: torch.Tensor, x0: torch.Tensor, steps: int):
    """Evaluate the model's RK4 path using caller-provided initial noise."""
    encoded = model.context(context, "crypto")

    def field(t_scalar, state):
        t = torch.as_tensor(t_scalar, device=state.device, dtype=state.dtype)
        return model.vf(t, state, encoded)

    state = integrate_ode(field, x0.clone(), t0=0.0, t1=1.0, steps=steps)
    return torch.softmax(state, dim=1)


def summarize_paired_method(baseline, slippage, completion, schedule_errors, seed=7):
    """Summarize a method only against the same episodes' TWAP baselines."""
    baseline_arr = np.asarray(baseline, dtype=np.float64)
    slippage_arr = np.asarray(slippage, dtype=np.float64)
    completion_arr = np.asarray(completion, dtype=np.float64)
    if baseline_arr.ndim != 1 or slippage_arr.ndim != 1 or len(baseline_arr) == 0:
        raise ValueError("paired samples must be non-empty one-dimensional arrays")
    if len(baseline_arr) != len(slippage_arr) or len(completion_arr) != len(slippage_arr):
        raise ValueError("baseline, slippage and completion must contain the same episodes")
    if not np.isfinite(baseline_arr).all() or not np.isfinite(slippage_arr).all() or not np.isfinite(completion_arr).all():
        raise ValueError("paired execution samples must be finite")
    mean, ci = paired_block_bootstrap_ci(
        baseline_arr, slippage_arr, seed=seed, n_boot=2000, block_size=5
    )
    errors = {
        key: float(np.mean([row[key] for row in schedule_errors]))
        for key in ("mean_abs_error", "max_abs_error", "mean_kl_reference_to_candidate")
    }
    errors["mean_episode_max_abs_error"] = errors.pop("max_abs_error")
    return {
        "slippage_bps": summary_stats(slippage_arr, seed=seed, n_boot=2000),
        "completion_mean": float(completion_arr.mean()),
        "episodes": int(len(slippage_arr)),
        "baseline_episodes": int(len(baseline_arr)),
        "improvement_vs_twap_bps": float(baseline_arr.mean() - slippage_arr.mean()),
        "paired_improvement_vs_twap_bps": mean,
        "paired_improvement_ci95_bps": ci,
        "schedule_error_vs_rk4": errors,
    }


def resolve_input(arg: str | None, patterns: list[str], label: str) -> Path:
    if arg:
        path = Path(arg)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise SystemExit(f"{label} file not found: {path}")
        return path
    matches = sorted(p for pattern in patterns for p in ROOT.glob(pattern))
    if not matches:
        raise SystemExit(f"No {label} file found under {ROOT}")
    return matches[0]


def initial_state_for_mode(x0: torch.Tensor, mode: str) -> torch.Tensor:
    """Choose the initial state for a controlled latent ablation."""
    if mode == "random":
        return x0.clone()
    if mode == "zero":
        return torch.zeros_like(x0)
    raise ValueError(f"unsupported latent mode: {mode}")


def ridge_descriptor(context: np.ndarray) -> np.ndarray:
    """Compact causal descriptor: latest L2 features plus recent 8-row mean."""
    x = np.asarray(context, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 8 or not np.isfinite(x).all():
        raise ValueError("context must be a finite (T,D) array with T >= 8")
    return np.concatenate([x[-1], x[-8:].mean(axis=0)])


def fit_ridge_forecaster(
    features: np.ndarray,
    mids: np.ndarray,
    train_end: int,
    ctx_len: int = CTX_LEN,
    horizon: int = HORIZON,
    stride: int = 8,
    regularization: float = 10.0,
) -> dict:
    """Fit a multi-horizon price-change ridge model using training rows only.

    Every target horizon ends before train_end. Evaluation-period rows cannot
    change the fit, feature means, or scaling.
    """
    x = np.asarray(features, dtype=np.float64)
    p = np.asarray(mids, dtype=np.float64)
    if x.ndim != 2 or p.ndim != 1 or len(x) != len(p):
        raise ValueError("features and mids must be aligned 2-D/1-D arrays")
    if not 0 < train_end <= len(x) or ctx_len < 8 or horizon < 1 or stride < 1:
        raise ValueError("invalid train boundary or forecast dimensions")
    if not np.isfinite(x[:train_end]).all() or not np.isfinite(p[:train_end]).all() or (p[:train_end] <= 0).any():
        raise ValueError("training data must be finite with positive mid prices")
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("regularization must be finite and positive")
    xs, ys = [], []
    for start in range(0, train_end - ctx_len - horizon + 1, stride):
        cut = start + ctx_len
        xs.append(ridge_descriptor(x[start:cut]))
        ys.append(np.log(p[cut:cut + horizon] / p[cut - 1]) * 1e4)
    if len(xs) < 2:
        raise ValueError("not enough training windows for ridge forecaster")
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    x_mean, x_std = xs.mean(axis=0), xs.std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    z = (xs - x_mean) / x_std
    y_mean = ys.mean(axis=0)
    gram = z.T @ z
    beta = np.linalg.solve(gram + regularization * np.eye(gram.shape[0]), z.T @ (ys - y_mean))
    return {
        "feature_mean": x_mean,
        "feature_std": x_std,
        "target_mean_bps": y_mean,
        "coefficients": beta,
        "train_windows": int(len(xs)),
        "train_end": int(train_end),
        "context_length": int(ctx_len),
        "horizon": int(horizon),
        "stride": int(stride),
        "regularization": float(regularization),
    }


def predict_ridge_schedule(model: dict, context: np.ndarray, side: str, strength: float) -> tuple[np.ndarray, np.ndarray]:
    """Convert causal ridge price forecasts into a normalized child-order schedule."""
    if side not in {"buy", "sell"} or not np.isfinite(strength) or strength <= 0:
        raise ValueError("side must be buy/sell and strength must be finite and positive")
    desc = ridge_descriptor(context)
    pred_bps = ((desc - model["feature_mean"]) / model["feature_std"]) @ model["coefficients"]
    pred_bps = pred_bps + model["target_mean_bps"]
    sign = 1.0 if side == "buy" else -1.0
    logits = -sign * strength * pred_bps
    logits -= np.max(logits)
    weights = np.exp(np.clip(logits, -60.0, 0.0))
    schedule = weights / weights.sum()
    if not np.isfinite(schedule).all() or np.any(schedule < 0) or not np.isclose(schedule.sum(), 1.0):
        raise RuntimeError("ridge schedule normalization failed")
    return pred_bps, schedule


def blend_schedules(cfm: np.ndarray, ridge: np.ndarray, ridge_weight: float) -> np.ndarray:
    """Geometrically blend two nonnegative execution distributions."""
    p = np.asarray(cfm, dtype=np.float64)
    q = np.asarray(ridge, dtype=np.float64)
    if p.ndim != 1 or q.shape != p.shape or p.size == 0:
        raise ValueError("schedules must be equal-length nonempty 1-D arrays")
    if not np.isfinite(p).all() or not np.isfinite(q).all() or np.any(p < 0) or np.any(q < 0):
        raise ValueError("schedules must be finite and non-negative")
    if not np.isfinite(ridge_weight) or not 0.0 <= ridge_weight <= 1.0:
        raise ValueError("ridge weight must be in [0,1]")
    p = np.maximum(p / max(p.sum(), 1e-300), 1e-12)
    q = np.maximum(q / max(q.sum(), 1e-300), 1e-12)
    logits = (1.0 - ridge_weight) * np.log(p) + ridge_weight * np.log(q)
    logits -= logits.max()
    result = np.exp(logits)
    result /= result.sum()
    return result


def temperature_schedule(schedule: np.ndarray, temperature: float) -> np.ndarray:
    """Sharpen/flatten a normalized execution schedule without changing its support."""
    p = np.asarray(schedule, dtype=np.float64)
    if p.ndim != 1 or p.size == 0 or not np.isfinite(p).all() or np.any(p < 0) or p.sum() <= 0:
        raise ValueError("schedule must be finite, non-negative, nonempty, and have positive mass")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    p = np.maximum(p / p.sum(), 1e-12)
    logits = np.log(p) / temperature
    logits -= logits.max()
    out = np.exp(logits)
    out /= out.sum()
    return out


def evaluate_one(
    model: CFMPolicy,
    sampler: FixedStepCryptoSampler,
    context: torch.Tensor,
    x0: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        return sampler(context, x0.clone())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--training-report", help="Optional JSON training report that records the FI-2010 provenance.")
    ap.add_argument("--data-source", choices=["public_futures", "user_supplied", "synthetic"], default="user_supplied",
                    help="Provenance label for the execution input files; use public_futures only for independently validated public Futures data.")
    ap.add_argument("--l2", help="Validated L2 CSV; defaults to the first real *_l2.csv file.")
    ap.add_argument("--trades", help="Validated trade CSV; defaults to the first real *_trades.csv file.")
    ap.add_argument("--side", choices=["buy", "sell"], required=True)
    ap.add_argument("--quantity", type=float, default=0.5)
    ap.add_argument("--train-ratio", type=float, default=0.60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rk4-steps", type=int, default=24)
    ap.add_argument("--euler-steps", default="2,3,4,6")
    ap.add_argument("--latent-modes", default="random", help="Comma-separated modes: random,zero")
    ap.add_argument("--temperatures", default="0.5,1,2", help="Positive schedule-temperature candidates")
    ap.add_argument("--dataset-label", default="unknown", help="Independent market window/date label")
    ap.add_argument("--include-fused", action="store_true")
    ap.add_argument("--fused-max-error", type=float, default=1e-4)
    ap.add_argument("--max-l2-rows", type=int, default=200000)
    ap.add_argument("--report", default="results/execution_grid.json")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if args.quantity <= 0 or not 0.1 <= args.train_ratio <= 0.9:
        ap.error("quantity must be positive and train-ratio must be in [0.1, 0.9]")
    if args.rk4_steps < 2 or args.max_l2_rows < 1 or args.fused_max_error <= 0:
        ap.error("rk4-steps >= 2, max-l2-rows >= 1 and fused-max-error > 0 are required")
    try:
        euler_steps = sorted(set(int(x) for x in args.euler_steps.split(",") if x.strip()))
    except ValueError:
        ap.error("--euler-steps must be comma-separated integers")
    if not euler_steps or min(euler_steps) < 2:
        ap.error("every Euler step count must be at least 2")
    latent_modes = list(dict.fromkeys(x.strip() for x in args.latent_modes.split(",") if x.strip()))
    if not latent_modes or any(x not in {"random", "zero"} for x in latent_modes):
        ap.error("latent-modes must be a comma-separated subset of random,zero")
    try:
        temperatures = sorted(set(float(x) for x in args.temperatures.split(",") if x.strip()))
    except ValueError:
        ap.error("temperatures must be comma-separated positive numbers")
    if not temperatures or any(not np.isfinite(x) or x <= 0 for x in temperatures):
        ap.error("temperatures must be finite and positive")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    l2_path = resolve_input(args.l2, ["data/real/crypto/*_l2.csv"], "L2")
    trade_path = resolve_input(args.trades, ["data/real/crypto/*_trades.csv"], "trade")

    l2 = load_l2_csv(l2_path, max_rows=args.max_l2_rows).snapshots
    trades = load_trades_csv(trade_path)
    features = l2_features(l2)
    if len(l2) <= CTX_LEN + HORIZON + 1 or not np.isfinite(features).all():
        raise SystemExit(f"Insufficient or invalid input: L2 rows={len(l2)}")
    if l2.ask0.le(l2.bid0).any():
        raise SystemExit("L2 contains crossed/locked top-of-book rows")
    start, end = l2.timestamp.min(), l2.timestamp.max()
    overlap = trades[(trades.timestamp >= start) & (trades.timestamp <= end)]
    if overlap.empty or overlap.timestamp.min() > start + (end - start) * 0.05 or overlap.timestamp.max() < end - (end - start) * 0.05:
        raise SystemExit("Trade prints do not cover the full L2 evaluation period")

    device = torch.device("cuda")
    model = CFMPolicy(8).to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    x_crypto = torch.from_numpy(features).to(device)
    ntrain = int(len(l2) * args.train_ratio)
    if ntrain <= CTX_LEN + HORIZON or ntrain >= len(l2) - HORIZON:
        raise SystemExit("Chronological split leaves no train/evaluation window")
    mids = ((l2.bid0 + l2.ask0) / 2).to_numpy(dtype=np.float64)
    ridge_model = fit_ridge_forecaster(features, mids, ntrain, ctx_len=CTX_LEN, horizon=HORIZON)
    ridge_strengths = (0.5, 1.0, 2.0, 4.0)
    simulator = ExecutionSimulator(quantity=args.quantity, side=args.side)
    baselines: list[float] = []
    base_completion: list[float] = []
    methods: dict[str, dict] = {}
    schedule_errors: dict[str, list[dict]] = {}
    samplers: dict[str, FixedStepCryptoSampler] = {}
    for steps in euler_steps:
        key = f"fixed_euler_{steps}"
        samplers[key] = FixedStepCryptoSampler(model, steps=steps).to(device).eval()
        schedule_errors[key] = []
    fused_error = None
    disabled_fused: set[str] = set()
    if args.include_fused:
        try:
            from torch.utils.cpp_extension import load
            ext = load(
                name="flowoe_execution_grid_fused",
                sources=[
                    str(ROOT / "src/flowoe_execution/cuda/fused_step.cpp"),
                    str(ROOT / "src/flowoe_execution/cuda/fused_step.cu"),
                ],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3"],
                verbose=False,
            )
            for steps in euler_steps:
                key = f"fused_euler_{steps}"
                samplers[key] = FixedStepCryptoSampler(
                    model, steps=steps, fused_step=ext.fused_euler_step
                ).to(device).eval()
                schedule_errors[key] = []
        except Exception as exc:
            fused_error = f"{type(exc).__name__}: {exc}"

    starts = list(range(ntrain, len(l2) - HORIZON, HORIZON * 3))
    if not starts:
        raise SystemExit("No held-out execution episodes were formed")
    episode_records = []
    for episode_idx, i in enumerate(starts):
        episode_record = {
            "episode_index": episode_idx,
            "l2_start": str(l2.iloc[i].timestamp),
            "l2_end": str(l2.iloc[i + HORIZON - 1].timestamp),
            "twap_slippage_bps": None,
            "twap_completion": None,
            "methods": {},
        }
        ctx = x_crypto[i - CTX_LEN:i].unsqueeze(0)
        # Reset the same per-episode latent for every method, making slippage
        # and schedule differences paired rather than noise-confounded.
        gen = torch.Generator(device=device)
        gen.manual_seed(args.seed + episode_idx)
        x_random = torch.randn((1, HORIZON), generator=gen, device=device)
        latent_states = {mode: initial_state_for_mode(x_random, mode) for mode in latent_modes}
        references = {
            mode: rk4_from_initial_noise(model, ctx, x_init, args.rk4_steps)
            for mode, x_init in latent_states.items()
        }
        window = l2.iloc[i:i + HORIZON]
        twap = simulator.run(
            window, np.full(HORIZON, 1.0 / HORIZON),
            benchmark="trade_vwap", trades=trades, strategy="twap",
        )
        baselines.append(twap.slippage_bps)
        base_completion.append(twap.completion)
        episode_record["twap_slippage_bps"] = twap.slippage_bps
        episode_record["twap_completion"] = twap.completion

        ridge_candidates = {}
        for strength in ridge_strengths:
            ridge_key = f"ridge_{strength:g}"
            pred_bps, schedule = predict_ridge_schedule(
                ridge_model, features[i - CTX_LEN:i], args.side, strength,
            )
            ridge_candidates[strength] = {"predicted_path_bps": pred_bps, "schedule": schedule}
            ridge_result = simulator.run(
                window, schedule, benchmark="trade_vwap", trades=trades,
                strategy=ridge_key,
            )
            methods.setdefault(ridge_key, {"baseline": [], "slippage": [], "completion": [], "schedule_errors": []})
            methods[ridge_key]["baseline"].append(twap.slippage_bps)
            methods[ridge_key]["slippage"].append(ridge_result.slippage_bps)
            methods[ridge_key]["completion"].append(ridge_result.completion)
            ridge_tensor = torch.as_tensor(schedule, dtype=references[latent_modes[0]].dtype, device=device).unsqueeze(0)
            methods[ridge_key]["schedule_errors"].append(compare_schedules(references[latent_modes[0]], ridge_tensor))
            episode_record["methods"][ridge_key] = {
                "slippage_bps": ridge_result.slippage_bps,
                "completion": ridge_result.completion,
                "predicted_path_bps": pred_bps.tolist(),
                "schedule": schedule.tolist(),
            }

        for mode in latent_modes:
            x_init = latent_states[mode]
            reference = references[mode]
            ref_schedule = reference[0].detach().cpu().numpy()
            suffix = "" if mode == "random" else f"_{mode}"
            rk4_key = f"rk4{suffix}"
            reference_result = simulator.run(
                window, ref_schedule, benchmark="trade_vwap",
                trades=trades, strategy=f"flowoe_{rk4_key}_{args.rk4_steps}",
            )
            methods.setdefault(rk4_key, {"baseline": [], "slippage": [], "completion": [], "schedule_errors": []})
            methods[rk4_key]["baseline"].append(twap.slippage_bps)
            methods[rk4_key]["slippage"].append(reference_result.slippage_bps)
            methods[rk4_key]["completion"].append(reference_result.completion)
            methods[rk4_key]["schedule_errors"].append({
                "mean_abs_error": 0.0, "max_abs_error": 0.0,
                "mean_kl_reference_to_candidate": 0.0,
            })
            episode_record["methods"][rk4_key] = {
                "slippage_bps": reference_result.slippage_bps,
                "completion": reference_result.completion,
                "schedule_error": {"mean_abs_error": 0.0, "max_abs_error": 0.0, "mean_kl_reference_to_candidate": 0.0},
            }

            for key, sampler in samplers.items():
                method_key = key if mode == "random" else f"{key}_{mode}"
                if method_key in disabled_fused:
                    continue
                pred = evaluate_one(model, sampler, ctx, x_init)
                raw_error = None
                if key.startswith("fused_euler_"):
                    ref_key = key.replace("fused_euler_", "fixed_euler_")
                    ref_pred = samplers[ref_key](ctx, x_init.clone())
                    raw_error = float((pred - ref_pred).abs().max().item())
                    if raw_error > args.fused_max_error:
                        fused_error = (
                            f"{method_key} max output error {raw_error:.8g} exceeded "
                            f"tolerance {args.fused_max_error:.8g} at episode {episode_idx}"
                        )
                        disabled_fused.add(method_key)
                        continue
                schedule = pred[0].detach().cpu().numpy()
                execution = simulator.run(
                    window, schedule, benchmark="trade_vwap", trades=trades,
                    strategy=method_key,
                )
                methods.setdefault(method_key, {"baseline": [], "slippage": [], "completion": [], "schedule_errors": []})
                methods[method_key]["baseline"].append(twap.slippage_bps)
                methods[method_key]["slippage"].append(execution.slippage_bps)
                methods[method_key]["completion"].append(execution.completion)
                err = compare_schedules(reference, pred)
                methods[method_key]["schedule_errors"].append(err)
                episode_record["methods"][method_key] = {
                    "slippage_bps": execution.slippage_bps,
                    "completion": execution.completion,
                    "schedule_error": err,
                }
                if raw_error is not None:
                    episode_record["methods"][method_key]["max_abs_error_vs_pytorch_euler"] = raw_error

                if key in {"fixed_euler_4", "fixed_euler_6"}:
                    for temp in temperatures:
                        if temp != 1.0:
                            temp_key = f"{method_key}_temp{temp:g}"
                            temp_schedule = temperature_schedule(schedule, temp)
                            temp_result = simulator.run(
                                window, temp_schedule, benchmark="trade_vwap",
                                trades=trades, strategy=temp_key,
                            )
                            methods.setdefault(temp_key, {"baseline": [], "slippage": [], "completion": [], "schedule_errors": []})
                            methods[temp_key]["baseline"].append(twap.slippage_bps)
                            methods[temp_key]["slippage"].append(temp_result.slippage_bps)
                            methods[temp_key]["completion"].append(temp_result.completion)
                            temp_tensor = torch.as_tensor(temp_schedule, dtype=reference.dtype, device=device).unsqueeze(0)
                            methods[temp_key]["schedule_errors"].append(compare_schedules(reference, temp_tensor))
                            episode_record["methods"][temp_key] = {
                                "slippage_bps": temp_result.slippage_bps,
                                "completion": temp_result.completion,
                                "schedule": temp_schedule.tolist(),
                                "temperature": temp,
                            }
                        ridge_schedule = ridge_candidates[2.0]["schedule"]
                        for gamma in (0.25, 0.5, 0.75):
                            blend = blend_schedules(temperature_schedule(schedule, temp), ridge_schedule, gamma)
                            if temp == 1.0:
                                hybrid_base = f"hybrid_{key}_ridge2_g{gamma:g}"
                            else:
                                hybrid_base = f"hybrid_{key}_temp{temp:g}_ridge2_g{gamma:g}"
                            hybrid_key = hybrid_base if mode == "random" else f"{hybrid_base}_zero"
                            hybrid_result = simulator.run(
                                window, blend, benchmark="trade_vwap", trades=trades,
                                strategy=hybrid_key,
                            )
                            methods.setdefault(hybrid_key, {"baseline": [], "slippage": [], "completion": [], "schedule_errors": []})
                            methods[hybrid_key]["baseline"].append(twap.slippage_bps)
                            methods[hybrid_key]["slippage"].append(hybrid_result.slippage_bps)
                            methods[hybrid_key]["completion"].append(hybrid_result.completion)
                            blend_tensor = torch.as_tensor(blend, dtype=reference.dtype, device=device).unsqueeze(0)
                            methods[hybrid_key]["schedule_errors"].append(compare_schedules(reference, blend_tensor))
                            episode_record["methods"][hybrid_key] = {
                                "slippage_bps": hybrid_result.slippage_bps,
                                "completion": hybrid_result.completion,
                                "schedule": blend.tolist(),
                                "temperature": temp,
                                "ridge_weight": gamma,
                                "max_abs_error_vs_pytorch_euler": None,
                            }
        episode_records.append(episode_record)

    summaries = {}
    for name, vals in methods.items():
        summaries[name] = summarize_paired_method(
            vals["baseline"], vals["slippage"], vals["completion"],
            vals["schedule_errors"], seed=args.seed,
        )

    training_report = None
    fi_provenance = None
    if args.training_report:
        report_path = Path(args.training_report)
        if not report_path.is_absolute():
            report_path = ROOT / report_path
        if not report_path.exists():
            raise SystemExit(f"Training report not found: {report_path}")
        training_report = {
            "path": str(report_path),
            "sha256": sha256(report_path),
        }
        try:
            source_report = json.loads(report_path.read_text(encoding="utf-8"))
            fi_provenance = {
                "fi_path": source_report.get("fi_path"),
                "fi_sha256": source_report.get("fi_sha256"),
                "fi_label_encoding_detected": source_report.get("fi_label_encoding_detected"),
                "training_seed": source_report.get("seed"),
                "training_windows": source_report.get("train_windows"),
                "fi_windows": source_report.get("fi_windows"),
            }
        except Exception:
            fi_provenance = {"parse_error": "could not parse training report"}

    result = {
        "created_utc": utc_now(),
        "source_commit": source_revision(ROOT),
        "workflow_commit": workflow_revision(),
        "status": "passed",
        "data_source": args.data_source,
        "instrument": "BTCUSDT",
        "venue": "Binance USD-M futures",
        "side": args.side,
        "quantity": args.quantity,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "rk4_steps": args.rk4_steps,
        "euler_steps": euler_steps,
        "latent_modes": latent_modes,
        "temperatures": temperatures,
        "dataset_label": args.dataset_label,
        "ridge_forecaster": {
            "train_windows": ridge_model["train_windows"],
            "train_end_index_exclusive": ridge_model["train_end"],
            "context_length": ridge_model["context_length"],
            "horizon": ridge_model["horizon"],
            "stride": ridge_model["stride"],
            "regularization": ridge_model["regularization"],
            "schedule_strengths": list(ridge_strengths),
            "hybrid_ridge_strength": 2.0,
            "hybrid_ridge_weights": [0.25, 0.5, 0.75],
            "fit_uses_evaluation_rows": False,
        },
        "context_len": CTX_LEN,
        "horizon": HORIZON,
        "evaluation_stride_snapshots": HORIZON * 3,
        "episodes": len(starts),
        "l2_rows": int(len(l2)),
        "trade_rows": int(len(overlap)),
        "l2_window_start": str(start),
        "l2_window_end": str(end),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "training_report": training_report,
        "fi_training_provenance": fi_provenance,
        "l2_path": str(l2_path),
        "l2_sha256": sha256(l2_path),
        "trades_path": str(trade_path),
        "trades_sha256": sha256(trade_path),
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "fused_extension_error": fused_error,
        "methods": summaries,
        "episode_results": episode_records,
        "note": "Execution-simulator slippage against market trade VWAP; no market-impact or real order-fill claim. Paired moving-block bootstrap uses 5 consecutive episodes per block. The data_source field is the caller-provided provenance label and must be supported by independent input validation.",
    }
    out = Path(args.report)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
