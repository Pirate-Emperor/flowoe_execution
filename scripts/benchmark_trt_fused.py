"""Benchmark the combined PyTorch context encoder + INT8 TensorRT + fused CUDA path."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from flowoe_execution.data import load_l2_csv
from flowoe_execution.features import l2_features
from flowoe_execution.model import CFMPolicy, FixedStepCryptoSampler
from flowoe_execution.provenance import source_revision, utc_now, workflow_revision
from scripts.benchmark_trt import output_error_metrics, stats
from scripts.build_trt_engine import accuracy_gate


ROOT = Path(__file__).resolve().parents[1]


class CombinedSampler(nn.Module):
    """INT8 TensorRT vector field calls with the Euler update fused in CUDA."""

    def __init__(self, policy, trt_vf, fused_step, steps: int):
        super().__init__()
        self.context = policy.context
        self.trt_vf = trt_vf
        self.fused_step = fused_step
        self.steps = int(steps)

    def forward(self, context: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        encoded = self.context.crypto_encode(context)
        state = x.clone().contiguous()
        dt = 1.0 / float(self.steps - 1)
        for i in range(self.steps - 1):
            t = state.new_full((state.shape[0],), float(i) * dt)
            velocity = self.trt_vf(t, state, encoded)
            self.fused_step(state, velocity.contiguous(), dt)
        return torch.softmax(state, dim=1)


@torch.no_grad()
def rk4_schedule(model: CFMPolicy, context: torch.Tensor, x0: torch.Tensor, steps: int = 64):
    encoded = model.context.crypto_encode(context)
    state = x0.clone()
    dt = 1.0 / float(steps)
    batch = state.shape[0]
    for i in range(steps):
        t0 = state.new_full((batch,), i * dt)
        th = state.new_full((batch,), (i + 0.5) * dt)
        t1 = state.new_full((batch,), (i + 1.0) * dt)
        k1 = model.vf(t0, state, encoded)
        k2 = model.vf(th, state + 0.5 * dt * k1, encoded)
        k3 = model.vf(th, state + 0.5 * dt * k2, encoded)
        k4 = model.vf(t1, state + dt * k3, encoded)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return torch.softmax(state, dim=1)


def timing(fn, runs: int, warmup: int):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        gpu, wall = [], []
        for _ in range(runs):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter()
            start.record()
            out = fn()
            end.record()
            torch.cuda.synchronize()
            wall.append((time.perf_counter() - t0) * 1000.0)
            gpu.append(start.elapsed_time(end))
    if out is None or not torch.isfinite(out).all():
        raise RuntimeError("inference path returned non-finite output")
    return {"cuda_event_ms": stats(gpu), "end_to_end_wall_ms": stats(wall)}, out


def resolve_data(checkpoint: Path):
    real = sorted((ROOT / "data/real/crypto").glob("*_l2.csv"))
    smoke = ROOT / "data/smoke/crypto_l2.csv"
    candidates = (real + ([smoke] if smoke.exists() else [])) if checkpoint.name == "cfm_policy.pt" else (
        ([smoke] if smoke.exists() else []) + real
    )
    for path in candidates:
        try:
            feats = l2_features(load_l2_csv(path, max_rows=50000).snapshots)
            if len(feats) >= 96:
                src = f"real crypto L2 ({path.stem})" if "data/real/" in str(path) else "synthetic smoke L2"
                return feats, src
        except Exception:
            continue
    raise SystemExit("No valid L2 input for TensorRT/fused benchmark")


def gate(max_path_error: float, max_allowed_error: float, mean_abs_rk4: float,
         max_mean_abs_rk4: float, mean_kl_rk4: float, max_mean_kl_rk4: float,
         batch1_p99_wall_ms: float | None, target_ms: float = 2.0) -> dict:
    vals = (max_path_error, max_allowed_error, mean_abs_rk4, max_mean_abs_rk4,
            mean_kl_rk4, max_mean_kl_rk4, target_ms)
    if not all(np.isfinite(v) for v in vals) or min(max_allowed_error, max_mean_abs_rk4, max_mean_kl_rk4, target_ms) <= 0:
        raise ValueError("gate thresholds and metrics must be finite, with positive limits")
    if batch1_p99_wall_ms is not None and (not np.isfinite(batch1_p99_wall_ms) or batch1_p99_wall_ms < 0):
        raise ValueError("batch-1 latency must be finite and non-negative")
    path_ok = max_path_error <= max_allowed_error
    quality_ok = mean_abs_rk4 <= max_mean_abs_rk4 and mean_kl_rk4 <= max_mean_kl_rk4
    latency_ok = batch1_p99_wall_ms is not None and batch1_p99_wall_ms < target_ms
    return {
        "path_accuracy_passed": bool(path_ok),
        "ode_quality_passed": bool(quality_ok),
        "latency_target_p99_ms": float(target_ms),
        "batch1_p99_latency_passed": bool(latency_ok),
        "sub_2ms_p99_and_quality_target_met": bool(path_ok and quality_ok and latency_ok),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="results/cfm_policy.pt")
    ap.add_argument("--engine", default="results/flowoe_vf_int8.pth")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--accuracy-cases", type=int, default=64)
    ap.add_argument("--accuracy-seed", type=int, default=20261011)
    ap.add_argument("--max-path-error", type=float, default=0.02)
    ap.add_argument("--max-mean-abs-rk4", type=float, default=0.015)
    ap.add_argument("--max-mean-kl-rk4", type=float, default=0.01)
    ap.add_argument("--output", default="results/tensorrt_fused_latency.json")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    if args.steps < 2 or args.runs < 1 or args.warmup < 0 or args.accuracy_cases < 1:
        ap.error("steps >= 2, runs >= 1, warmup >= 0, accuracy-cases >= 1 required")
    if min(args.max_path_error, args.max_mean_abs_rk4, args.max_mean_kl_rk4) <= 0:
        ap.error("quality thresholds must be positive")
    import tensorrt as trt
    from torch2trt_dynamic import TRTModule
    from torch.utils.cpp_extension import load

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    engine_path = Path(args.engine)
    if not engine_path.is_absolute():
        engine_path = ROOT / engine_path
    if not checkpoint.exists() or not engine_path.exists():
        raise SystemExit("trained checkpoint or vector-field TensorRT engine is missing")
    model = CFMPolicy(8).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    trt_vf = TRTModule()
    trt_vf.load_state_dict(torch.load(engine_path, map_location="cuda", weights_only=False))
    ext = load(
        name="flowoe_fused_trt_euler",
        sources=[
            str(ROOT / "src/flowoe_execution/cuda/fused_step.cpp"),
            str(ROOT / "src/flowoe_execution/cuda/fused_step.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    combined = CombinedSampler(model, trt_vf, ext.fused_euler_step, args.steps).cuda().eval()
    pytorch = FixedStepCryptoSampler(model, steps=args.steps).cuda().eval()
    features, source = resolve_data(checkpoint)
    if len(features) < 32 + args.accuracy_cases:
        starts = np.linspace(0, len(features) - 32, num=args.accuracy_cases, dtype=int)
    else:
        starts = np.linspace(0, len(features) - 32, num=args.accuracy_cases, dtype=int)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.accuracy_seed)
    contexts = torch.stack([
        torch.from_numpy(features[i:i + 32]) for i in starts
    ]).cuda()
    noises = torch.randn((len(starts), 8), generator=generator, device="cuda")
    accuracy = {}
    max_path_error = 0.0
    all_rk4, all_combined = [], []
    latency = {}
    for batch_size in (1, 4):
        refs, candidates, rk4s = [], [], []
        for offset in range(0, len(starts), batch_size):
            ctx = contexts[offset:offset + batch_size]
            x0 = noises[offset:offset + batch_size]
            ref_out = pytorch(ctx, x0.clone())
            trt_out = combined(ctx, x0.clone())
            err = float((ref_out - trt_out).abs().max().item())
            max_path_error = max(max_path_error, err)
            rk4 = rk4_schedule(model, ctx, x0, steps=64)
            refs.append(ref_out.detach())
            candidates.append(trt_out.detach())
            rk4s.append(rk4.detach())
        ref_all, candidate_all, rk4_all = torch.cat(refs), torch.cat(candidates), torch.cat(rk4s)
        accuracy[str(batch_size)] = {
            "cases": int(len(candidate_all)),
            "max_abs_error_vs_pytorch_euler": float((ref_all - candidate_all).abs().max().item()),
            "vs_rk4_64_steps": output_error_metrics(rk4_all, candidate_all),
        }
        all_rk4.append(rk4_all)
        all_combined.append(candidate_all)

        timing_stats, _ = timing(
            lambda: combined(contexts[:batch_size], noises[:batch_size].clone()),
            args.runs, args.warmup,
        )
        torch_stats, _ = timing(
            lambda: pytorch(contexts[:batch_size], noises[:batch_size].clone()),
            args.runs, args.warmup,
        )
        latency[str(batch_size)] = {
            "combined_int8_trt_plus_fused_cuda": timing_stats,
            "pytorch_fp32_fixed_euler": torch_stats,
            "speedup_mean_wall": torch_stats["end_to_end_wall_ms"]["mean_ms"] / timing_stats["end_to_end_wall_ms"]["mean_ms"],
            "speedup_mean_cuda_event": torch_stats["cuda_event_ms"]["mean_ms"] / timing_stats["cuda_event_ms"]["mean_ms"],
        }

    quality = output_error_metrics(torch.cat(all_rk4), torch.cat(all_combined))
    batch1_p99 = latency.get("1", {}).get("combined_int8_trt_plus_fused_cuda", {}).get("end_to_end_wall_ms", {}).get("p99_ms")
    result = {
        "created_utc": utc_now(),
        "source_commit": source_revision(ROOT),
        "workflow_commit": workflow_revision(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "tensorrt_version": trt.__version__,
        "data_source": source,
        "checkpoint": str(checkpoint),
        "engine": str(engine_path),
        "steps": args.steps,
        "runs": args.runs,
        "warmup": args.warmup,
        "accuracy_seed": args.accuracy_seed,
        "accuracy_cases": int(len(starts)),
        "accuracy_by_batch": accuracy,
        "max_abs_output_error_vs_pytorch_euler": max_path_error,
        "mean_abs_schedule_error_vs_rk4": quality["mean_abs_output_error"],
        "mean_kl_reference_to_combined": quality["mean_kl_pytorch_to_tensorrt"],
        "latency": latency,
        "batch1_p99_end_to_end_wall_ms": batch1_p99,
        **gate(
            max_path_error, args.max_path_error,
            quality["mean_abs_output_error"], args.max_mean_abs_rk4,
            quality["mean_kl_pytorch_to_tensorrt"], args.max_mean_kl_rk4,
            batch1_p99,
        ),
    }
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["sub_2ms_p99_and_quality_target_met"]:
        raise SystemExit("combined TensorRT + fused CUDA latency or accuracy gate failed")


if __name__ == "__main__":
    main()
