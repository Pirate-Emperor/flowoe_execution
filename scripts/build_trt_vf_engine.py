"""Build an INT8 TensorRT vector-field engine for the fused Euler loop."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from flowoe_execution.data import load_l2_csv
from flowoe_execution.features import l2_features
from flowoe_execution.model import CFMPolicy
from flowoe_execution.provenance import source_revision, utc_now, workflow_revision

ROOT = Path(__file__).resolve().parents[1]


class VectorFieldModule(nn.Module):
    """Expose just the time-conditioned vector field as a TensorRT module."""

    def __init__(self, policy: CFMPolicy):
        super().__init__()
        self.vf = policy.vf

    def forward(self, t: torch.Tensor, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        return self.vf(t, x, ctx)


def register_selu_converter():
    """Use native TensorRT SELU without the converter's shape-rank helper."""
    import tensorrt as trt
    from torch2trt_dynamic.torch2trt_dynamic import tensorrt_converter

    @tensorrt_converter("torch.nn.SELU.forward")
    @tensorrt_converter("torch.nn.modules.activation.SELU.forward")
    @tensorrt_converter("torch.selu_")
    @tensorrt_converter("torch.nn.functional.selu")
    @tensorrt_converter("torch.selu")
    def convert_selu(ctx):
        value = next(
            (arg for arg in reversed(ctx.method_args) if isinstance(arg, torch.Tensor)),
            None,
        )
        if value is None or getattr(value, "_trt", None) is None:
            raise RuntimeError("SELU input has no TensorRT conversion")
        layer = ctx.network.add_activation(value._trt, trt.ActivationType.SELU)
        layer.alpha = 1.6732632423543772
        layer.beta = 1.0507009873554805
        ctx.method_return._trt = layer.get_output(0)


def select_contexts(policy: CFMPolicy, checkpoint: Path, count: int, seed: int):
    """Make deterministic calibration triplets from matching available L2 data."""
    real_files = sorted((ROOT / "data/real/crypto").glob("*_l2.csv"))
    smoke_file = ROOT / "data/smoke/crypto_l2.csv"
    choices = (real_files + ([smoke_file] if smoke_file.exists() else [])) if checkpoint.name == "cfm_policy.pt" else (
        ([smoke_file] if smoke_file.exists() else []) + real_files
    )
    source = None
    features = None
    for path in choices:
        try:
            feats = l2_features(load_l2_csv(path, max_rows=20000).snapshots)
            if len(feats) >= 32:
                source = f"real crypto L2 ({path.stem})" if "data/real/" in str(path) else "synthetic smoke L2"
                features = feats
                break
        except Exception:
            continue
    if features is None:
        raise SystemExit("No valid L2 context source available for INT8 calibration")
    starts = np.linspace(0, len(features) - 32, num=count, dtype=int)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    rows = []
    with torch.no_grad():
        for i, start in enumerate(starts):
            raw = torch.from_numpy(features[start:start + 32]).unsqueeze(0).cuda()
            encoded = policy.context.crypto_encode(raw)
            x = torch.randn((1, 8), generator=generator, device="cuda")
            t = torch.tensor([i / max(1, count - 1)], dtype=torch.float32, device="cuda")
            rows.append({"t": t, "x": x, "ctx": encoded})
    return rows, source


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="results/cfm_policy.pt")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--output", default="results/flowoe_vf_int8.pth")
    ap.add_argument("--calibration-size", type=int, default=256)
    ap.add_argument("--calibration-seed", type=int, default=20261011)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    if args.steps < 2 or args.calibration_size < 32:
        ap.error("steps must be >= 2 and calibration-size >= 32")
    try:
        import tensorrt as trt
        from torch2trt_dynamic import module2trt, BuildEngineConfig, SequenceDataset
        register_selu_converter()
    except Exception as exc:
        raise SystemExit(f"TensorRT/converter unavailable: {type(exc).__name__}: {exc}")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    policy = CFMPolicy(8).cuda().eval()
    policy.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    wrapper = VectorFieldModule(policy).cuda().eval()
    calibration, calibration_source = select_contexts(
        policy, checkpoint, args.calibration_size, args.calibration_seed,
    )
    t = torch.zeros((1,), dtype=torch.float32, device="cuda")
    x = torch.zeros((1, 8), dtype=torch.float32, device="cuda")
    ctx = torch.zeros((1, 96), dtype=torch.float32, device="cuda")
    cfg = BuildEngineConfig(
        shape_ranges={
            "t": {"min": (1,), "opt": (1,), "max": (4,)},
            "x": {"min": (1, 8), "opt": (1, 8), "max": (4, 8)},
            "ctx": {"min": (1, 96), "opt": (1, 96), "max": (4, 96)},
        },
        int8=True,
        int8_calib_dataset=SequenceDataset(calibration),
        int8_batch_size=1,
        int8_cache_file=str(ROOT / "results/flowoe_vf_int8.cache"),
    )
    engine = module2trt(wrapper, args=[t, x, ctx], config=cfg)
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(engine.state_dict(), out)
    report = {
        "created_utc": utc_now(),
        "source_commit": source_revision(ROOT),
        "workflow_commit": workflow_revision(),
        "status": "built",
        "precision_requested": "INT8",
        "component": "time-conditioned CFM vector field only",
        "tensorrt_version": trt.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "steps": args.steps,
        "calibration_samples": len(calibration),
        "calibration_seed": args.calibration_seed,
        "calibration_source": calibration_source,
        "checkpoint_path": str(checkpoint),
        "engine_path": str(out),
        "context_encoder": "PyTorch FP32",
        "euler_update": "fused CUDA extension at inference",
    }
    (ROOT / "results/tensorrt_vf_build.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
