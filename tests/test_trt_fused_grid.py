import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_trt_fused.py"
spec = importlib.util.spec_from_file_location("benchmark_trt_fused", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_step_grid_is_unique_sorted_and_includes_legacy_step():
    assert mod.parse_steps_grid("6,2,4,4", anchor=3) == [2, 3, 4, 6]


def test_step_grid_rejects_invalid_values():
    with pytest.raises(ValueError, match="at least 2"):
        mod.parse_steps_grid("2,1,4")
    with pytest.raises(ValueError, match="comma-separated integers"):
        mod.parse_steps_grid("2,fast,4")
    with pytest.raises(ValueError, match="must not be empty"):
        mod.parse_steps_grid(",,", anchor=4)


def test_inference_gate_requires_accuracy_quality_and_latency():
    passed = mod.gate(0.01, 0.02, 0.01, 0.015, 0.005, 0.01, 1.5)
    assert passed["sub_2ms_p99_and_quality_target_met"]
    assert not mod.gate(0.03, 0.02, 0.01, 0.015, 0.005, 0.01, 1.5)["sub_2ms_p99_and_quality_target_met"]
    assert not mod.gate(0.01, 0.02, 0.02, 0.015, 0.005, 0.01, 1.5)["sub_2ms_p99_and_quality_target_met"]
    assert not mod.gate(0.01, 0.02, 0.01, 0.015, 0.005, 0.01, 2.0)["sub_2ms_p99_and_quality_target_met"]
