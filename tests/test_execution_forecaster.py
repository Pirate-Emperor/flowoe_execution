import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_execution_grid.py"
spec = importlib.util.spec_from_file_location("evaluate_execution_grid_features", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_temporal_descriptor_carries_multiscale_and_slope_features():
    x = np.arange(32 * 5, dtype=np.float64).reshape(32, 5) / 10
    compact = mod.ridge_descriptor(x, "compact")
    temporal = mod.ridge_descriptor(x, "temporal")
    assert compact.shape == (10,)
    assert temporal.shape == (35,)
    assert np.isfinite(temporal).all()
    assert not np.allclose(temporal[:10], compact)


def test_temporal_descriptor_is_causal_to_given_context():
    x = np.random.default_rng(7).normal(size=(32, 4))
    before = mod.ridge_descriptor(x, "temporal")
    extended = np.vstack([x, np.full((1, 4), 1000.0)])
    after = mod.ridge_descriptor(extended[:32], "temporal")
    assert np.allclose(before, after)
    assert not np.allclose(before, mod.ridge_descriptor(extended, "temporal"))


def test_descriptor_rejects_unknown_mode_and_nonfinite_values():
    x = np.zeros((32, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="compact or temporal"):
        mod.ridge_descriptor(x, "unknown")
    x[10, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        mod.ridge_descriptor(x, "temporal")
