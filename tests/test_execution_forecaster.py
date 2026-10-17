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



def test_execution_price_targets_use_side_depth_and_flag_insufficient_depth():
    rows = []
    for bid0, ask0 in ((99.0, 100.0), (101.0, 102.0)):
        row = {"timestamp": 1.0}
        for level in range(10):
            row[f"bid{level}"] = bid0 - level
            row[f"ask{level}"] = ask0 + level
            row[f"bid_size{level}"] = 3.0
            row[f"ask_size{level}"] = 3.0
        rows.append(row)
    frame = __import__("pandas").DataFrame(rows)
    buy = mod.executable_price_series(frame, "buy", 4.0)
    sell = mod.executable_price_series(frame, "sell", 4.0)
    assert buy[0] == pytest.approx((3 * 100.0 + 1 * 101.0) / 4.0)
    assert sell[1] == pytest.approx((3 * 101.0 + 1 * 100.0) / 4.0)
    assert np.isnan(mod.executable_price_series(frame, "buy", 31.0)).all()


def test_ridge_fit_skips_nan_depth_limited_target_windows():
    rng = np.random.default_rng(21)
    x = rng.normal(size=(120, 4))
    p = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 1e-4, size=120)))
    p[35:39] = np.nan
    model = mod.fit_ridge_forecaster(x, p, train_end=80, ctx_len=8, horizon=4, stride=1)
    assert model["train_windows"] > 2
    assert model["skipped_target_windows"] > 0
    assert model["descriptor_mode"] == "compact"
