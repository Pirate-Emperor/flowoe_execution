import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_execution_grid.py"
spec = importlib.util.spec_from_file_location("evaluate_execution_grid_oracle", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def book_frame():
    rows = []
    for bid0, ask0, bsz, asz in ((99.0, 100.0, 3.0, 3.0), (101.0, 102.0, 3.0, 3.0)):
        row = {"timestamp": 1.0}
        for level in range(10):
            row[f"bid{level}"] = bid0 - level
            row[f"ask{level}"] = ask0 + level
            row[f"bid_size{level}"] = bsz
            row[f"ask_size{level}"] = asz
        rows.append(row)
    return pd.DataFrame(rows)


def test_buy_oracle_uses_lowest_future_ask_first():
    schedule, meta = mod.hindsight_oracle_schedule(book_frame(), 1.0, "buy")
    assert np.allclose(schedule, [1.0, 0.0])
    assert meta["hindsight_fill_capacity"] == pytest.approx(1.0)
    assert not meta["capacity_limited"]


def test_sell_oracle_uses_highest_future_bid_first():
    schedule, meta = mod.hindsight_oracle_schedule(book_frame(), 1.0, "sell")
    assert np.allclose(schedule, [0.0, 1.0])
    assert meta["hindsight_fill_capacity"] == pytest.approx(1.0)
    assert not meta["capacity_limited"]


def test_oracle_reports_depth_limited_order_and_still_normalizes():
    frame = book_frame()
    schedule, meta = mod.hindsight_oracle_schedule(frame, 100.0, "buy")
    assert np.allclose(schedule.sum(), 1.0)
    assert meta["capacity_limited"]
    assert meta["hindsight_fill_capacity"] == pytest.approx(60.0)


@pytest.mark.parametrize("side", ["buy", "sell"])
def test_oracle_rejects_invalid_quantity_and_side(side):
    with pytest.raises(ValueError, match="finite and positive"):
        mod.hindsight_oracle_schedule(book_frame(), float("nan"), side)
    with pytest.raises(ValueError, match="side must be"):
        mod.hindsight_oracle_schedule(book_frame(), 1.0, "hold")
