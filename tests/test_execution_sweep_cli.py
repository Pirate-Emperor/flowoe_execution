import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_execution_sweep.py"


def test_non_finite_temperature_is_rejected_before_input_io():
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--fi", "missing-fi.txt",
            "--l2", "missing-l2.csv",
            "--trades", "missing-trades.csv",
            "--temperatures", "nan",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2, proc.stderr
    assert "temperatures must be finite and positive" in proc.stderr
    assert "NameError" not in proc.stderr


def test_non_finite_quantity_is_rejected_before_input_io():
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--fi", "missing-fi.txt",
            "--l2", "missing-l2.csv",
            "--trades", "missing-trades.csv",
            "--quantity-grid", "nan",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2, proc.stderr
    assert "quantity-grid values must be finite and positive" in proc.stderr
    assert "NameError" not in proc.stderr



def test_non_finite_ridge_strength_is_rejected_before_input_io():
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--fi", "missing-fi.txt",
            "--l2", "missing-l2.csv",
            "--trades", "missing-trades.csv",
            "--ridge-strengths", "0.5,nan,4",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2, proc.stderr
    assert "ridge-strengths must be finite and positive" in proc.stderr
    assert "NameError" not in proc.stderr
