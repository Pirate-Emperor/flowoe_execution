import numpy as np
import torch
import pytest

from scripts.run_experiment import expert, fi_training_windows
from scripts.evaluate_execution_grid import compare_schedules, summarize_paired_method, initial_state_for_mode
from scripts.benchmark_trt import accuracy_gate, latency_target_gate


def test_fi_training_windows_keep_all_label_horizons_inside_training_split():
    features = torch.arange(100 * 4, dtype=torch.float32).reshape(100, 4)
    labels = np.arange(100 * 5, dtype=np.int16).reshape(100, 5)
    contexts, targets = fi_training_windows(
        features, labels, train_ratio=0.60, context_len=8, stride=4, max_label_horizon=10
    )
    assert len(contexts) == len(targets) > 0
    split = 60
    for ctx, target in zip(contexts, targets):
        final_row = int(ctx[-1, 0].item() // 4)
        assert final_row + 10 < split
        torch.testing.assert_close(target, torch.as_tensor(labels[final_row], dtype=torch.long))


@pytest.mark.parametrize("args", [
    (np.zeros((5, 4)), np.zeros((4, 5)), 0.6, 2, 1, 1),
    (np.zeros((5, 4)), np.zeros((5, 5)), 1.0, 2, 1, 1),
    (np.zeros((5, 4)), np.zeros((5, 5)), 0.6, 0, 1, 1),
])
def test_fi_training_windows_reject_invalid_inputs(args):
    with pytest.raises(ValueError):
        fi_training_windows(*args)



def test_schedule_comparison_is_zero_for_identical_normalized_outputs():
    p = torch.softmax(torch.tensor([[1.0, 2.0, -1.0], [0.4, 0.1, 0.2]]), dim=1)
    result = compare_schedules(p, p.clone())
    assert result["mean_abs_error"] == pytest.approx(0.0)
    assert result["max_abs_error"] == pytest.approx(0.0)
    assert result["mean_kl_reference_to_candidate"] == pytest.approx(0.0, abs=1e-7)


def test_schedule_comparison_rejects_shape_mismatch_and_nonfinite_values():
    with pytest.raises(ValueError, match="identical"):
        compare_schedules(torch.ones(1, 3), torch.ones(1, 4))
    with pytest.raises(ValueError, match="finite"):
        compare_schedules(torch.tensor([[1.0, float("nan")]]), torch.ones(1, 2))



def test_paired_method_summary_accepts_only_completed_episode_pairs():
    baseline = [2.0, 2.5, 2.2]
    slippage = [1.8, 2.4, 2.0]
    completion = [1.0, 0.9, 1.0]
    errors = [
        {"mean_abs_error": 0.0, "max_abs_error": 0.0, "mean_kl_reference_to_candidate": 0.0}
        for _ in baseline
    ]
    result = summarize_paired_method(baseline, slippage, completion, errors, seed=11)
    assert result["episodes"] == 3
    assert result["baseline_episodes"] == 3
    assert result["improvement_vs_twap_bps"] == pytest.approx(0.1666666667)
    assert result["paired_improvement_ci95_bps"][0] <= result["paired_improvement_vs_twap_bps"]


def test_paired_method_summary_rejects_partial_baseline_alignment():
    errors = [{"mean_abs_error": 0.0, "max_abs_error": 0.0, "mean_kl_reference_to_candidate": 0.0}]
    with pytest.raises(ValueError, match="same episodes"):
        summarize_paired_method([2.0, 2.1], [1.8], [1.0], errors)



def test_tensor_rt_accuracy_gate_is_separate_from_latency_claim():
    passed = accuracy_gate(0.01, 0.05)
    failed = accuracy_gate(0.08, 0.05)
    assert passed["status"] == "passed"
    assert passed["accuracy_gate_passed"] is True
    assert failed["status"] == "failed_accuracy"
    assert failed["accuracy_gate_passed"] is False
    assert "latency_claim_allowed" not in passed


def test_latency_claim_requires_accuracy_and_batch1_p99_below_target():
    assert latency_target_gate(True, 1.9)["latency_claim_allowed"] is True
    assert latency_target_gate(True, 2.0)["latency_claim_allowed"] is False
    assert latency_target_gate(True, 2.7)["sub_2ms_batch1_p99_target_met"] is False
    assert latency_target_gate(False, 1.2)["latency_claim_allowed"] is False
    assert latency_target_gate(True, None)["latency_claim_allowed"] is False


def test_tensor_rt_gates_reject_invalid_thresholds():
    with pytest.raises(ValueError, match="finite"):
        accuracy_gate(float("nan"), 0.05)
    with pytest.raises(ValueError, match="tolerance positive"):
        accuracy_gate(0.01, 0.0)
    with pytest.raises(ValueError, match="finite and positive"):
        latency_target_gate(True, 1.0, target_ms=float("inf"))



def test_initial_state_ablation_random_clone_and_zero():
    x = torch.tensor([[1.0, -2.0, 3.0]])
    random_state = initial_state_for_mode(x, "random")
    zero_state = initial_state_for_mode(x, "zero")
    torch.testing.assert_close(random_state, x)
    torch.testing.assert_close(zero_state, torch.zeros_like(x))
    assert random_state.data_ptr() != x.data_ptr()
    x[0, 0] = 99.0
    assert random_state[0, 0].item() == pytest.approx(1.0)


def test_initial_state_ablation_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unsupported latent mode"):
        initial_state_for_mode(torch.zeros(1, 8), "future")



def test_ridge_forecaster_uses_training_rows_only_and_returns_valid_schedules():
    from scripts.evaluate_execution_grid import fit_ridge_forecaster, predict_ridge_schedule

    rng = np.random.default_rng(31)
    n = 180
    features = rng.normal(size=(n, 45)).astype(np.float64)
    mids = 100.0 * np.exp(np.cumsum(rng.normal(0, 1e-4, size=n)))
    train_end = 120
    model_a = fit_ridge_forecaster(features, mids, train_end, ctx_len=32, horizon=8, stride=4)
    changed_features = features.copy()
    changed_mids = mids.copy()
    changed_features[train_end:] += 10000.0
    changed_mids[train_end:] *= 2.0
    model_b = fit_ridge_forecaster(changed_features, changed_mids, train_end, ctx_len=32, horizon=8, stride=4)
    np.testing.assert_allclose(model_a["coefficients"], model_b["coefficients"], atol=0, rtol=0)
    np.testing.assert_allclose(model_a["feature_mean"], model_b["feature_mean"], atol=0, rtol=0)
    assert model_a["train_windows"] > 1
    ctx = features[train_end - 32:train_end]
    pred, buy = predict_ridge_schedule(model_a, ctx, "buy", 1.0)
    _, sell = predict_ridge_schedule(model_a, ctx, "sell", 1.0)
    assert pred.shape == (8,)
    assert np.isfinite(pred).all()
    assert np.isclose(buy.sum(), 1.0)
    assert np.isclose(sell.sum(), 1.0)
    assert (buy >= 0).all() and (sell >= 0).all()
    with pytest.raises(ValueError):
        predict_ridge_schedule(model_a, ctx, "hold", 1.0)



def test_geometric_schedule_blend_endpoints_and_validation():
    from scripts.evaluate_execution_grid import blend_schedules

    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.1, 0.3, 0.6])
    np.testing.assert_allclose(blend_schedules(p, q, 0.0), p / p.sum(), atol=1e-12)
    np.testing.assert_allclose(blend_schedules(p, q, 1.0), q / q.sum(), atol=1e-12)
    middle = blend_schedules(p, q, 0.5)
    assert np.isclose(middle.sum(), 1.0)
    assert (middle > 0).all()
    with pytest.raises(ValueError, match="in"):
        blend_schedules(p, q, 1.1)
    with pytest.raises(ValueError, match="finite"):
        blend_schedules(np.array([0.1, np.nan, 0.9]), q, 0.5)



def test_cross_date_block_bootstrap_preserves_separate_dates_and_is_reproducible():
    from scripts.aggregate_execution_sweeps import cross_date_block_bootstrap_ci

    days = [[1.0] * 12, [3.0] * 12, [5.0] * 12]
    a = cross_date_block_bootstrap_ci(days, seed=17, n_boot=500, block_size=3)
    b = cross_date_block_bootstrap_ci(days, seed=17, n_boot=500, block_size=3)
    assert a == b
    assert a[0] <= 3.0 <= a[1]


def test_cross_date_block_bootstrap_rejects_empty_or_nonfinite_series():
    from scripts.aggregate_execution_sweeps import cross_date_block_bootstrap_ci

    with pytest.raises(ValueError):
        cross_date_block_bootstrap_ci([])
    with pytest.raises(ValueError):
        cross_date_block_bootstrap_ci([[1.0, float("nan")]])
