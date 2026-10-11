import pytest
import torch

from scripts.benchmark_trt import accuracy_gate, latency_target_gate, output_error_metrics
from scripts.build_trt_engine import add_selu_activation, calibration_noise


def test_output_error_metrics_are_zero_for_matching_schedules():
    ref = torch.softmax(torch.tensor([[0.1, 0.2, 0.3, 0.4]]), dim=1)
    result = output_error_metrics(ref, ref.clone())
    assert result["max_abs_output_error"] == pytest.approx(0.0)
    assert result["mean_abs_output_error"] == pytest.approx(0.0)
    assert result["mean_kl_pytorch_to_tensorrt"] == pytest.approx(0.0, abs=1e-7)
    assert result["pytorch_mean_schedule_sum"] == pytest.approx(1.0)
    assert result["tensorrt_mean_schedule_sum"] == pytest.approx(1.0)


def test_output_error_metrics_measure_drift_and_kl():
    ref = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    candidate = torch.tensor([[0.12, 0.18, 0.31, 0.39]])
    result = output_error_metrics(ref, candidate)
    assert result["max_abs_output_error"] == pytest.approx(0.02)
    assert result["mean_abs_output_error"] > 0
    assert result["mean_kl_pytorch_to_tensorrt"] > 0


def test_output_error_metrics_reject_shape_mismatch_and_nonfinite():
    with pytest.raises(ValueError, match="identical"):
        output_error_metrics(torch.ones(1, 3), torch.ones(1, 4))
    with pytest.raises(ValueError, match="finite"):
        output_error_metrics(torch.tensor([[float("nan"), 0.5]]), torch.tensor([[0.5, 0.5]]))


def test_accuracy_gate_uses_inclusive_threshold():
    assert accuracy_gate(0.02, 0.02)["accuracy_gate_passed"] is True
    assert accuracy_gate(0.02001, 0.02)["accuracy_gate_passed"] is False
    with pytest.raises(ValueError):
        accuracy_gate(float("nan"), 0.02)


def test_latency_gate_does_not_accept_exactly_two_milliseconds():
    assert latency_target_gate(True, 1.99)["latency_claim_allowed"] is True
    assert latency_target_gate(True, 2.0)["latency_claim_allowed"] is False
    assert latency_target_gate(False, 1.0)["latency_claim_allowed"] is False



def test_int8_calibration_noise_is_reproducible_and_not_repeated():
    x = torch.zeros((1, 8), dtype=torch.float32)
    a = calibration_noise(x, count=16, seed=99)
    b = calibration_noise(x, count=16, seed=99)
    c = calibration_noise(x, count=16, seed=100)
    assert len(a) == 16
    assert all(torch.equal(u, v) for u, v in zip(a, b))
    assert not torch.equal(a[0], a[1])
    assert not torch.equal(a[0], c[0])


def test_calibration_noise_rejects_nonpositive_count():
    with pytest.raises(ValueError, match="positive"):
        calibration_noise(torch.zeros((1, 8)), count=0)


def test_selu_converter_bypasses_dynamic_rank_helper():
    class FakeLayer:
        alpha = None
        beta = None

        def get_output(self, index=0):
            assert index == 0
            return "converted-selu"

    class FakeNetwork:
        def __init__(self):
            self.args = None
            self.layer = FakeLayer()

        def add_activation(self, tensor, activation):
            self.args = (tensor, activation)
            return self.layer

    class FakeTRT:
        class ActivationType:
            SELU = "SELU"

    network = FakeNetwork()
    converted = object()
    value = type("TensorProxy", (), {"_trt": converted})()
    out = add_selu_activation(network, value, FakeTRT)
    assert out == "converted-selu"
    assert network.args == (converted, "SELU")
    assert network.layer.alpha == pytest.approx(1.6732632423543772)
    assert network.layer.beta == pytest.approx(1.0507009873554805)


def test_selu_converter_rejects_tensor_without_trt_value():
    class FakeTRT:
        class ActivationType:
            SELU = "SELU"

    with pytest.raises(RuntimeError, match="not been converted"):
        add_selu_activation(object(), object(), FakeTRT)
