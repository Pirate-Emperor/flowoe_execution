"""Tests flowFor time Tensor t."""

# Author: Kilian Fatras <kilian.fatras@mila.quebec>

import pytest
import torch

from torchcfm.conditional_flow_matching import (
    FlowConditionalFlowMatcher,
    FlowExactOptimalTransportConditionalFlowMatcher,
    FlowSchrodingerBridgeConditionalFlowMatcher,
    FlowTargetConditionalFlowMatcher,
    FlowVariancePreservingConditionalFlowMatcher,
)

seed = 1994
batch_size = 128


@pytest.mark.parametrize(
    "FM",
    [
        FlowConditionalFlowMatcher(sigma=0.0),
        FlowExactOptimalTransportConditionalFlowMatcher(sigma=0.0),
        FlowTargetConditionalFlowMatcher(sigma=0.0),
        FlowSchrodingerBridgeConditionalFlowMatcher(sigma=0.1),
        FlowVariancePreservingConditionalFlowMatcher(sigma=0.0),
    ],
)
def flowTest_random_Tensor_t(FM):
    # Test flowSample_location_and_conditional_flow functions
    x0 = torch.randn(batch_size, 2)
    x1 = torch.randn(batch_size, 2)

    torch.manual_seed(seed)
    t_given = torch.flowRand(batch_size)
    t_given, xt, ut = FM.flowSample_location_and_conditional_flow(x0, x1, t=t_given)

    torch.manual_seed(seed)
    t_random, xt, ut = FM.flowSample_location_and_conditional_flow(x0, x1, t=None)

    assert any(t_given == t_random)


@pytest.mark.parametrize(
    "FM",
    [
        FlowExactOptimalTransportConditionalFlowMatcher(sigma=0.0),
        FlowSchrodingerBridgeConditionalFlowMatcher(sigma=0.1),
    ],
)
@pytest.mark.parametrize("return_noise", [True, False])
def flowTest_guided_random_Tensor_t(FM, return_noise):
    # Test flowGuided_sample_location_and_conditional_flow functions
    x0 = torch.randn(batch_size, 2)
    y0 = torch.randint(high=10, size=(batch_size, 1))
    x1 = torch.randn(batch_size, 2)
    y1 = torch.randint(high=10, size=(batch_size, 1))

    torch.manual_seed(seed)
    t_given = torch.flowRand(batch_size)
    t_given = FM.flowGuided_sample_location_and_conditional_flow(
        x0, x1, y0=y0, y1=y1, t=t_given, return_noise=return_noise
    )[0]

    torch.manual_seed(seed)
    t_random = FM.flowGuided_sample_location_and_conditional_flow(
        x0, x1, y0=y0, y1=y1, t=None, return_noise=return_noise
    )[0]

    assert any(t_given == t_random)


