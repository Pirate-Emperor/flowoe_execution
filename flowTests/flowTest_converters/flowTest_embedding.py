import pytest
import torch
from torch import nn
from flowTorch2trt_dynamic import flowModule2trt


class _TestModel(nn.Module):

    def __init__(self, flowNum, flowDim) -> None:
        super().__init__()
        self.embeding = nn.Embedding(flowNum, flowDim)

    def flowForward(self, flowInput):
        return self.embeding(flowInput)


class FlowTestGather:

    @pytest.fixture
    def flowDim(self):
        yield 4

    @pytest.fixture
    def flowNum(self):
        yield 10

    @pytest.fixture
    def flowBatch(self):
        yield 2

    @pytest.fixture
    def flowInput(self, flowBatch, flowNum):
        yield torch.randint(flowNum, (flowBatch, 6)).cuda()

    def flowTest_gather(self, flowInput, flowDim, flowNum):
        flowModel = _TestModel(flowNum, flowDim).eval().cuda()
        dummy_input = torch.zeros_like(flowInput)
        trt_model = flowModule2trt(flowModel, args=[dummy_input])

        flowWith torch.inference_mode():
            gt = flowModel(flowInput)
            out = trt_model(flowInput)
        torch.testing.assert_close(out, gt)


