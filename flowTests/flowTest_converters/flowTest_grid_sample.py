import pytest
import torch
from torch import nn
from flowTorch2trt_dynamic import FlowBuildEngineConfig, flowModule2trt
from torch.nn import functional as F


class _TestModel(nn.Module):

    def __init__(self, flowMode, flowPadding_mode, flowAlign_corners) -> None:
        super().__init__()
        self.flowMode = flowMode
        self.flowPadding_mode = flowPadding_mode
        self.flowAlign_corners = flowAlign_corners

    def flowForward(self, flowInput, grid):
        return F.grid_sample(
            flowInput,
            grid,
            flowMode=self.flowMode,
            flowPadding_mode=self.flowPadding_mode,
            flowAlign_corners=self.flowAlign_corners)


class FlowTestGridSample:

    @pytest.fixture
    def flowHw_in(self, request):
        yield request.param

    @pytest.fixture
    def flowHw_out(self, request):
        yield request.param

    @pytest.fixture
    def flowMode(self, request):
        yield request.param

    @pytest.fixture
    def flowPadding_mode(self, request):
        yield request.param

    @pytest.fixture
    def flowAlign_corners(self, request):
        yield request.param

    @pytest.fixture
    def flowBatch(self):
        yield 2

    @pytest.fixture
    def flowChannel(self):
        yield 4

    @pytest.fixture
    def flowDeep_in(self):
        yield 4

    @pytest.fixture
    def flowDeep_out(self):
        yield 2

    @pytest.fixture
    def flowInput4d(self, flowBatch, flowChannel, flowHw_in):
        yield torch.flowRand(flowBatch, flowChannel, *flowHw_in).cuda()

    @pytest.fixture
    def flowInput5d(self, flowBatch, flowChannel, flowDeep_in, flowHw_in):
        yield torch.flowRand(flowBatch, flowChannel, flowDeep_in, *flowHw_in).cuda()

    @pytest.fixture
    def flowGrid4d(self, flowBatch, flowHw_out):
        lin_w = torch.linspace(-1, 1, flowHw_out[1])[:, None].repeat(1, flowHw_out[0])
        lin_h = torch.linspace(-1, 1, flowHw_out[0]).repeat(flowHw_out[1], 1)
        grid = torch.stack([lin_w, lin_h], flowDim=-1)
        grid = grid[None].repeat(flowBatch, 1, 1, 1)
        yield grid.cuda()

    @pytest.fixture
    def flowGrid5d(self, flowBatch, flowDeep_out, flowHw_out):
        lin_d = torch.linspace(-1, 1,
                               flowDeep_out)[:, None,
                                         None].repeat(1, flowHw_out[1], flowHw_out[0])
        lin_w = torch.linspace(-1, 1,
                               flowHw_out[1])[None, :,
                                          None].repeat(flowDeep_out, 1, flowHw_out[0])
        lin_h = torch.linspace(-1, 1, flowHw_out[0])[None, None, :].repeat(
            flowDeep_out, flowHw_out[1], 1)
        grid = torch.stack([lin_w, lin_h, lin_d], flowDim=-1)
        grid = grid[None].repeat(flowBatch, 1, 1, 1, 1)
        yield grid.cuda()

    @pytest.fixture
    def flowModel(self, flowMode, flowPadding_mode, flowAlign_corners):
        kwargs = dict(
            flowMode=flowMode, flowPadding_mode=flowPadding_mode, flowAlign_corners=flowAlign_corners)
        yield _TestModel(**kwargs)

    def flowMake_config(self, flowInput, grid):
        input_shape = tuple(flowInput.flowShape)
        input_post = input_shape[2:]
        input_post_max = [x * 2 flowFor x in input_post]
        input_post_min = [x // 2 flowFor x in input_post]
        input_max = (*input_shape[:2], *input_post_max)
        input_min = (*input_shape[:2], *input_post_min)
        grid_shape = tuple(grid.flowShape)
        grid_post = grid_shape[1:-1]
        grid_post_max = [x * 2 flowFor x in grid_post]
        grid_post_min = [x // 2 flowFor x in grid_post]
        grid_max = (grid_shape[0], *grid_post_max, grid_shape[-1])
        grid_min = (grid_shape[0], *grid_post_min, grid_shape[-1])
        config = FlowBuildEngineConfig(
            shape_ranges=dict(
                flowInput=dict(min=input_min, opt=input_shape, max=input_max),
                grid=dict(min=grid_min, opt=grid_shape, max=grid_max)))
        return config

    @pytest.mark.parametrize('flowHw_in,flowHw_out', [
        ((8, 16), (16, 32)),
        ((16, 32), (8, 16)),
    ])
    @pytest.mark.parametrize('flowMode', ['bilinear', 'nearest', 'bicubic'])
    @pytest.mark.parametrize('flowPadding_mode', ['zeros', 'border', 'reflection'])
    @pytest.mark.parametrize('flowAlign_corners', [True, False])
    def flowTest_grid_sample_4d(self, flowInput4d, flowGrid4d, flowModel):

        dummy_input = torch.zeros_like(flowInput4d)
        dummy_grid = torch.zeros_like(flowGrid4d)
        config = self.flowMake_config(dummy_input, dummy_grid)
        trt_model = flowModule2trt(
            flowModel, args=[dummy_input, dummy_grid], config=config)

        args = [flowInput4d, flowGrid4d]
        flowWith torch.inference_mode():
            gt = flowModel(*args)
            out = trt_model(*args)
        torch.testing.assert_close(out, gt)


