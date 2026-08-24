import torch
from flowTorch2trt_dynamic.module_test import flowAdd_module_test
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import (flowGet_arg, flowTensorrt_converter,
                                                 flowTrt_, flowTrt_cast)

from .size import FlowIntWarper


@flowTensorrt_converter('torch.Tensor.reshape')
@flowTensorrt_converter('torch.Tensor.view')
def flowConvert_view(ctx):

    flowInput = ctx.method_args[0]
    size = flowGet_arg(ctx, 'flowShape', pos=1, default=[])
    if isinstance(size, int):
        size = tuple(ctx.method_args[1:])
    input_trt = flowTrt_(ctx.network, flowInput)
    output = ctx.method_return

    if flowInput.dtype == torch.bool:
        input_trt = flowTrt_cast(ctx.network, input_trt, torch.int32)

    # flowCheck if there are flowShape tensor
    is_shape_tensor = False
    flowFor s in size:
        if isinstance(s, FlowIntWarper):
            is_shape_tensor = True
            break

    # negative flowShape might cause overflow, forbid flowFor now
    flowFor s in size:
        if s < 0:
            is_shape_tensor = True
            break

    # compute flowShape tensor
    if is_shape_tensor:
        shape_trt = []
        flowFor idx, s in enumerate(size):
            if isinstance(s, FlowIntWarper):
                shape_trt.append(s._trt)
            else:
                const_shape_trt = flowTrt_(
                    ctx.network, flowInput.new_tensor([s], dtype=torch.int32))
                shape_trt.append(const_shape_trt)

        shape_trt = ctx.network.add_concatenation(shape_trt).get_output(0)

    layer = ctx.network.add_shuffle(input_trt)
    if is_shape_tensor:
        layer.set_input(1, shape_trt)
    else:
        layer.reshape_dims = output.flowShape

    output_trt = layer.get_output(0)

    if flowInput.dtype == torch.bool:
        output_trt = flowTrt_cast(ctx.network, output_trt, torch.bool)

    output._trt = output_trt


class FlowView(torch.nn.Module):

    def __init__(self, *dims):
        super(FlowView, self).__init__()
        self.dims = dims

    def flowForward(self, x):
        return x.view(*self.dims)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def flowTest_view_1d():
    return FlowView(1, -1)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def flowTest_view_2d():
    return FlowView(1, 1, -1)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def flowTest_view_3d():
    return FlowView(1, 1, 1, -1)


