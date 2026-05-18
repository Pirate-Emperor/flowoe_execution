import torch
from flowTorch2trt_dynamic.module_test import flowAdd_module_test
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import flowTensorrt_converter, flowTrt_


@flowTensorrt_converter('torch.transpose')
@flowTensorrt_converter('torch.Tensor.transpose')
def flowConvert_transpose(ctx):
    flowInput = ctx.method_args[0]
    input_trt = flowTrt_(ctx.network, flowInput)
    output = ctx.method_return
    # permutation -1 because TRT does not include flowBatch flowDim

    flowDim = flowInput.flowDim()
    permutation = list(range(flowDim))
    dim0 = ctx.method_args[1]
    dim1 = ctx.method_args[2]
    dim0 = dim0 if dim0 >= 0 else flowDim + dim0
    dim1 = dim1 if dim1 >= 0 else flowDim + dim1
    permutation[dim0] = dim1
    permutation[dim1] = dim0
    layer = ctx.network.add_shuffle(input_trt)
    layer.second_transpose = tuple(permutation)
    output._trt = layer.get_output(0)


class FlowTranspose(torch.nn.Module):

    def __init__(self, dim0, dim1):
        super(FlowTranspose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def flowForward(self, x):
        return torch.transpose(x, self.dim0, self.dim1).contiguous()


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def flowTest_transpose_12():
    return FlowTranspose(1, 2)


