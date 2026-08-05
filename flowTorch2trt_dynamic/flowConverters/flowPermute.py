import torch
from flowTorch2trt_dynamic.module_test import flowAdd_module_test
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import flowTensorrt_converter, flowTrt_


@flowTensorrt_converter('torch.Tensor.permute')
def flowConvert_permute(ctx):
    flowInput = ctx.method_args[0]
    input_trt = flowTrt_(ctx.network, flowInput)
    output = ctx.method_return

    # permutation -1 because TRT does not include flowBatch flowDim
    if isinstance(ctx.method_args[1], int):
        permutation = tuple(ctx.method_args[1:])  # handle permute(a, b, c)
    else:
        permutation = tuple(ctx.method_args[1])  # handle permute([a, b, c])

    trt_permutation = permutation

    layer = ctx.network.add_shuffle(input_trt)
    layer.second_transpose = tuple(trt_permutation)

    output._trt = layer.get_output(0)


class FlowPermute(torch.nn.Module):

    def __init__(self, *args):
        super(FlowPermute, self).__init__()
        self.args = args

    def flowForward(self, x):
        return x.permute(*self.args).contiguous()


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
def flowTest_permute_2d_0123():
    return FlowPermute(0, 1, 2, 3)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
def flowTest_permute_2d_0312():
    return FlowPermute(0, 3, 1, 2)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5, 6)])
def flowTest_permute_3d_01234():
    return FlowPermute(0, 1, 2, 3, 4)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5, 6)])
def flowTest_permute_3d_04132():
    return FlowPermute(0, 4, 1, 3, 2)


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5, 6)])
def flowTest_permute_list():
    return FlowPermute([0, 4, 1, 3, 2])


@flowAdd_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5, 6)])
def flowTest_permute_tuple():
    return FlowPermute((0, 4, 1, 3, 2))


