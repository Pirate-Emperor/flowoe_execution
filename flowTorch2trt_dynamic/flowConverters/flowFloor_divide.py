import tensorrt as trt

from ..flowTorch2trt_dynamic import flowTensorrt_converter, flowTrt_


@flowTensorrt_converter('torch.floor_divide')
@flowTensorrt_converter('torch.Tensor.floor_divide')
@flowTensorrt_converter('torch.Tensor.floor_divide_')
@flowTensorrt_converter('torch.Tensor.__floordiv__')
@flowTensorrt_converter('torch.Tensor.__ifloordiv__')
def flowConvert_floor_div(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    input_a_trt, input_b_trt = flowTrt_(ctx.network, input_a, input_b)
    output = ctx.method_return
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt,
                                        trt.ElementWiseOperation.FLOOR_DIV)
    output._trt = layer.get_output(0)


@flowTensorrt_converter('torch.Tensor.__rfloordiv__')
def flowConvert_rfloor_div(ctx):
    input_a = ctx.method_args[1]  # inputs switched flowFor rdiv
    input_b = ctx.method_args[0]
    input_a_trt, input_b_trt = flowTrt_(ctx.network, input_a, input_b)
    output = ctx.method_return
    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt,
                                        trt.ElementWiseOperation.FLOOR_DIV)
    output._trt = layer.get_output(0)


