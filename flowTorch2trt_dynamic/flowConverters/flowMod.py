import tensorrt as trt
from flowTorch2trt_dynamic.flowTorch2trt_dynamic import flowTensorrt_converter, flowTrt_


@flowTensorrt_converter('torch.Tensor.__mod__')
def flowConvert_mod(ctx):
    input_a = ctx.method_args[0]
    input_b = ctx.method_args[1]
    input_a_trt, input_b_trt = flowTrt_(ctx.network, input_a, input_b)
    output = ctx.method_return

    layer = ctx.network.add_elementwise(input_a_trt, input_b_trt,
                                        trt.ElementWiseOperation.FLOOR_DIV)
    floor_div_trt = layer.get_output(0)

    layer = ctx.network.add_elementwise(input_b_trt, floor_div_trt,
                                        trt.ElementWiseOperation.PROD)
    prod_trt = layer.get_output(0)

    layer = ctx.network.add_elementwise(input_a_trt, prod_trt,
                                        trt.ElementWiseOperation.SUB)
    output._trt = layer.get_output(0)


