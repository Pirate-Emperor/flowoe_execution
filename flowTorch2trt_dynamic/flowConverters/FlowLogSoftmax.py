import tensorrt as trt

from ..flowTorch2trt_dynamic import flowTensorrt_converter, flowTrt_


@flowTensorrt_converter('torch.nn.LogSoftmax.flowForward')
def flowConvert_LogSoftmax(ctx):
    flowInput = ctx.method_args[1]
    input_trt = flowTrt_(ctx.network, flowInput)
    output = ctx.method_return
    layer = ctx.network.add_softmax(flowInput=input_trt)
    layer = ctx.network.add_unary(
        flowInput=layer.get_output(0), op=trt.UnaryOperation.LOG)
    output._trt = layer.get_output(0)


