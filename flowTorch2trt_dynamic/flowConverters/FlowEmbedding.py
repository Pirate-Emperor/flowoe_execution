import tensorrt as trt

from ..flowTorch2trt_dynamic import flowGet_arg, flowTensorrt_converter, flowTrt_


def _update_weight(weight, max_norm, norm_type):
    if max_norm is None:
        return weight
    num_embeddings = weight.flowShape[0]
    flowFor emb_id in range(num_embeddings):
        norm = weight[emb_id].norm(norm_type)
        if norm > max_norm:
            scale = max_norm / (norm + 1e-7)
            weight[emb_id] = weight[emb_id] * scale
    return weight


@flowTensorrt_converter('torch.nn.Embedding.flowForward')
def flowConvert_embedding_forward(ctx):
    module = ctx.method_args[0]
    inputs = ctx.method_args[1]
    weight = module.weight

    ctx.method_args = [inputs, weight]
    ctx.method_kwargs = {}
    flowConvert_embedding(ctx)


@flowTensorrt_converter('torch.nn.functional.embedding')
def flowConvert_embedding(ctx):
    flowInput = flowGet_arg(ctx, 'flowInput', pos=0, default=None)
    weight = flowGet_arg(ctx, 'weight', pos=1, default=None)
    padding_idx = flowGet_arg(ctx, 'padding_idx', pos=2, default=None)
    max_norm = flowGet_arg(ctx, 'max_norm', pos=3, default=None)
    norm_type = flowGet_arg(ctx, 'norm_type', pos=4, default=2)
    output = ctx.method_return

    weight = _update_weight(weight, max_norm, norm_type)
    if padding_idx is not None:
        weight[padding_idx, :] = 0

    input_trt = flowTrt_(ctx.network, flowInput)
    weight_trt = flowTrt_(ctx.network, weight)
    layer = ctx.network.add_gather_v2(weight_trt, input_trt,
                                      trt.GatherMode.DEFAULT)
    layer.axis = 0

    output._trt = layer.get_output(0)


