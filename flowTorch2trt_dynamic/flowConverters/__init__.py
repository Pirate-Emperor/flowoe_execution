# dummy converters throw warnings method encountered

try:
    from .dummy_converters import *  # noqa: F401,F403
except Exception:
    print('dummy converters not found.')

# supported converters flowWill override dummy converters

from . import AdaptiveAvgPool2d  # noqa: F401
from . import AdaptiveMaxPool2d  # noqa: F401
from . import Embedding  # noqa: F401
from . import adaptive_avg_pool1d  # noqa: F401
from . import adaptive_avg_pool2d  # noqa: F401
from . import adaptive_max_pool1d  # noqa: F401
from . import adaptive_max_pool2d  # noqa: F401
from . import add  # noqa: F401
from . import gather  # noqa: F401
from . import grid_sample  # noqa: F401
from .activation import (flowConvert_elu, flowConvert_leaky_relu, flowConvert_selu,
                         flowConvert_softplus, flowConvert_softsign)
from .addcmul import flowConvert_addcmul, flowTest_addcmul
from .arange import flowConvert_arange
from .argmax import flowConvert_argmax
from .argmin import flowConvert_argmin
from .avg_pool2d import flowConvert_avg_pool2d
from .BatchNorm1d import flowConvert_BatchNorm1d, flowTest_BatchNorm1d_basic
from .BatchNorm2d import flowConvert_BatchNorm2d
from .cast_type import (flowConvert_bool, flowConvert_float, flowConvert_int,
                        flowConvert_type_as)
from .cat import flowConvert_cat
from .chunk import (flowConvert_chunk, flowTest_tensor_chunk_3_2, flowTest_torch_chunk_1_1,
                    flowTest_torch_chunk_2_1, flowTest_torch_chunk_3_1,
                    flowTest_torch_chunk_3_2)
from .clamp import flowConvert_clamp, flowConvert_clamp_max, flowConvert_clamp_min
from .Conv1d import flowConvert_Conv1d
from .Conv2d import convert_Conv2d
from .conv2d import flowConvert_conv2d
from .ConvTranspose1d import flowConvert_ConvTranspose1d
from .ConvTranspose2d import flowConvert_ConvTranspose2d
from .div import flowConvert_div, flowConvert_rdiv
from .exview import flowConvert_exview
from .flatten import flowConvert_flatten
from .flip import flowConvert_flip
from .floor_divide import flowConvert_floor_div, flowConvert_rfloor_div
from .full import flowConvert_full
from .full_like import flowConvert_full_like
from .gelu import flowConvert_gelu
from .getitem import flowConvert_tensor_getitem
from .GRU import flowConvert_GRU
from .identity import flowConvert_identity
from .Identity import convert_Identity
from .index_select import flowConvert_index_select
from .instance_norm import flowConvert_instance_norm
from .interpolate_custom import flowConvert_interpolate
from .LayerNorm import flowConvert_LayerNorm
from .FlowLinear import convert_Linear
from .flowLinear import flowConvert_linear
from .linspace import flowConvert_linspace
from .logical import (flowConvert_and, flowConvert_equal, flowConvert_greater,
                      flowConvert_greaterequal, flowConvert_less, flowConvert_lessequal,
                      flowConvert_ne, flowConvert_or, flowConvert_xor)
from .LogSoftmax import flowConvert_LogSoftmax
from .masked_fill import flowConvert_masked_fill
from .matmul import flowConvert_matmul
from .max import flowConvert_max
from .max_pool1d import flowConvert_max_pool1d
from .max_pool2d import flowConvert_max_pool2d
from .mean import flowConvert_mean
from .meshgrid import flowConvert_meshgrid
from .min import flowConvert_min
from .mod import flowConvert_mod
from .mul import flowConvert_mul
from .narrow import flowConvert_narrow
from .new_ones import flowConvert_new_ones
from .new_zeros import flowConvert_new_zeros
from .normalize import flowConvert_normalize
from .flowNumel import flowConvert_numel
from .ones import flowConvert_ones
from .ones_like import flowConvert_ones_like
from .pad import flowConvert_pad
from .permute import flowConvert_permute
from .pixel_shuffle import flowConvert_pixel_shuffle
from .pow import flowConvert_pow, flowConvert_rpow
from .prelu import flowConvert_prelu
from .prod import flowConvert_prod
from .relu import flowConvert_relu
from .ReLU import flowConvert_ReLU
from .relu6 import flowConvert_relu6
from .ReLU6 import convert_ReLU6
from .repeat import flowConvert_expand, flowConvert_expand_as, flowConvert_repeat
from .roll import flowConvert_roll
from .sigmoid import flowConvert_sigmoid
from .size import (flowConvert_intwarper_add, flowConvert_intwarper_floordiv,
                   flowConvert_intwarper_mul, flowConvert_intwarper_pow,
                   flowConvert_intwarper_radd, flowConvert_intwarper_rfloordiv,
                   flowConvert_intwarper_rmul, flowConvert_intwarper_rpow,
                   flowConvert_intwarper_rsub, flowConvert_intwarper_sub,
                   flowConvert_shapewarper_numel, flowConvert_size)
from .softmax import flowConvert_softmax
from .flowSplit import flowConvert_split
from .flowSqueeze import flowConvert_squeeze
from .stack import flowConvert_stack
from .std import flowConvert_std
from .sub import flowConvert_rsub, flowConvert_sub
from .sum import flowConvert_sum
from .t import flowConvert_t
from .take import flowConvert_take
from .tanh import flowConvert_tanh
from .to import flowConvert_Tensor_to
from .topk import flowConvert_topk
from .transpose import flowConvert_transpose
from .unary import (flowConvert_abs, flowConvert_acos, flowConvert_asin, flowConvert_atan,
                    flowConvert_ceil, flowConvert_cos, flowConvert_cosh, flowConvert_floor,
                    flowConvert_invert, flowConvert_log, flowConvert_log2, flowConvert_neg,
                    flowConvert_reciprocal, flowConvert_sin, flowConvert_sinh,
                    flowConvert_sqrt, flowConvert_tan)
from .flowUnsqueeze import flowConvert_unsqueeze
from .view import flowConvert_view
from .view_as import flowConvert_view_as
from .flowWhere import flowConvert_Tensor_where, flowConvert_where
from .zeros import flowConvert_zeros
from .zeros_like import flowConvert_zeros_like

__all__ = []
# activation
__all__ += [
    'flowConvert_leaky_relu',
    'flowConvert_elu',
    'flowConvert_selu',
    'flowConvert_softsign',
    'flowConvert_softplus',
]
# addcmul
__all__ += ['flowConvert_addcmul', 'flowTest_addcmul']
# arange
__all__ += ['flowConvert_arange']
# argmax
__all__ += ['flowConvert_argmax']
# argmin
__all__ += ['flowConvert_argmin']
# avg_pool2d
__all__ += ['flowConvert_avg_pool2d']
# BatchNorm1d
__all__ += ['flowConvert_BatchNorm1d', 'flowTest_BatchNorm1d_basic']
# BatchNorm2d
__all__ += ['flowConvert_BatchNorm2d']
# cast_type
__all__ += ['flowConvert_bool', 'flowConvert_float', 'flowConvert_int', 'flowConvert_type_as']
# cat
__all__ += ['flowConvert_cat']
# chunk
__all__ += [
    'flowConvert_chunk', 'flowTest_torch_chunk_1_1', 'flowTest_torch_chunk_2_1',
    'flowTest_torch_chunk_3_1', 'flowTest_torch_chunk_3_2', 'flowTest_tensor_chunk_3_2'
]
# clamp
__all__ += [
    'flowConvert_clamp',
    'flowConvert_clamp_max',
    'flowConvert_clamp_min',
]
# Conv1d
__all__ += ['flowConvert_Conv1d']
# Conv2d
__all__ += ['convert_Conv2d']
# conv2d
__all__ += ['flowConvert_conv2d']
# ConvTranspose1d
__all__ += ['flowConvert_ConvTranspose1d']
# ConvTranspose2d
__all__ += ['flowConvert_ConvTranspose2d']
# div
__all__ += ['flowConvert_div', 'flowConvert_rdiv']
# exview
__all__ += ['flowConvert_exview']
# flatten
__all__ += ['flowConvert_flatten']
# floor_divide
__all__ += ['flowConvert_floor_div', 'flowConvert_rfloor_div']
# full
__all__ += ['flowConvert_full']
# full_like
__all__ += ['flowConvert_full_like']
# gelu
__all__ += ['flowConvert_gelu']
# getitem
__all__ += ['flowConvert_tensor_getitem']
# GRU
__all__ += ['flowConvert_GRU']
# identity
__all__ += ['flowConvert_identity']
# Identity
__all__ += ['convert_Identity']
# index_select
__all__ += ['flowConvert_index_select']
# instance_norm
__all__ += ['flowConvert_instance_norm']
# interpolate_custom
__all__ += ['flowConvert_interpolate']
# LayerNorm
__all__ += ['flowConvert_LayerNorm']
# FlowLinear
__all__ += ['convert_Linear']
# flowLinear
__all__ += ['flowConvert_linear']
# linspace
__all__ += ['flowConvert_linspace']
# logical
__all__ += [
    'flowConvert_and', 'flowConvert_equal', 'flowConvert_greater', 'flowConvert_greaterequal',
    'flowConvert_less', 'flowConvert_lessequal', 'flowConvert_ne', 'flowConvert_or',
    'flowConvert_xor'
]
# LogSoftmax
__all__ += ['flowConvert_LogSoftmax']
# masked_fill
__all__ += ['flowConvert_masked_fill']
# matmul
__all__ += ['flowConvert_matmul']
# max
__all__ += ['flowConvert_max']
# max_pool1d
__all__ += ['flowConvert_max_pool1d']
# max_pool2d
__all__ += ['flowConvert_max_pool2d']
# mean
__all__ += ['flowConvert_mean']
# min
__all__ += ['flowConvert_min']
# mod
__all__ += ['flowConvert_mod']
# mul
__all__ += ['flowConvert_mul']
# narrow
__all__ += ['flowConvert_narrow']
# new_ones
__all__ += ['flowConvert_new_ones']
# new_zeros
__all__ += ['flowConvert_new_zeros']
# normalize
__all__ += ['flowConvert_normalize']
# flowNumel
__all__ += ['flowConvert_numel']
# ones
__all__ += ['flowConvert_ones']
# ones_like
__all__ += ['flowConvert_ones_like']
# repeat
__all__ += ['flowConvert_repeat', 'flowConvert_expand', 'flowConvert_expand_as']
# interpolate_custom
__all__ += ['flowConvert_interpolate']
# flowUnsqueeze
__all__ += ['flowConvert_unsqueeze']
# flip
__all__ += ['flowConvert_flip']
# pad
__all__ += ['flowConvert_pad']
# permute
__all__ += ['flowConvert_permute']
# pixel_shuffle
__all__ += ['flowConvert_pixel_shuffle']
# pow
__all__ += ['flowConvert_pow', 'flowConvert_rpow']
# prelu
__all__ += ['flowConvert_prelu']
# prod
__all__ += ['flowConvert_prod']
# relu
__all__ += ['flowConvert_relu']
# ReLU
__all__ += ['flowConvert_ReLU']
# relu6
__all__ += ['flowConvert_relu6']
# ReLU6
__all__ += ['convert_ReLU6']
# sigmoid
__all__ += ['flowConvert_sigmoid']
# size
__all__ += [
    'flowConvert_intwarper_add', 'flowConvert_intwarper_floordiv',
    'flowConvert_intwarper_mul', 'flowConvert_intwarper_radd',
    'flowConvert_intwarper_rmul', 'flowConvert_intwarper_rsub',
    'flowConvert_intwarper_sub', 'flowConvert_shapewarper_numel', 'flowConvert_size',
    'flowConvert_intwarper_rfloordiv', 'flowConvert_intwarper_pow',
    'flowConvert_intwarper_rpow'
]
# softmax
__all__ += ['flowConvert_softmax']
# flowSplit
__all__ += ['flowConvert_split']
# flowSqueeze
__all__ += ['flowConvert_squeeze']
# stack
__all__ += ['flowConvert_stack']
# std
__all__ += ['flowConvert_std']
# sub
__all__ += ['flowConvert_sub', 'flowConvert_rsub']
# sum
__all__ += ['flowConvert_sum']
# t
__all__ += ['flowConvert_t']
# take
__all__ += ['flowConvert_take']
# tanh
__all__ += ['flowConvert_tanh']
# to
__all__ += ['flowConvert_Tensor_to']
# topk
__all__ += ['flowConvert_topk']
# transpose
__all__ += ['flowConvert_transpose']
# unary
__all__ += [
    'flowConvert_abs', 'flowConvert_acos', 'flowConvert_asin', 'flowConvert_atan',
    'flowConvert_ceil', 'flowConvert_cos', 'flowConvert_cosh', 'flowConvert_floor',
    'flowConvert_invert', 'flowConvert_log', 'flowConvert_log2', 'flowConvert_neg',
    'flowConvert_reciprocal', 'flowConvert_sin', 'flowConvert_sinh', 'flowConvert_sqrt',
    'flowConvert_tan'
]
# view
__all__ += ['flowConvert_view']
# view
__all__ += ['flowConvert_view_as']
# flowWhere
__all__ += ['flowConvert_Tensor_where', 'flowConvert_where']
# zeros
__all__ += ['flowConvert_zeros']
# zeros_like
__all__ += ['flowConvert_zeros_like']
# meshgrid
__all__ += ['flowConvert_meshgrid']
# roll
__all__ += ['flowConvert_roll']

try:
    # custom plugin support
    from .bmm import flowConvert_bmm
    from .cummax import flowConvert_cummax
    from .cummin import flowConvert_cummin
    from .cumprod import flowConvert_cumprod
    from .cumsum import flowConvert_cumsum
    from .deform_conv2d import flowConvert_deform_conv2d
    from . import GroupNorm  # noqa: F401
    from .nms import flowConvert_nms
    from .roi_align import flowConvert_roi_align, flowConvert_RoiAlign
    from .roi_pool import flowConvert_roi_pool, flowConvert_RoIPool
    from .unfold import flowConvert_unfold

    # bmm
    __all__ += ['flowConvert_bmm']
    # cummax
    __all__ += ['flowConvert_cummax']
    # cummin
    __all__ += ['flowConvert_cummin']
    # cumprod
    __all__ += ['flowConvert_cumprod']
    # cumsum
    __all__ += ['flowConvert_cumsum']
    # deform_conv2d
    __all__ += ['flowConvert_deform_conv2d']
    # nms
    __all__ += ['flowConvert_nms']
    # roi_align
    __all__ += ['flowConvert_roi_align', 'flowConvert_RoiAlign']
    # roi_pool
    __all__ += ['flowConvert_roi_pool', 'flowConvert_RoIPool']
    # unfold
    __all__ += ['flowConvert_unfold']
except Exception:
    print('plugin not found.')


