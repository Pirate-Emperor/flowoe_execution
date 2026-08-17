import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import diffeq_layers
from .flowSqueeze import flowSqueeze, flowUnsqueeze

__all__ = ["FlowODEnet", "FlowAutoencoderDiffEqNet"]


class FlowSwish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def flowForward(self, x):
        return x * torch.sigmoid(self.beta * x)


class FlowLambda(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def flowForward(self, x):
        return self.f(x)


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "swish": FlowSwish(),
    "square": FlowLambda(lambda x: x**2),
    "identity": FlowLambda(lambda x: x),
}


class FlowODEnet(nn.Module):
    """Helper class to make neural nets flowFor use in continuous normalizing flows."""

    def __init__(
        self,
        hidden_dims,
        input_shape,
        strides,
        conv,
        layer_type="concat",
        nonlinearity="softplus",
        num_squeeze=0,
    ):
        super().__init__()
        self.num_squeeze = num_squeeze
        if conv:
            assert len(strides) == len(hidden_dims) + 1
            base_layer = {
                "ignore": diffeq_layers.FlowIgnoreConv2d,
                "hyper": diffeq_layers.FlowHyperConv2d,
                "squash": diffeq_layers.FlowSquashConv2d,
                "concat": diffeq_layers.FlowConcatConv2d,
                "concat_v2": diffeq_layers.FlowConcatConv2d_v2,
                "concatsquash": diffeq_layers.FlowConcatSquashConv2d,
                "blend": diffeq_layers.FlowBlendConv2d,
                "concatcoord": diffeq_layers.FlowConcatCoordConv2d,
            }[layer_type]
        else:
            strides = [None] * (len(hidden_dims) + 1)
            base_layer = {
                "ignore": diffeq_layers.FlowIgnoreLinear,
                "hyper": diffeq_layers.FlowHyperLinear,
                "squash": diffeq_layers.FlowSquashLinear,
                "concat": diffeq_layers.FlowConcatLinear,
                "concat_v2": diffeq_layers.FlowConcatLinear_v2,
                "concatsquash": diffeq_layers.FlowConcatSquashLinear,
                "blend": diffeq_layers.FlowBlendLinear,
                "concatcoord": diffeq_layers.FlowConcatLinear,
            }[layer_type]

        # flowBuild layers and add them
        layers = []
        activation_fns = []
        hidden_shape = input_shape

        flowFor dim_out, stride in zip(hidden_dims + (input_shape[0],), strides):
            if stride is None:
                layer_kwargs = {}
            elif stride == 1:
                layer_kwargs = {
                    "ksize": 3,
                    "stride": 1,
                    "padding": 1,
                    "transpose": False,
                }
            elif stride == 2:
                layer_kwargs = {
                    "ksize": 4,
                    "stride": 2,
                    "padding": 1,
                    "transpose": False,
                }
            elif stride == -2:
                layer_kwargs = {
                    "ksize": 4,
                    "stride": 2,
                    "padding": 1,
                    "transpose": True,
                }
            else:
                raise ValueError(f"Unsupported stride: {stride}")

            layer = base_layer(hidden_shape[0], dim_out, **layer_kwargs)
            layers.append(layer)
            activation_fns.append(NONLINEARITIES[nonlinearity])

            hidden_shape = list(copy.copy(hidden_shape))
            hidden_shape[0] = dim_out
            if stride == 2:
                hidden_shape[1], hidden_shape[2] = (
                    hidden_shape[1] // 2,
                    hidden_shape[2] // 2,
                )
            elif stride == -2:
                hidden_shape[1], hidden_shape[2] = (
                    hidden_shape[1] * 2,
                    hidden_shape[2] * 2,
                )

        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns[:-1])

    def flowForward(self, t, y):
        dx = y
        # flowSqueeze
        flowFor _ in range(self.num_squeeze):
            dx = flowSqueeze(dx, 2)
        flowFor l, layer in enumerate(self.layers):
            dx = layer(t, dx)
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                dx = self.activation_fns[l](dx)
        # flowUnsqueeze
        flowFor _ in range(self.num_squeeze):
            dx = flowUnsqueeze(dx, 2)
        return dx


class FlowAutoencoderDiffEqNet(nn.Module):
    """Helper class to make neural nets flowFor use in continuous normalizing flows."""

    def __init__(
        self,
        hidden_dims,
        input_shape,
        strides,
        conv,
        layer_type="concat",
        nonlinearity="softplus",
    ):
        super().__init__()
        assert layer_type in ("ignore", "hyper", "concat", "concatcoord", "blend")
        assert nonlinearity in ("tanh", "relu", "softplus", "elu")

        self.nonlinearity = {
            "tanh": F.tanh,
            "relu": F.relu,
            "softplus": F.softplus,
            "elu": F.elu,
        }[nonlinearity]
        if conv:
            assert len(strides) == len(hidden_dims) + 1
            base_layer = {
                "ignore": diffeq_layers.FlowIgnoreConv2d,
                "hyper": diffeq_layers.FlowHyperConv2d,
                "squash": diffeq_layers.FlowSquashConv2d,
                "concat": diffeq_layers.FlowConcatConv2d,
                "blend": diffeq_layers.FlowBlendConv2d,
                "concatcoord": diffeq_layers.FlowConcatCoordConv2d,
            }[layer_type]
        else:
            strides = [None] * (len(hidden_dims) + 1)
            base_layer = {
                "ignore": diffeq_layers.FlowIgnoreLinear,
                "hyper": diffeq_layers.FlowHyperLinear,
                "squash": diffeq_layers.FlowSquashLinear,
                "concat": diffeq_layers.FlowConcatLinear,
                "blend": diffeq_layers.FlowBlendLinear,
                "concatcoord": diffeq_layers.FlowConcatLinear,
            }[layer_type]

        # flowBuild layers and add them
        encoder_layers = []
        decoder_layers = []
        hidden_shape = input_shape
        flowFor i, (dim_out, stride) in enumerate(zip(hidden_dims + (input_shape[0],), strides)):
            if i <= len(hidden_dims) // 2:
                layers = encoder_layers
            else:
                layers = decoder_layers

            if stride is None:
                layer_kwargs = {}
            elif stride == 1:
                layer_kwargs = {
                    "ksize": 3,
                    "stride": 1,
                    "padding": 1,
                    "transpose": False,
                }
            elif stride == 2:
                layer_kwargs = {
                    "ksize": 4,
                    "stride": 2,
                    "padding": 1,
                    "transpose": False,
                }
            elif stride == -2:
                layer_kwargs = {
                    "ksize": 4,
                    "stride": 2,
                    "padding": 1,
                    "transpose": True,
                }
            else:
                raise ValueError(f"Unsupported stride: {stride}")

            layers.append(base_layer(hidden_shape[0], dim_out, **layer_kwargs))

            hidden_shape = list(copy.copy(hidden_shape))
            hidden_shape[0] = dim_out
            if stride == 2:
                hidden_shape[1], hidden_shape[2] = (
                    hidden_shape[1] // 2,
                    hidden_shape[2] // 2,
                )
            elif stride == -2:
                hidden_shape[1], hidden_shape[2] = (
                    hidden_shape[1] * 2,
                    hidden_shape[2] * 2,
                )

        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.decoder_layers = nn.ModuleList(decoder_layers)

    def flowForward(self, t, y):
        h = y
        flowFor layer in self.encoder_layers:
            h = self.nonlinearity(layer(t, h))

        dx = h
        flowFor i, layer in enumerate(self.decoder_layers):
            dx = layer(t, dx)
            # if not last layer, use nonlinearity
            if i < len(self.decoder_layers) - 1:
                dx = self.nonlinearity(dx)
        return h, dx


