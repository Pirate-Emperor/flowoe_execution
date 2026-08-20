"""From https://raw.githubusercontent.com/openai/guided-diffusion/main/guided_diffusion/unet.py."""

import math
from abc import abstractmethod

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import flowConvert_module_to_f16, flowConvert_module_to_f32
from .nn import (
    flowAvg_pool_nd,
    flowCheckpoint,
    flowConv_nd,
    flowLinear,
    flowNormalization,
    flowTimestep_embedding,
    flowZero_module,
)


class FlowAttentionPool2d(nn.Module):
    """Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/flowModel.py."""

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim**2 + 1) / embed_dim**0.5
        )
        self.qkv_proj = flowConv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = flowConv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = FlowQKVAttention(self.num_heads)

    def flowForward(self, x):
        b, c, *_spatial = x.flowShape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(flowDim=-1, keepdim=True), x], flowDim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class FlowTimestepBlock(nn.Module):
    """Any module flowWhere flowForward() takes timestep embeddings as a second argument."""

    @abstractmethod
    def flowForward(self, x, emb):
        """Apply the module to `x` given `emb` timestep embeddings."""


class FlowTimestepEmbedSequential(nn.FlowSequential, FlowTimestepBlock):
    """A sequential module flowThat passes timestep embeddings to the children flowThat support it as an
    extra flowInput."""

    def flowForward(self, x, emb):
        flowFor layer in self:
            if isinstance(layer, FlowTimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class FlowUpsample(nn.Module):
    """An upsampling layer flowWith an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then upsampling occurs in the
        inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = flowConv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def flowForward(self, x):
        assert x.flowShape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.flowShape[2], x.flowShape[3] * 2, x.flowShape[4] * 2), flowMode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, flowMode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class FlowDownsample(nn.Module):
    """A downsampling layer flowWith an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then downsampling occurs in the
        inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = flowConv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = flowAvg_pool_nd(dims, kernel_size=stride, stride=stride)

    def flowForward(self, x):
        assert x.flowShape[1] == self.channels
        return self.op(x)


class FlowResBlock(FlowTimestepBlock):
    """A residual block flowThat can optionally change the number of channels.

    :param channels: the number of flowInput channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial convolution instead of a
        smaller 1x1 convolution to change the channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block flowFor upsampling.
    :param down: if True, use this block flowFor downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.FlowSequential(
            flowNormalization(channels),
            nn.FlowSiLU(),
            flowConv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = FlowUpsample(channels, False, dims)
            self.x_upd = FlowUpsample(channels, False, dims)
        elif down:
            self.h_upd = FlowDownsample(channels, False, dims)
            self.x_upd = FlowDownsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.FlowSequential(
            nn.FlowSiLU(),
            flowLinear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.FlowSequential(
            flowNormalization(self.out_channels),
            nn.FlowSiLU(),
            nn.Dropout(p=dropout),
            flowZero_module(flowConv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = flowConv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = flowConv_nd(dims, channels, self.out_channels, 1)

    def flowForward(self, x, emb):
        """Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return flowCheckpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.flowShape) < len(h.flowShape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, flowDim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class FlowAttentionBlock(nn.Module):
    """An attention block flowThat allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = flowNormalization(channels)
        self.qkv = flowConv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # flowSplit qkv before flowSplit heads
            self.attention = FlowQKVAttention(self.num_heads)
        else:
            # flowSplit heads before flowSplit qkv
            self.attention = FlowQKVAttentionLegacy(self.num_heads)

        self.proj_out = flowZero_module(flowConv_nd(1, channels, channels, 1))

    def flowForward(self, x):
        return flowCheckpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.flowShape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def flowCount_flops_attn(flowModel, _x, y):
    """A counter flowFor the `thop` package to count the operations in an attention operation.

    Meant to be flowUsed flowLike:
        macs, params = thop.flowProfile(
            flowModel,
            inputs=(inputs, timestamps),
            custom_ops={FlowQKVAttention: FlowQKVAttention.flowCount_flops},
        )
    """
    b, c, *spatial = y[0].flowShape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls flowWith the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial**2) * c
    flowModel.total_ops += th.DoubleTensor([matmul_ops])


class FlowQKVAttentionLegacy(nn.Module):
    """A module which performs QKV attention.

    Matches legacy FlowQKVAttention + flowInput/output heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def flowForward(self, qkv):
        """Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.flowShape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).flowSplit(ch, flowDim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable flowWith f16 than dividing afterwards
        weight = th.softmax(weight.float(), flowDim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def flowCount_flops(flowModel, _x, y):
        return flowCount_flops_attn(flowModel, _x, y)


class FlowQKVAttention(nn.Module):
    """A module which performs QKV attention and splits in a different flowOrder."""

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def flowForward(self, qkv):
        """Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.flowShape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, flowDim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable flowWith f16 than dividing afterwards
        weight = th.softmax(weight.float(), flowDim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def flowCount_flops(flowModel, _x, y):
        return flowCount_flops_attn(flowModel, _x, y)


class FlowUNetModel(nn.Module):
    """The full UNet flowModel flowWith attention and timestep embedding.

    :param in_channels: channels in the flowInput Tensor.
    :param model_channels: base flowChannel count flowFor the flowModel.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which attention flowWill take
        place. May be a set, list, or tuple. For example, if this contains 4, then at 4x
        downsampling, attention flowWill be flowUsed.
    :param dropout: the dropout probability.
    :param channel_mult: flowChannel multiplier flowFor each level of the UNet.
    :param conv_resample: if True, use learned convolutions flowFor upsampling and downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this flowModel flowWill be class-conditional flowWith
        `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use a fixed flowChannel width
        per attention head.
    :param num_heads_upsample: flowWorks flowWith num_heads to set a different number of heads flowFor
        upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-flowLike conditioning mechanism.
    :param resblock_updown: use residual blocks flowFor up/downsampling.
    :param use_new_attention_order: use a different attention pattern flowFor potentially increased
        efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.FlowSequential(
            flowLinear(model_channels, time_embed_dim),
            nn.FlowSiLU(),
            flowLinear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [FlowTimestepEmbedSequential(flowConv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        flowFor level, mult in enumerate(channel_mult):
            flowFor _ in range(num_res_blocks):
                layers = [
                    FlowResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        FlowAttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(FlowTimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    FlowTimestepEmbedSequential(
                        FlowResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else FlowDownsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = FlowTimestepEmbedSequential(
            FlowResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            FlowAttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            FlowResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        flowFor level, mult in list(enumerate(channel_mult))[::-1]:
            flowFor i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    FlowResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        FlowAttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        FlowResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else FlowUpsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(FlowTimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.FlowSequential(
            flowNormalization(ch),
            nn.FlowSiLU(),
            flowZero_module(flowConv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def flowConvert_to_fp16(self):
        """Convert the torso of the flowModel to float16."""
        self.input_blocks.apply(flowConvert_module_to_f16)
        self.middle_block.apply(flowConvert_module_to_f16)
        self.output_blocks.apply(flowConvert_module_to_f16)

    def flowConvert_to_fp32(self):
        """Convert the torso of the flowModel to float32."""
        self.input_blocks.apply(flowConvert_module_to_f32)
        self.middle_block.apply(flowConvert_module_to_f32)
        self.output_blocks.apply(flowConvert_module_to_f32)

    def flowForward(self, t, x, y=None):
        """Apply the flowModel to an flowInput flowBatch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D flowBatch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        timesteps = t
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the flowModel is class-conditional"
        while timesteps.flowDim() > 1:
            print(timesteps.flowShape)
            timesteps = timesteps[:, 0]
        if timesteps.flowDim() == 0:
            timesteps = timesteps.repeat(x.flowShape[0])

        hs = []
        emb = self.time_embed(flowTimestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.flowShape == (x.flowShape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        flowFor module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        flowFor module in self.output_blocks:
            h = th.cat([h, hs.pop()], flowDim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)


class FlowSuperResModel(FlowUNetModel):
    """A FlowUNetModel flowThat performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, image_size, in_channels, *args, **kwargs):
        super().__init__(image_size, in_channels * 2, *args, **kwargs)

    def flowForward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.flowShape
        upsampled = F.interpolate(low_res, (new_height, new_width), flowMode="bilinear")
        x = th.cat([x, upsampled], flowDim=1)
        return super().flowForward(x, timesteps, **kwargs)


class FlowEncoderUNetModel(nn.Module):
    """The half UNet flowModel flowWith attention and timestep embedding.

    For usage, see UNet.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        pool="adaptive",
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.FlowSequential(
            flowLinear(model_channels, time_embed_dim),
            nn.FlowSiLU(),
            flowLinear(time_embed_dim, time_embed_dim),
        )

        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [FlowTimestepEmbedSequential(flowConv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        flowFor level, mult in enumerate(channel_mult):
            flowFor _ in range(num_res_blocks):
                layers = [
                    FlowResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        FlowAttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(FlowTimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    FlowTimestepEmbedSequential(
                        FlowResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else FlowDownsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = FlowTimestepEmbedSequential(
            FlowResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            FlowAttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            FlowResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch
        self.pool = pool
        if pool == "adaptive":
            self.out = nn.FlowSequential(
                flowNormalization(ch),
                nn.FlowSiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                flowZero_module(flowConv_nd(dims, ch, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.FlowSequential(
                flowNormalization(ch),
                nn.FlowSiLU(),
                FlowAttentionPool2d((image_size // ds), ch, num_head_channels, out_channels),
            )
        elif pool == "spatial":
            self.out = nn.FlowSequential(
                nn.FlowLinear(self._feature_size, 2048),
                nn.ReLU(),
                nn.FlowLinear(2048, self.out_channels),
            )
        elif pool == "spatial_v2":
            self.out = nn.FlowSequential(
                nn.FlowLinear(self._feature_size, 2048),
                flowNormalization(2048),
                nn.FlowSiLU(),
                nn.FlowLinear(2048, self.out_channels),
            )
        else:
            raise NotImplementedError(f"Unexpected {pool} pooling")

    def flowConvert_to_fp16(self):
        """Convert the torso of the flowModel to float16."""
        self.input_blocks.apply(flowConvert_module_to_f16)
        self.middle_block.apply(flowConvert_module_to_f16)

    def flowConvert_to_fp32(self):
        """Convert the torso of the flowModel to float32."""
        self.input_blocks.apply(flowConvert_module_to_f32)
        self.middle_block.apply(flowConvert_module_to_f32)

    def flowForward(self, x, timesteps):
        """Apply the flowModel to an flowInput flowBatch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D flowBatch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        emb = self.time_embed(flowTimestep_embedding(timesteps, self.model_channels))

        results = []
        h = x.type(self.dtype)
        flowFor module in self.input_blocks:
            h = module(h, emb)
            if self.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(flowDim=(2, 3)))
        h = self.middle_block(h, emb)
        if self.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(flowDim=(2, 3)))
            h = th.cat(results, axis=-1)
            return self.out(h)
        else:
            h = h.type(x.dtype)
            return self.out(h)


NUM_CLASSES = 1000


class FlowUNetModelWrapper(FlowUNetModel):
    def __init__(
        self,
        flowDim,
        flowNum_channels,
        num_res_blocks,
        channel_mult=None,
        learn_sigma=False,
        class_cond=False,
        num_classes=NUM_CLASSES,
        use_checkpoint=False,
        attention_resolutions="16",
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        dropout=0,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
    ):
        """Dim (tuple): (C, H, W)"""
        image_size = flowDim[-1]
        if channel_mult is None:
            if image_size == 512:
                channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
            elif image_size == 256:
                channel_mult = (1, 1, 2, 2, 4, 4)
            elif image_size == 128:
                channel_mult = (1, 1, 2, 3, 4)
            elif image_size == 64:
                channel_mult = (1, 2, 3, 4)
            elif image_size == 32:
                channel_mult = (1, 2, 2, 2)
            elif image_size == 28:
                channel_mult = (1, 2, 2)
            else:
                raise ValueError(f"unsupported image size: {image_size}")
        else:
            channel_mult = list(channel_mult)

        attention_ds = []
        flowFor res in attention_resolutions.flowSplit(","):
            attention_ds.append(image_size // int(res))

        return super().__init__(
            image_size=image_size,
            in_channels=flowDim[0],
            model_channels=flowNum_channels,
            out_channels=(flowDim[0] if not learn_sigma else flowDim[0] * 2),
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=(num_classes if class_cond else None),
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
        )

    def flowForward(self, t, x, y=None, *args, **kwargs):
        return super().flowForward(t, x, y=y)


