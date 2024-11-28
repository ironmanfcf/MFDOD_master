# -*- encoding:utf-8 -*-
# !/usr/bin/env python


"""
authors: changfeng feng

Implementation of `C²Former: calibrated and complementary transformer for RGB-infrared object detection.`__

__ https://ieeexplore.ieee.org/document/10472947/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

import pdb
import warnings

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm


from mmcv.cnn import (build_conv_layer, build_norm_layer, build_plugin_layer)
from torch import nn as nn

from mfod.registry import MODELS
# from ..utils import ResLayer



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None):
        super(BasicBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None):
        """Bottleneck block for ResNet.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
            assert all(p['position'] in allowed_position for p in plugins)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None

        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            self.after_conv2_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv2'
            ]
            self.after_conv3_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv3'
            ]

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                dcn,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                planes * self.expansion, self.after_conv3_plugins)

    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(x)
        return out

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class ResLayer(nn.Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Default: True
    """

    def __init__(self,
                 block,
                 inplanes,
                 planes,
                 num_blocks,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 downsample_first=True,
                 **kwargs):
        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        if downsample_first:
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
            inplanes = planes * block.expansion
            for _ in range(1, num_blocks):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))

        else:  # downsample_first=False is for HourglassModule
            for _ in range(num_blocks - 1):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=inplanes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
        super(ResLayer, self).__init__(*layers)

class TwoStreamResNet(BaseModule):
    """TwoStreamResNet backbone.

        Please refer to the `paper <https://arxiv.org/abs/1512.03385>`_ for
        details.

        Args:
            depth (int): Network depth, from {18, 34, 50, 101, 152}.
            in_channels (int): Number of input image channels. Default: 3.
            stem_channels (int): Output channels of the stem layer. Default: 64.
            base_channels (int): Middle channels of the first stage. Default: 64.
            num_stages (int): Stages of the network. Default: 4.
            strides (Sequence[int]): Strides of the first block of each stage.
                Default: ``(1, 2, 2, 2)``.
            dilations (Sequence[int]): Dilation of each stage.
                Default: ``(1, 1, 1, 1)``.
            out_indices (Sequence[int]): Output from which stages. If only one
                stage is specified, a single tensor (feature map) is returned,
                otherwise multiple stages are specified, a tuple of tensors will
                be returned. Default: ``(3, )``.
            style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
                layer is the 3x3 conv layer, otherwise the stride-two layer is
                the first 1x1 conv layer.
            deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
                Default: False.
            avg_down (bool): Use AvgPool instead of stride conv when
                downsampling in the bottleneck. Default: False.
            frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
                -1 means not freezing any parameters. Default: -1.
            conv_cfg (dict | None): The config dict for conv layers. Default: None.
            norm_cfg (dict): The config dict for norm layers.
            norm_eval (bool): Whether to set norm layers to eval mode, namely,
                freeze running stats (mean and var). Note: Effect on Batch Norm
                and its variants only. Default: False.
            with_cp (bool): Use checkpoint or not. Using checkpoint will save some
                memory while slowing down the training speed. Default: False.
            zero_init_residual (bool): Whether to use zero init for last norm layer
                in resblocks to let them behave as identity. Default: True.
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True,
                 pretrained=None,
                 init_cfg=None
                 ):
        super(TwoStreamResNet, self).__init__()

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)

        #  layers
        self.vis_res_layers = []
        self.lwir_res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            planes = base_channels * 2 ** i
            vis_res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=stage_plugins)
            lwir_res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=stage_plugins)

            self.inplanes = planes * self.block.expansion
            layer_name = f'vis_layer{i + 1}'
            self.add_module(layer_name, vis_res_layer)
            self.vis_res_layers.append(layer_name)

            layer_name = f'lwir_layer{i + 1}'
            self.add_module(layer_name, lwir_res_layer)
            self.lwir_res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2**(
            len(self.stage_blocks) - 1)


    def make_stage_plugins(self, plugins, stage_idx):
        """Make plugins for ResNet ``stage_idx`` th stage.

        Currently we support to insert ``context_block``,
        ``empirical_attention_block``, ``nonlocal_block`` into the backbone
        like ResNet/ResNeXt. They could be inserted after conv1/conv2/conv3 of
        Bottleneck.

        An example of plugins format could be:

        Examples:
            >>> plugins=[
            ...     dict(cfg=dict(type='xxx', arg1='xxx'),
            ...          stages=(False, True, True, True),
            ...          position='after_conv2'),
            ...     dict(cfg=dict(type='yyy'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='1'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='2'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3')
            ... ]
            >>> self = ResNet(depth=18)
            >>> stage_plugins = self.make_stage_plugins(plugins, 0)
            >>> assert len(stage_plugins) == 3

        Suppose ``stage_idx=0``, the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->conv3->yyy->zzz1->zzz2

        Suppose 'stage_idx=1', the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->xxx->conv3->yyy->zzz1->zzz2

        If stages is missing, the plugin would be applied to all stages.

        Args:
            plugins (list[dict]): List of plugins cfg to build. The postfix is
                required if multiple same type plugins are inserted.
            stage_idx (int): Index of stage to build

        Returns:
            list[dict]: Plugins for current stage
        """
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            # whether to insert plugin into current stage
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.vis_stem.eval()
                self.lwir_stem.eval()
                for param in self.vis_stem.parameters():
                    param.requires_grad = False
                for param in self.lwir_stem.parameters():
                    param.requires_grad = False
            else:
                self.vis_norm1.eval()
                self.lwir_norm1.eval()
                for m in [self.vis_conv1, self.vis_norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

                for m in [self.lwir_conv1, self.lwir_norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            vis_m = getattr(self, f'vis_layer{i}')
            vis_m.eval()
            for param in vis_m.parameters():
                param.requires_grad = False
            lwir_m = getattr(self, f'lwir_layer{i}')
            lwir_m.eval()
            for param in lwir_m.parameters():
                param.requires_grad = False

    @property
    def vis_norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.vis_norm1_name)

    @property
    def lwir_norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.lwir_norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            # vis
            self.vis_stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True)
            )
            # lwir
            self.lwir_stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True)
            )
        else:
            # vis
            self.vis_conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.vis_norm1_name, vis_norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix='vis')
            self.add_module(self.vis_norm1_name, vis_norm1)
            # lwir
            self.lwir_conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.lwir_norm1_name, lwir_norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix='lwir')
            self.add_module(self.lwir_norm1_name, lwir_norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(TwoStreamResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


@MODELS.register_module()
class C2FormerResNet(TwoStreamResNet):
    def __init__(self,
                 fmap_size=(56, 56),
                 dims_in=[256, 512, 1024, 2048],
                 dims_out=[96, 192, 384, 768],
                 num_heads=[3, 6, 12, 24],
                 cca_strides=[-1, -1, -1, -1],
                 groups=[-1, -1, -1, -1],
                 offset_range_factor=[1, 2, 3, 4],
                 no_offs=[False, False, False, False],
                 attn_drop_rate=0.0,
                 drop_rate=0.0,
                 **kwargs
                 ):
        super(C2FormerResNet, self).__init__(**kwargs)

        self.num_heads = num_heads
        self.fmap_size = fmap_size
        # add C2Former
        self.c2formers = nn.ModuleList()
        self.vis_convlist1 = nn.ModuleList()
        self.lwir_convlist1 = nn.ModuleList()
        self.vis_convlist2 = nn.ModuleList()
        self.lwir_convlist2 = nn.ModuleList()
        for i in range(self.num_stages):
            hc = dims_out[i] // self.num_heads[i]
            self.c2formers.append(
                C2Former(self.fmap_size, self.fmap_size, self.num_heads[i],
                            hc, groups[i], attn_drop_rate, drop_rate,
                            cca_strides[i], offset_range_factor[i],
                            no_offs[i], i)
            )

            self.vis_convlist1.append(
                nn.Sequential(nn.Conv2d(dims_in[i], dims_out[i], (1, 1), (1, 1)), nn.ReLU()))
            self.lwir_convlist1.append(
                nn.Sequential(nn.Conv2d(dims_in[i], dims_out[i], (1, 1), (1, 1)), nn.ReLU()))
            self.vis_convlist2.append(
                nn.Sequential(nn.Conv2d(dims_out[i], dims_in[i], (1, 1), (1, 1)), nn.ReLU()))
            self.lwir_convlist2.append(
                nn.Sequential(nn.Conv2d(dims_out[i], dims_in[i], (1, 1), (1, 1)), nn.ReLU()))
            self.fmap_size = (self.fmap_size[0] // 2, self.fmap_size[1] // 2)
    # forward function
    def forward(self, vis_x, lwir_x):
        # resnet part
        if self.deep_stem:
            vis_x = self.vis_stem(vis_x)
            lwir_x = self.lwir_stem(lwir_x)
        else:
            vis_x = self.vis_conv1(vis_x)
            vis_x = self.vis_norm1(vis_x)
            vis_x = self.relu(vis_x)

            lwir_x = self.lwir_conv1(lwir_x)
            lwir_x = self.lwir_norm1(lwir_x)
            lwir_x = self.relu(lwir_x)
        vis_x = self.maxpool(vis_x)
        lwir_x = self.maxpool(lwir_x)

        outs = []
        for i in range(self.num_stages):
            # resnet
            vis_layer_name = self.vis_res_layers[i]
            vis_res_layer = getattr(self, vis_layer_name)
            vis_x = vis_res_layer(vis_x)

            lwir_layer_name = self.lwir_res_layers[i]
            lwir_res_layer = getattr(self, lwir_layer_name)
            lwir_x = lwir_res_layer(lwir_x)

            # c2former
            visinputconv = self.vis_convlist1[i]
            lwirinputconv = self.lwir_convlist1[i]

            visoutputconv = self.vis_convlist2[i]
            lwiroutputconv = self.lwir_convlist2[i]

            c2former = self.c2formers[i]

            input_vis_x = visinputconv(vis_x)
            input_lwir_x = lwirinputconv(lwir_x)

            out_vis, out_lwir = c2former(input_vis_x, input_lwir_x)
            out_vis = visoutputconv(out_vis)
            out_lwir = lwiroutputconv(out_lwir)
            vis_x = vis_x + out_lwir
            lwir_x = lwir_x + out_vis

            if i in self.out_indices:
                out = vis_x + lwir_x
                outs.append(out)

        return tuple(outs)

class C2Former(nn.Module):

    def __init__(
            self, q_size, kv_size, n_heads, n_head_channels, n_groups,
            attn_drop, proj_drop, stride,
            offset_range_factor,
            no_off, stage_idx
    ):

        super(C2Former,self).__init__()
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = kv_size
        self.nc = n_head_channels * n_heads
        self.qnc = n_head_channels * n_heads * 2
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor

        ksizes = [9, 7, 5, 3]
        kk = ksizes[stage_idx]

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, kk // 2, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )

        self.proj_q_lwir = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )
        self.proj_q_vis = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )
        self.proj_combinq = nn.Conv2d(
            self.qnc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k_lwir = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )
        self.proj_k_vis = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )
        self.proj_v_lwir = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )
        self.proj_v_vis = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out_lwir = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )
        self.proj_out_vis = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )
        self.vis_proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.lwir_proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.vis_attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.lwir_attn_drop = nn.Dropout(attn_drop, inplace=True)

        self.vis_MN = ModalityNorm(self.nc, use_residual=True, learnable=True)
        self.lwir_MN = ModalityNorm(self.nc, use_residual=True, learnable=True)

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self, vis_x, lwir_x):

        B, C, H, W = vis_x.size()
        dtype, device = vis_x.dtype, vis_x.device
        # concat two tensor
        x = torch.cat([vis_x,lwir_x],1)
        combin_q = self.proj_combinq(x)

        q_off = einops.rearrange(combin_q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off) 
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        vis_reference = self._get_ref_points(Hk, Wk, B, dtype, device)
        lwir_reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.no_off:
            offset = offset.fill(0.0)

        if self.offset_range_factor >= 0:
            vis_pos = vis_reference + offset
            lwir_pos = lwir_reference
        else:
            vis_pos = (vis_reference + offset).tanh()
            lwir_pos = lwir_reference.tanh()

        vis_x_sampled = F.grid_sample(
            input=vis_x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=vis_pos[..., (1, 0)],  
            mode='bilinear', align_corners=True)  

        lwir_x_sampled = F.grid_sample(
            input=lwir_x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=lwir_pos[..., (1, 0)],  
            mode='bilinear', align_corners=True)  

        vis_x_sampled = vis_x_sampled.reshape(B, C, 1, n_sample)
        lwir_x_sampled = lwir_x_sampled.reshape(B, C, 1, n_sample)
        
        q_lwir = self.proj_q_lwir(self.vis_MN(vis_x, lwir_x))
        q_lwir = q_lwir.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k_vis = self.proj_k_vis(vis_x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v_vis = self.proj_v_vis(vis_x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        q_vis = self.proj_q_vis(self.lwir_MN(lwir_x, vis_x))
        q_vis = q_vis.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k_lwir = self.proj_k_lwir(lwir_x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v_lwir = self.proj_v_lwir(lwir_x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        attn_vis = torch.einsum('b c m, b c n -> b m n', q_lwir, k_vis)  
        attn_vis = attn_vis.mul(self.scale)
        attn_vis = F.softmax(attn_vis, dim=2)
        attn_vis = self.vis_attn_drop(attn_vis)
        out_vis = torch.einsum('b m n, b c n -> b c m', attn_vis, v_vis)
        out_vis = out_vis.reshape(B, C, H, W)
        out_vis = self.vis_proj_drop(self.proj_out_vis(out_vis))

        attn_lwir = torch.einsum('b c m, b c n -> b m n', q_vis, k_lwir)  
        attn_lwir = attn_lwir.mul(self.scale)
        attn_lwir = F.softmax(attn_lwir, dim=2)
        attn_lwir = self.lwir_attn_drop(attn_lwir)
        out_lwir = torch.einsum('b m n, b c n -> b c m', attn_lwir, v_lwir)
        out_lwir = out_lwir.reshape(B, C, H, W)
        out_lwir = self.lwir_proj_drop(self.proj_out_lwir(out_lwir))

        return out_vis, out_lwir

class LayerNormProxy(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')

# Modality Norm
class ModalityNorm(nn.Module):
    def __init__(self, nf, use_residual=True, learnable=True):
        super(ModalityNorm, self).__init__()

        self.learnable = learnable
        self.norm_layer = nn.InstanceNorm2d(nf, affine=False)

        if self.learnable:
            self.conv= nn.Sequential(nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
                                             nn.ReLU(inplace=True))
            self.conv_gamma = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.conv_beta = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

            self.use_residual = use_residual

            # initialization
            self.conv_gamma.weight.data.zero_()
            self.conv_beta.weight.data.zero_()
            self.conv_gamma.bias.data.zero_()
            self.conv_beta.bias.data.zero_()

    def forward(self, lr, ref):
        ref_normed = self.norm_layer(ref)
        if self.learnable:
            x = self.conv(lr)
            gamma = self.conv_gamma(x)
            beta = self.conv_beta(x)

        b, c, h, w = lr.size()
        lr = lr.view(b, c, h * w)
        lr_mean = torch.mean(lr, dim=-1, keepdim=True).unsqueeze(3)
        lr_std = torch.std(lr, dim=-1, keepdim=True).unsqueeze(3)

        if self.learnable:
            if self.use_residual:
                gamma = gamma + lr_std
                beta = beta + lr_mean
            else:
                gamma = 1 + gamma
        else:
            gamma = lr_std
            beta = lr_mean

        out = ref_normed * gamma + beta

        return out
    

