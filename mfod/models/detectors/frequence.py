import copy
import warnings
import math
import time
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_


from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Union

import torch
from mmengine.model import BaseModel
from torch import Tensor

from mmdet.structures import DetDataSample, OptSampleList, SampleList
from mmdet.utils import InstanceList, OptConfigType, OptMultiConfig


import matplotlib.pyplot as plt
ForwardResults = Union[Dict[str, torch.Tensor], List[DetDataSample],
                       Tuple[torch.Tensor], torch.Tensor]

from mfod.registry import MODELS

# from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
# from .base import RotatedBaseDetector

from pytorch_wavelets import DWTForward, DWTInverse


from .two_stage_ts import Two_Stage_TS


@MODELS.register_module()
class FrequenceDet(Two_Stage_TS):
    """Base class for rotated two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """
    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg=None):
        super(FrequenceDet, self).__init__(backbone=backbone,
                                     neck=neck,
                                     rpn_head=rpn_head,
                                     roi_head=roi_head,
                                     train_cfg=train_cfg,
                                     test_cfg=test_cfg,
                                     data_preprocessor=data_preprocessor,
                                     init_cfg=init_cfg)
        
        # 初始化 DWTTransformerattentionlayer 模块
        self.dwt_transformer_attention = DWTTransformerattentionlayer(channels=neck.out_channels)

    def extract_feat(self, batch_inputs: Tensor,
                     batch_inputs_ir: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).
            batch_inputs_ir (Tensor): Input images of shape (N, C, H, W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs)
        x_ir = self.backbone(batch_inputs_ir)
       
        if self.with_neck:
            x = self.neck(x)
            x_ir = self.neck(x_ir)

        # 对每个层级的特征进行处理
        processed_feats = []
        for feat, feat_ir in zip(x, x_ir):
            # 使用 DWTTransformerattentionlayer 进行特征处理
            processed_feat = self.dwt_transformer_attention(feat, feat_ir)
            processed_feats.append(processed_feat)

        return tuple(processed_feats)
    
class DWTTransformerattentionlayer(nn.Module):

    def __init__(self, channels):
        super(DWTTransformerattentionlayer, self).__init__()
        self.trans_LL_HH = CBAM(in_channels=channels)
        self.trans_HH_LL = CBAM(in_channels=channels)
        self.trans_LH_HL = CBAM(in_channels=channels)
        self.trans_HL_LH = CBAM(in_channels=channels)
        self.wavelet = WaveletTransform(wavelet='haar')

    def forward(self, x_vis, x_ir):
        # 对可见光特征进行小波变换
        yl_vis, yh_vis = self.wavelet.dwt2_transform(x_vis)
        LL_vis, LH_vis, HL_vis, HH_vis = yl_vis, yh_vis[0][:, :, 0, :, :], yh_vis[0][:, :, 1, :, :], yh_vis[0][:, :, 2, :, :]

        # 对红外特征进行小波变换
        yl_ir, yh_ir = self.wavelet.dwt2_transform(x_ir)
        LL_ir, LH_ir, HL_ir, HH_ir = yl_ir, yh_ir[0][:, :, 0, :, :], yh_ir[0][:, :, 1, :, :], yh_ir[0][:, :, 2, :, :]

        # 对不同模态间的分量进行 CBAM 调制
        LL_vis, HH_ir = self.trans_LL_HH(LL_vis, HH_ir)
        HH_vis, LL_ir = self.trans_HH_LL(HH_vis, LL_ir)
        LH_vis, HL_ir = self.trans_LH_HL(LH_vis, HL_ir)
        HL_vis, LH_ir = self.trans_HL_LH(HL_vis, LH_ir)

        # 重建处理过后的分量
        yl_vis = LL_vis
        new_yh_vis = torch.cat([LH_vis.unsqueeze(2), HL_vis.unsqueeze(2), HH_vis.unsqueeze(2)], dim=2)
        yh_vis[0] = new_yh_vis

        yl_ir = LL_ir
        new_yh_ir = torch.cat([LH_ir.unsqueeze(2), HL_ir.unsqueeze(2), HH_ir.unsqueeze(2)], dim=2)
        yh_ir[0] = new_yh_ir

        # 逆小波变换重建特征
        processed_feat_vis = self.wavelet.idwt2_transform(yl_vis, yh_vis)
        processed_feat_ir = self.wavelet.idwt2_transform(yl_ir, yh_ir)

        return (processed_feat_vis + processed_feat_ir) / 2  
    
    
class WaveletTransform(nn.Module):
    def __init__(self, wavelet='haar'):
        """
        初始化小波变换类。

        参数：
        wavelet: 使用的小波函数类型，默认为 'haar'
        """
        super(WaveletTransform, self).__init__()
        self.wavelet = wavelet
        self.dwt = DWTForward(J=1, wave=wavelet, mode='zero').cuda()  # 使用GPU加速的DWT
        self.idwt = DWTInverse(wave=wavelet, mode='zero').cuda()  # 使用GPU加速的IDWT

    def dwt2_transform(self, x):
        """
        对形状为 [b, c, h, w] 的张量进行离散小波变换，并将结果存储为四个张量。

        参数：
        x: 输入张量，形状为 [b, c, h, w]

        返回：
        四个张量，分别为逼近系数、水平细节系数、垂直细节系数和对角线细节系数
        """
        # 直接在 GPU 上进行小波变换
        yl, yh = self.dwt(x)

        return yl, yh

    def idwt2_transform(self, yl, yh):
        """
        对离散小波变换系数进行反变换，恢复到原始空间域。

        参数：
        cA_tensor: 逼近系数张量
        cH_tensor: 水平细节系数张量
        cV_tensor: 垂直细节系数张量
        cD_tensor: 对角线细节系数张量

        返回：
        一个张量，形状为 [b, c, h * 2, w * 2]
        """
        img_recon = self.idwt((yl, yh))  # 进行逆小波变换

        return img_recon


# 通道注意力模块 (Channel Attention Module, CAM)
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化

        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)  # 第一个全连接层
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)  # 第二个全连接层

        self.sigmoid = nn.Sigmoid()

    def forward(self, A):
        # 全局平均池化和最大池化，分别得到不同的特征表示
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(A))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(A))))

        # 将两者相加并经过 Sigmoid 函数生成通道注意力权重
        out = self.sigmoid(avg_out + max_out)

        return out


# 空间注意力模块 (Spatial Attention Module, SAM)
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)  # 7x7 卷积生成空间注意力

        self.sigmoid = nn.Sigmoid()

    def forward(self, A):
        # 通过通道维度的平均池化和最大池化来获得空间特征
        avg_out = torch.mean(A, dim=1, keepdim=True)
        max_out, _ = torch.max(A, dim=1, keepdim=True)

        # 将两者拼接 (B, 2, H, W) 作为输入，通过 7x7 卷积生成空间权重
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))

        return out


# CBAM 模块，将通道和空间注意力结合
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention1 = ChannelAttention(in_channels, reduction)
        # self.spatial_attention1 = SpatialAttention()
        self.channel_attention2 = ChannelAttention(in_channels, reduction)
        # self.spatial_attention2 = SpatialAttention()

    def forward(self, A, B):
        channel_attention_a = self.channel_attention1(A)
        # spatial_attention_a = self.spatial_attention1(A)
        channel_attention_b = self.channel_attention2(B)
        # spatial_attention_b = self.spatial_attention2(B)

        # channel_attention_a = self.channel_attention1(A)
        # channel_attention_A = channel_attention_a * A
        # spatial_attention_a = self.spatial_attention1(channel_attention_A)
        # channel_attention_b = self.channel_attention2(B)
        # channel_attention_B = channel_attention_b * B
        # spatial_attention_b = self.spatial_attention2(channel_attention_B)

        A = A * channel_attention_b
        # A = A * spatial_attention_b
        B = B * channel_attention_a  # 对每个通道进行加权
        # B = B * spatial_attention_a  # 对每个空间位置进行加权

        return A, B


