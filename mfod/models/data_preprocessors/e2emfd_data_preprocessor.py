# Copyright (c) OpenMMLab. All rights reserved.
import random
from numbers import Number
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.dist import barrier, broadcast, get_dist_info
from mmengine.logging import MessageHub
from mmengine.model import BaseDataPreprocessor, ImgDataPreprocessor
from mmengine.structures import PixelData
from mmengine.utils import is_seq_of
from torch import Tensor

from mmdet.models.utils import unfold_wo_center
from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mfod.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks
from mmdet.utils import ConfigType
import math 

try:
    import skimage
except ImportError:
    skimage = None

from mmengine.utils import is_seq_of
from mmengine.model.utils import stack_batch
import cv2

from mmdet.models.data_preprocessors import DetDataPreprocessor
"""

"""


        # for key in results.get('img_fields', ['img']):
        #     if key == 'img1' or key == 'img2':
        #         results[key] = mmcv.imnormalize(results[key], self.mean, self.std,
        #                                         self.to_rgb)
        # results['img_norm_cfg'] = dict(
        #     mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        
        # results['F_ir'] = results['F_ir']/255
        # results['F_rgb'] = results['F_rgb']/255
        # results['visimage_bri'] = results['visimage_bri']/255
        # return results


@MODELS.register_module()
class E2EMFDDataPreprocessor(DetDataPreprocessor):
    """Data preprocessor for multi-modal detection inputs.

    This class handles multiple types of images in the inputs, performing
    normalization, padding, and other preprocessing steps for each type of image.

    Args:
        mean_ir (Sequence[Number], optional): The pixel mean of R, G, B channels for IR images.
            Defaults to None.
        std_ir (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels for IR images. Defaults to None.
    """

    def __init__(self,
                 mean_ir: Sequence[float] = None,
                 std_ir: Sequence[float] = None,
                 **kwargs):
        super().__init__(**kwargs)

        if mean_ir is not None:
            assert len(mean_ir) == 3 or len(mean_ir) == 1, (
                '`mean` should have 1 or 3 values, to be compatible with '
                f'RGB or gray image, but got {len(mean_ir)} values')
            assert len(std_ir) == 3 or len(std_ir) == 1, (  # type: ignore
                '`std` should have 1 or 3 values, to be compatible with RGB '  # type: ignore # noqa: E501
                f'or gray image, but got {len(std_ir)} values')  # type: ignore
            self._enable_normalize_ir = True
            self.register_buffer('mean_ir',
                                 torch.tensor(mean_ir).view(-1, 1, 1), False)
            self.register_buffer('std_ir',
                                 torch.tensor(std_ir).view(-1, 1, 1), False)

        else:
            self._enable_normalize_ir = False
  
    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding and other preprocessing steps.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.

        """

 
        inputs_vis_bri, inputs_vis_clr, inputs_F_vis = self.forward_rgb(data=data, training=training)
        data['visimage_bri'], data['visimage_clr'], data['inputs_F_vis'] = inputs_vis_bri, inputs_vis_clr, inputs_F_vis       
        
        inputs_ir, inputs_F_ir = self.forward_ir(data=data, training=training)
        data['inputs_ir'], data['inputs_F_ir'] =inputs_ir, inputs_F_ir 
        data = super().forward(data=data, training=training)
        
        inputs,  data_samples = data['inputs'], data['data_samples']        
        if training and self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs_ir, data_samples = batch_aug(inputs_ir, data_samples)
        # 确保所有输出在同一设备上
        device = inputs.device
        inputs_ir = inputs_ir.to(device)
        inputs_vis_bri = inputs_vis_bri.to(device)
        inputs_vis_clr = inputs_vis_clr.to(device)
        inputs_F_vis = inputs_F_vis.to(device)
        inputs_F_ir = inputs_F_ir.to(device)        

        return {'inputs': inputs, 'inputs_ir': inputs_ir, 'inputs_vis_bri': inputs_vis_bri, \
            'inputs_vis_clr':inputs_vis_clr, 'inputs_F_vis': inputs_F_vis,'inputs_F_ir': inputs_F_ir,\
                'data_samples': data_samples}
    
    def forward_rgb(self, data: dict, training: bool = False) -> dict:
        """Process visible light images in the data.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data with processed visible light images.
        """
        data = self.cast_data(data)
        # 从 data 中读取可见光图像
        _batch_inputs = data['inputs']

        # 初始化存储亮度图像、颜色图像和特征图像的列表
        inputs_vis_bri = []
        inputs_vis_clr = []
        inputs_F_vis = []

        # 处理每个图像
        if is_seq_of(_batch_inputs, torch.Tensor):
            for _batch_input in _batch_inputs:
                # 获取亮度图像和颜色图像
                img_bri, img_color = self.bri_clr_loader1(_batch_input)
                img_bri = img_bri.float() / 255.0
                img_color = img_color.float()

                # 将亮度图像和颜色图像从 HWC 格式转换为 CHW 格式
                img_bri = img_bri.permute(2, 0, 1)
                img_color = img_color.permute(2, 0, 1)

                # 复制特征图像
                F_rgb = _batch_input.float() / 255.0

                # 添加到列表中
                inputs_vis_bri.append(img_bri)
                inputs_vis_clr.append(img_color)
                inputs_F_vis.append(F_rgb)

            # 使用 stack_batch 对图像进行堆叠和填充
            inputs_vis_bri = stack_batch(inputs_vis_bri, self.pad_size_divisor, self.pad_value)
            inputs_vis_clr = stack_batch(inputs_vis_clr, self.pad_size_divisor, self.pad_value)
            inputs_F_vis = stack_batch(inputs_F_vis, self.pad_size_divisor, self.pad_value)

        else:
            raise TypeError('Output of `cast_data` should be a dict of '
                            'list/tuple with inputs and data_samples, '
                            f'but got {type(data)}: {data}')

        return inputs_vis_bri, inputs_vis_clr, inputs_F_vis
   
    def forward_ir(self, data: dict, training: bool = False) -> Tuple[Tensor, Tensor]:
        """Perform normalization, padding and other preprocessing steps for IR images.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Tuple[Tensor, Tensor]: Original batch inputs and processed batch inputs.
        """
        data = self.cast_data(data)
        _batch_inputs_ir = data['inputs_ir']
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs_ir, torch.Tensor):
            batch_inputs_ir = []
            processed_inputs_ir = []
            for _batch_input_ir in _batch_inputs_ir:
                # 复制并除以255获得F_ir
                processed_input_ir = _batch_input_ir.clone() / 255.0
                processed_inputs_ir.append(processed_input_ir)
                # channel transform
                if self._channel_conversion:
                    _batch_input_ir = _batch_input_ir[[2, 1, 0], ...]
                # Convert to float after channel conversion to ensure efficiency
                _batch_input_ir = _batch_input_ir.float()
                # Normalization.
                if self._enable_normalize_ir:
                    if self.mean_ir.shape[0] == 3:
                        assert _batch_input_ir.dim() == 3 and _batch_input_ir.shape[0] == 3, (
                            'If the mean has 3 values, the input tensor should be in shape of (3, H, W), '
                            f'but got the tensor with shape {_batch_input_ir.shape}')
                    _batch_input_ir = (_batch_input_ir - self.mean_ir) / self.std_ir
                batch_inputs_ir.append(_batch_input_ir)
            # Pad and stack Tensor.
            batch_inputs_ir = stack_batch(batch_inputs_ir, self.pad_size_divisor, self.pad_value)
            processed_inputs_ir = stack_batch(processed_inputs_ir, self.pad_size_divisor, self.pad_value)
        # Process data with `default_collate`.

        else:
            raise TypeError('Output of `cast_data` should be a dict of list/tuple with inputs and data_samples, '
                            f'but got {type(data)}: {data}')

        return batch_inputs_ir, processed_inputs_ir


    # def bri_clr_loader1(data):
    #     """
    #     Load brightness and color channel of input image.

    #     Args:
    #         data : Image data in RGB format.

    #     Returns:
    #         Tuple: Brightness and color channels of the input image.
    #     """
    #     img1 = cv2.cvtColor(data, cv2.COLOR_BGR2HSV) 
    #     # img1 = cv2.cvtColor(data, cv2.COLOR_RGB2YCrCb)
    #     color = img1[:, :, 0:2]
    #     brightness = img1[:, :, 2]  
        
    #     return brightness[..., None], color

    def bri_clr_loader1(self, data:Tensor) -> np.ndarray: 
        """
        Load brightness and color channel of input image.

        #这里与原代码不太一致，因为在packdetdata时data中的数据已被转换
        # 为tensor无法直接使用cvtcolor，需要进行一步转换
        
        Args:
            data : Image data in CHW format Tensor.

        Returns:
            Tuple: Brightness and color channels of the input image.
        """

        # 将 CHW 格式的 Tensor 转换为 HWC 格式的 NumPy 数组
        data = data.permute(1, 2, 0).cpu().numpy()

        # 使用 OpenCV 进行颜色空间转换
        img1 = cv2.cvtColor(data, cv2.COLOR_RGB2HSV)

        # 提取颜色通道和亮度通道
        color = img1[:, :, 0:2]
        brightness = img1[:, :, 2]

        # 将亮度和颜色通道转换为 Tensor
        brightness = torch.tensor(brightness[..., None])
        color = torch.tensor(color)

        return brightness, color


    def _normalize(self, img: np.ndarray, mean: List[float], std: List[float]) -> np.ndarray:
        """Normalize the image."""
        img = img.astype(np.float32)
        img = (img - mean) / std
        if self.to_rgb:
            img = img[..., ::-1]
        return img

    def _pad(self, img: np.ndarray, pad_shape: tuple) -> np.ndarray:
        """Pad the image to the specified shape."""
        pad_h, pad_w = pad_shape
        padded_img = np.zeros((pad_h, pad_w, img.shape[2]), dtype=img.dtype)
        padded_img[:img.shape[0], :img.shape[1], ...] = img
        return padded_img

    def _get_pad_shape(self, data: dict) -> List[tuple]:
        """Get the pad_shape of each image based on data and
        pad_size_divisor."""
        _batch_inputs = data['inputs']
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_pad_shape = []
            for ori_input in _batch_inputs:
                pad_h = int(
                    np.ceil(ori_input.shape[1] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                pad_w = int(
                    np.ceil(ori_input.shape[2] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                batch_pad_shape.append((pad_h, pad_w))
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_inputs.shape}')
            pad_h = int(
                np.ceil(_batch_inputs.shape[2] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            pad_w = int(
                np.ceil(_batch_inputs.shape[3] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            batch_pad_shape = [(pad_h, pad_w)] * _batch_inputs.shape[0]
        else:
            raise TypeError('Output of `cast_data` should be a dict '
                            'or a tuple with inputs and data_samples, but got'
                            f'{type(data)}: {data}')
        return batch_pad_shape
    

    def pad_gt_masks(self,
                     batch_data_samples: Sequence[DetDataSample]) -> None:
        """Pad gt_masks to shape of batch_input_shape."""
        if 'masks' in batch_data_samples[0].gt_instances:
            for data_samples in batch_data_samples:
                masks = data_samples.gt_instances.masks
                data_samples.gt_instances.masks = masks.pad(
                    data_samples.batch_input_shape,
                    pad_val=self.mask_pad_value)

    def pad_gt_sem_seg(self,
                       batch_data_samples: Sequence[DetDataSample]) -> None:
        """Pad gt_sem_seg to shape of batch_input_shape."""
        if 'gt_sem_seg' in batch_data_samples[0]:
            for data_samples in batch_data_samples:
                gt_sem_seg = data_samples.gt_sem_seg.sem_seg
                h, w = gt_sem_seg.shape[-2:]
                pad_h, pad_w = data_samples.batch_input_shape
                gt_sem_seg = F.pad(
                    gt_sem_seg,
                    pad=(0, max(pad_w - w, 0), 0, max(pad_h - h, 0)),
                    mode='constant',
                    value=self.seg_pad_value)
                data_samples.gt_sem_seg = PixelData(sem_seg=gt_sem_seg)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, mean_ir={self.mean_ir}, std_ir={self.std_ir}, '
        repr_str += f'pad_size_divisor={self.pad_size_divisor}, pad_value={self.pad_value}, '
        repr_str += f'pad_mask={self.pad_mask}, mask_pad_value={self.mask_pad_value}, '
        repr_str += f'pad_seg={self.pad_seg}, seg_pad_value={self.seg_pad_value}, '
        repr_str += f'bgr_to_rgb={self.bgr_to_rgb}, rgb_to_bgr={self.rgb_to_bgr}, '
        repr_str += f'boxtype2tensor={self.boxtype2tensor}, non_blocking={self.non_blocking}, '
        repr_str += f'batch_augments={self.batch_augments})'
        return repr_str