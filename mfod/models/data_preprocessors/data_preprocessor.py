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

from mmdet.models.data_preprocessors import DetDataPreprocessor




@MODELS.register_module()
class PairedDetDataPreprocessor(DetDataPreprocessor):
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
        
        data = self.forward_ir(data=data, training=training)
        inputs_ir = data['inputs_ir']
        data = super().forward(data=data, training=training)
        
        inputs,  data_samples = data['inputs'], data['data_samples']        
        if training and self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs_ir, data_samples = batch_aug(inputs_ir, data_samples)

        return {'inputs': inputs, 'inputs_ir': inputs_ir, 'data_samples': data_samples}

    def forward_ir(self, data: dict, training: bool = False) -> dict:
        """Perform normalization,padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        data = self.cast_data(data)  # type: ignore
        batch_pad_shape = self._get_pad_shape_ir(data)
        # data = super().forward(data=data, training=training)
        # data = self.cast_data(data)  # type: ignore
        _batch_inputs_ir , _batch_input, data_samples = data['inputs_ir'],data['inputs'], data['data_samples']
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs_ir, torch.Tensor):
            batch_inputs_ir = []
            for _batch_input_ir in _batch_inputs_ir:
                # channel transform
                if self._channel_conversion:
                    _batch_input_ir = _batch_input_ir[[2, 1, 0], ...]
                # Convert to float after channel conversion to ensure
                # efficiency
                _batch_input_ir = _batch_input_ir.float()
                # Normalization.
                if self._enable_normalize_ir:
                    if self.mean_ir.shape[0] == 3:
                        assert _batch_input_ir.dim(
                        ) == 3 and _batch_input_ir.shape[0] == 3, (
                            'If the mean has 3 values, the input tensor '
                            'should in shape of (3, H, W), but got the tensor '
                            f'with shape {_batch_input_ir.shape}')
                    _batch_input_ir = (_batch_input_ir - self.mean_ir) / self.std_ir
                batch_inputs_ir.append(_batch_input_ir)
            # Pad and stack Tensor.
            batch_inputs_ir = stack_batch(batch_inputs_ir, self.pad_size_divisor,
                                       self.pad_value)
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs_ir, torch.Tensor):
            assert _batch_inputs_ir.dim() == 4, (
                'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_inputs_ir.shape}')
            if self._channel_conversion:
                _batch_inputs_ir = _batch_inputs_ir[:, [2, 1, 0], ...]
            # Convert to float after channel conversion to ensure
            # efficiency
            _batch_inputs_ir = _batch_inputs_ir.float()
            if self._enable_normalize:
                _batch_inputs_ir = (_batch_inputs_ir - self.mean_ir) / self.std_ir
            h, w = _batch_inputs_ir.shape[2:]
            target_h = math.ceil(
                h / self.pad_size_divisor) * self.pad_size_divisor
            target_w = math.ceil(
                w / self.pad_size_divisor) * self.pad_size_divisor
            pad_h = target_h - h
            pad_w = target_w - w
            batch_inputs_ir = F.pad(_batch_inputs_ir, (0, pad_w, 0, pad_h),
                                 'constant', self.pad_value)
        else:
            raise TypeError('Output of `cast_data` should be a dict of '
                            'list/tuple with inputs and data_samples, '
                            f'but got {type(data)}: {data}')
            
        if data_samples is not None:
            # NOTE the batched image size information may be useful, e.g.
            # in DETR, this is needed for the construction of masks, which is
            # then used for the transformer_head.
            batch_input_shape = tuple(batch_inputs_ir[0].size()[-2:])
            for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
                data_sample.set_metainfo({
                    'batch_input_shape_ir': batch_input_shape,
                    'pad_shape_ir': pad_shape
                })

        return {'inputs':  _batch_input, 'inputs_ir': batch_inputs_ir, 'data_samples': data_samples}


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
    
    def _get_pad_shape_ir(self, data: dict) -> List[tuple]:
        """Get the pad_shape of each image based on data and
        pad_size_divisor."""
        _batch_inputs = data['inputs_ir']
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