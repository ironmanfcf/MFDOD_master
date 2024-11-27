# Copyright (c) OpenMMLab. All rights reserved.
import random
from numbers import Number
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
try:
    import skimage
except ImportError:
    skimage = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmengine.dist import barrier, broadcast, get_dist_info
from mmengine.logging import MessageHub
from mmengine.model import BaseDataPreprocessor, ImgDataPreprocessor
from mmengine.structures import PixelData
from mmengine.utils import is_seq_of

from mmdet.models.utils import unfold_wo_center
from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks
from mmdet.utils import ConfigType

from typing import List, Dict
import torch
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine.structures import InstanceData, PixelData
from mfod.registry import MODELS
from mmdet.structures import DetDataSample

@MODELS.register_module()
class MultiDetDataPreprocessor(ImgDataPreprocessor):
    """Data preprocessor for multi-modal detection inputs.

    This class handles multiple types of images in the inputs, performing
    normalization, padding, and other preprocessing steps for each type of image.

    Args:
        pad_size_divisor (int): The divisor for padding size. Defaults to 32.
        mean (List[float]): The mean values for normalization. Defaults to [123.675, 116.28, 103.53].
        std (List[float]): The standard deviation values for normalization. Defaults to [58.395, 57.12, 57.375].
        to_rgb (bool): Whether to convert the image from BGR to RGB. Defaults to True.
    """

    def __init__(self,
                 mean: Sequence[float] = [123.675, 116.28, 103.53],
                 std: Sequence[float] = [58.395, 57.12, 57.375],
                 pad_size_divisor: int = 32,
                 pad_value: Union[float, int] = 0,
                 pad_mask: bool = False,
                 mask_pad_value: int = 0,
                 pad_seg: bool = False,
                 seg_pad_value: int = 255,
                 bgr_to_rgb: bool = True,
                 rgb_to_bgr: bool = False,
                 boxtype2tensor: bool = True,
                 non_blocking: Optional[bool] = False,
                 batch_augments: Optional[List[dict]] = None):
        super().__init__(
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            non_blocking=non_blocking)
        if batch_augments is not None:
            self.batch_augments = nn.ModuleList(
                [MODELS.build(aug) for aug in batch_augments])
        else:
            self.batch_augments = None
        self.pad_mask = pad_mask
        self.mask_pad_value = mask_pad_value
        self.pad_seg = pad_seg
        self.seg_pad_value = seg_pad_value
        self.boxtype2tensor = boxtype2tensor

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding and other preprocessing steps.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        batch_pad_shape = self._get_pad_shape(data)
        data = super().forward(data=data, training=training)
        inputs, data_samples = data['inputs'], data['data_samples']

        if data_samples is not None:
            batch_input_shape = tuple(inputs[0].size()[-2:])
            for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
                data_sample.set_metainfo({
                    'batch_input_shape': batch_input_shape,
                    'pad_shape': pad_shape
                })

            if self.boxtype2tensor:
                samplelist_boxtype2tensor(data_samples)

            if self.pad_mask and training:
                self.pad_gt_masks(data_samples)

            if self.pad_seg and training:
                self.pad_gt_sem_seg(data_samples)

        if training and self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)

        return {'inputs': inputs, 'data_samples': data_samples}

    def _get_pad_shape(self, data: dict) -> List[tuple]:
        """Get the pad_shape of each image based on data and pad_size_divisor."""
        _batch_inputs = data['inputs']
        batch_pad_shape = []

        for modality_inputs in _batch_inputs:
            if isinstance(modality_inputs, list) and all(isinstance(i, torch.Tensor) for i in modality_inputs):
                for ori_input in modality_inputs:
                    pad_h = int(np.ceil(ori_input.shape[1] / self.pad_size_divisor)) * self.pad_size_divisor
                    pad_w = int(np.ceil(ori_input.shape[2] / self.pad_size_divisor)) * self.pad_size_divisor
                    batch_pad_shape.append((pad_h, pad_w))
            elif isinstance(modality_inputs, torch.Tensor):
                assert modality_inputs.dim() == 4, (
                    'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                    'or a list of tensor, but got a tensor with shape: '
                    f'{modality_inputs.shape}')
                pad_h = int(np.ceil(modality_inputs.shape[2] / self.pad_size_divisor)) * self.pad_size_divisor
                pad_w = int(np.ceil(modality_inputs.shape[3] / self.pad_size_divisor)) * self.pad_size_divisor
                batch_pad_shape.extend([(pad_h, pad_w)] * modality_inputs.shape[0])
            else:
                raise TypeError('Output of `cast_data` should be a dict '
                                'or a tuple with inputs and data_samples, but got'
                                f'{type(data)}: {data}')
        return batch_pad_shape

    def pad_gt_masks(self, batch_data_samples: Sequence[DetDataSample]) -> None:
        """Pad gt_masks to shape of batch_input_shape."""
        if 'masks' in batch_data_samples[0].gt_instances:
            for data_samples in batch_data_samples:
                masks = data_samples.gt_instances.masks
                data_samples.gt_instances.masks = masks.pad(
                    data_samples.batch_input_shape,
                    pad_val=self.mask_pad_value)

    def pad_gt_sem_seg(self, batch_data_samples: Sequence[DetDataSample]) -> None:
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
        repr_str += f'(pad_size_divisor={self.pad_size_divisor}, mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str