# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import numpy as np
import torch

import cv2
from typing import Optional, Tuple, Union
import warnings

import mmcv
from mmcv.transforms import BaseTransform

import mmengine.fileio as fileio
from mmengine.fileio import get
from mmengine.structures import BaseDataElement

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import get_box_type
from mmdet.structures.bbox.box_type import autocast_box_type
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from mmcv.transforms import LoadImageFromFile as MMCV_LoadImageFromFile

from mfod.registry import TRANSFORMS

# Copyright (c) OpenMMLab. All rights reserved.



@TRANSFORMS.register_module()
class LoadPatchFromImage(MMCV_LoadImageFromFile):
    """Load an patch from the huge image.

    Similar with :obj:`LoadImageFromFile`, but only reserve a patch of
    ``results['img']`` according to ``results['win']``.
    """

    def __call__(self, results):
        """Call functions to add image meta information.

        Args:
            results (dict): Result dict with image in ``results['img']``.

        Returns:
            dict: The dict contains the loaded patch and meta information.
        """

        img = results['img']
        x_start, y_start, x_stop, y_stop = results['win']
        width = x_stop - x_start
        height = y_stop - y_start

        patch = img[y_start:y_stop, x_start:x_stop]
        if height > patch.shape[0] or width > patch.shape[1]:
            patch = mmcv.impad(patch, shape=(height, width))

        if self.to_float32:
            patch = patch.astype(np.float32)

        results['filename'] = None
        results['ori_filename'] = None
        results['img'] = patch
        results['img_shape'] = patch.shape
        results['ori_shape'] = patch.shape
        results['img_fields'] = ['img']
        return results

def bri_clr_loader1(data):
    img1 = cv2.cvtColor(data, cv2.COLOR_BGR2HSV) 
    # img1 = cv2.cvtColor(data, cv2.COLOR_RGB2YCrCb)
    color = img1[:, :, 0:2]
    brightness = img1[:, :, 2]
    return brightness[..., None], color

# @TRANSFORMS.register_module()
# class LoadImagePairFromFile(LoadImageFromFile):
#     """Load dualspectral image pair from two files.

#     Required keys are "img_prefix" and "img_info" (a dict that must contain the
#     key "filename"). Added or updated keys are "filename", "img", "img_shape",
#     "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
#     "scale_factor" (1.0) and "v/t_img_norm_cfg" (means=0 and stds=1).

#     Args:
#         spectrals (tuple/list): Names of folders denoting different spectrals.
#         to_float32 (bool): Whether to convert the loaded image to a float32
#             numpy array. If set to False, the loaded image is an uint8 array.
#             Defaults to False.
#         color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
#             Defaults to 'color'.
#         file_client_args (dict): Arguments to instantiate a FileClient.
#             See :class:`mmcv.fileio.FileClient` for details.
#             Defaults to ``dict(backend='disk')``.
#     """
#     def __init__(self,
#                  spectrals=('rgb', 'ir'),
#                  to_float32=False,
#                  color_type='color',
#                  channel_order='bgr',
#                  file_client_args=dict(backend='disk')):
#         self.to_float32 = to_float32
#         self.color_type = color_type
#         self.channel_order = channel_order
#         self.file_client_args = file_client_args.copy()
#         self.file_client = None
#         self.spectrals = spectrals
        
#     def __call__(self, results):
#         if self.file_client is None:
#             self.file_client = mmcv.FileClient(**self.file_client_args)

#         if results['img_prefix'] is not None:
#             filename1 = osp.join(results['img_prefix'], results['img_info']['filename'])
#             filename2 = osp.join(results['img_prefix'].replace(self.spectrals[0], self.spectrals[1], 1), 
#                                  results['img_info']['filename'])
#         else:
#             filename1 = results['img_info']['filename']
#             filename2 = results['img_info']['filename'].replace(self.spectrals[0], self.spectrals[1], 1)

#         img1_bytes = self.file_client.get(filename1)
#         img2_bytes = self.file_client.get(filename2)
#         img1 = mmcv.imfrombytes(img1_bytes, flag=self.color_type, channel_order=self.channel_order)
#         img2 = mmcv.imfrombytes(img2_bytes, flag=self.color_type, channel_order=self.channel_order)
#         if self.to_float32:
#             img1 = img1.astype(np.float32)
#             img2 = img2.astype(np.float32)

#         results['filename'] = (filename1, filename2)
#         results['ori_filename'] = results['img_info']['filename']
#         results['img1'] = img1
#         results['img2'] = img2
#         visimage_bri, visimage_clr = bri_clr_loader1(img1.copy())
#         results['visimage_bri'] = visimage_bri
#         results['visimage_clr'] = visimage_clr
#         results['F_rgb'] = img1
#         results['F_ir'] = img2
#         results['img_shape'] = img1.shape
#         results['ori_shape'] = img1.shape
        
#         results['img_fields'] = ['img1', 'img2', 'visimage_bri', 'visimage_clr', 'F_rgb', 'F_ir']
#         return results

#     def __repr__(self):
#         repr_str = (f'{self.__class__.__name__}('
#                     f'to_float32={self.to_float32}, '
#                     f'spectrals={self.spectrals},'
#                     f"color_type='{self.color_type}', "
#                     f'file_client_args={self.file_client_args})')
#         return repr_str
    
@TRANSFORMS.register_module()
class LoadImagePairFromFile(MMCV_LoadImageFromFile):
    """Load dualspectral image pair from two files.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "v/t_img_norm_cfg" (means=0 and stds=1).

    Args:
        spectrals (tuple/list): Names of folders denoting different spectrals.
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmengine.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """
    def __init__(self,
                 spectrals=('rgb', 'ir'),
                 to_float32=False,
                 color_type='color',
                 imdecode_backend='cv2',
                 file_client_args=None,
                 ignore_empty=False,
                 *,
                 backend_args=None):
        super().__init__(to_float32=to_float32,
                         color_type=color_type,
                         imdecode_backend=imdecode_backend,
                         file_client_args=file_client_args,
                         ignore_empty=ignore_empty,
                         backend_args=backend_args)
        self.spectrals = spectrals

    def transform(self, results):
        if self.file_client_args is not None:
            self.file_client = fileio.FileClient.infer_client(self.file_client_args, results['img_path'])
        else:
            self.file_client = None

        # if results['img_prefix'] is not None:
        #     filename1 = osp.join(results['img_prefix'], results['img_path'])
        #     filename2 = osp.join(results['img_prefix'].replace(self.spectrals[0], self.spectrals[1], 1), 
        #                          results['img_path'])
        # else:
        filename1 = results['img_path']
        filename2 = results['img_path'].replace(self.spectrals[0], self.spectrals[1], 1)#这里的 1 参数表示只替换第一个匹配项。

        img1_bytes = fileio.get(filename1, backend_args=self.backend_args) if self.file_client is None else self.file_client.get(filename1)
        img2_bytes = fileio.get(filename2, backend_args=self.backend_args) if self.file_client is None else self.file_client.get(filename2)
        img1 = mmcv.imfrombytes(img1_bytes, flag=self.color_type, backend=self.imdecode_backend)
        img2 = mmcv.imfrombytes(img2_bytes, flag=self.color_type, backend=self.imdecode_backend)
        if self.to_float32:
            img1 = img1.astype(np.float32)
            img2 = img2.astype(np.float32)

        results['filename'] = (filename1, filename2)
        results['ori_filename'] = results['img_path']
        results['img1'] = img1
        results['img2'] = img2
        visimage_bri, visimage_clr = bri_clr_loader1(img1.copy())
        results['visimage_bri'] = visimage_bri
        results['visimage_clr'] = visimage_clr
        results['F_rgb'] = img1
        results['F_ir'] = img2
        results['img_shape'] = img1.shape
        results['ori_shape'] = img1.shape
        
        results['img_fields'] = ['img1', 'img2', 'visimage_bri', 'visimage_clr', 'F_rgb', 'F_ir']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f'spectrals={self.spectrals},'
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', ")

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str


