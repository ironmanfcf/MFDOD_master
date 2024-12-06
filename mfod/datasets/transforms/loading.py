# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadImageFromFile
from mmdet.datasets.transforms import LoadAnnotations 
from mmengine.fileio import get
from mmengine.structures import BaseDataElement
import mmengine.fileio as fileio

from mmdet.structures.bbox import get_box_type
from mmdet.structures.bbox.box_type import autocast_box_type
from mmdet.structures.mask import BitmapMasks, PolygonMasks

from mfod.registry import TRANSFORMS


"""
@author: changfeng feng 
"""
@TRANSFORMS.register_module()
class LoadPairedImageFromFile(LoadImageFromFile):
    """Load an image from file.

    Required Keys:

    - img_path
    - img_ir_path

    Modified Keys:

    - img
    - img_ir
    - img_shape
    - ori_shape
    - img_shape_ir
    - ori_shape_ir

    """
    def transform(self, results):
        """Functions to load RGB image and LWIR image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        filename_ir = results['img_ir_path']
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename)
                img_bytes = file_client.get(filename)
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename_ir )
                img_lwir_bytes = file_client.get(filename_ir )
            else:
                img_bytes = fileio.get(
                    filename, backend_args=self.backend_args)
                img_lwir_bytes = fileio.get(
                    filename_ir, backend_args=self.backend_args)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            ir_img = mmcv.imfrombytes(
                img_lwir_bytes, flag=self.color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        assert img is not None, f'failed to load image: {filename}'
        assert ir_img is not None, f'failed to load IR image: {filename_ir}'

       
        if self.to_float32:
            img = img.astype(np.float32)
            ir_img = ir_img.astype(np.float32)

        results['img'] = img
        results['img_ir'] = ir_img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        results['img_shape_ir'] = ir_img.shape[:2]
        results['ori_shape_ir'] = ir_img.shape[:2]

        return results

"""
@author: changfeng feng
"""
@TRANSFORMS.register_module()
class LoadIRImageFromFile(LoadImageFromFile):
    """Load an image from file.

    Required Keys:

    - img_ir_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    """

    def transform(self, results):
        """Functions to load RGB image and LWIR image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        filename_ir = results['img_ir_path']
        try:
            if self.file_client_args is not None:

                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename_ir)
                img_lwir_bytes = file_client.get(filename_ir)
            else:
                img_lwir_bytes = fileio.get(
                    filename_ir, backend_args=self.backend_args)
            ir_img = mmcv.imfrombytes(
                img_lwir_bytes, flag=self.color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        assert ir_img is not None, f'failed to load LWIR image: {filename_ir}'

        if self.to_float32:
            ir_img = ir_img.astype(np.float32)

        results['img'] =  ir_img
        results['img_shape'] = ir_img.shape[:2]
        results['ori_shape'] = ir_img.shape[:2]
        return results
