# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Optional, Tuple, Union
import warnings
import mmengine.fileio as fileio

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
from mmcv.transforms import LoadImageFromFile
from mmengine.fileio import get
from mmengine.structures import BaseDataElement

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import get_box_type
from mmdet.structures.bbox.box_type import autocast_box_type
from mmdet.structures.mask import BitmapMasks, PolygonMasks

# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
from mmdet.datasets.pipelines import LoadImageFromFile

from ..builder import ROTATED_PIPELINES
import cv2


@ROTATED_PIPELINES.register_module()
class LoadPatchFromImage(LoadImageFromFile):
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

@ROTATED_PIPELINES.register_module()
class LoadImagePairFromFile(LoadImageFromFile):
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
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """
    def __init__(self,
                 spectrals=('rgb', 'ir'),
                 to_float32=False,
                 color_type='color',
                 channel_order='bgr',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.spectrals = spectrals
        
    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results['img_prefix'] is not None:
            filename1 = osp.join(results['img_prefix'], results['img_info']['filename'])
            filename2 = osp.join(results['img_prefix'].replace(self.spectrals[0], self.spectrals[1], 1), 
                                 results['img_info']['filename'])
        else:
            filename1 = results['img_info']['filename']
            filename2 = results['img_info']['filename'].replace(self.spectrals[0], self.spectrals[1], 1)

        img1_bytes = self.file_client.get(filename1)
        img2_bytes = self.file_client.get(filename2)
        img1 = mmcv.imfrombytes(img1_bytes, flag=self.color_type, channel_order=self.channel_order)
        img2 = mmcv.imfrombytes(img2_bytes, flag=self.color_type, channel_order=self.channel_order)
        if self.to_float32:
            img1 = img1.astype(np.float32)
            img2 = img2.astype(np.float32)

        results['filename'] = (filename1, filename2)
        results['ori_filename'] = results['img_info']['filename']
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
                    f'file_client_args={self.file_client_args})')
        return repr_str


@TRANSFORMS.register_module()
class LoadDualAnnotations(MMCV_LoadAnnotations):
    """Load and process the ``instances`` and ``seg_map`` annotation provided by dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            'instances': [
                {
                    'bbox': [x1, y1, x2, y2],
                    'bbox_label': 1,
                    'mask': list[list[float]] or dict,
                }
            ],
            'seg_map_path': 'a/b/c'
        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {
            'gt_bboxes': BaseBoxes(N, 4),
            'gt_bboxes_labels': np.ndarray(N, ),
            'gt_masks': PolygonMasks (H, W) or BitmapMasks (H, W),
            'gt_seg_map': np.ndarray (H, W),
        }

    Required Keys:
    - height
    - width
    - instances
      - bbox (optional)
      - bbox_label
      - mask (optional)
      - ignore_flag
    - seg_map_path (optional)

    Added Keys:
    - gt_bboxes (BaseBoxes[torch.float32])
    - gt_bboxes_labels (np.int64)
    - gt_masks (BitmapMasks | PolygonMasks)
    - gt_seg_map (np.uint8)
    - gt_ignore_flags (bool)

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation. Defaults to True.
        with_label (bool): Whether to parse and load the label annotation. Defaults to True.
        with_mask (bool): Whether to parse and load the mask annotation. Default: False.
        with_seg (bool): Whether to parse and load the semantic segmentation annotation. Defaults to False.
        poly2mask (bool): Whether to convert mask to bitmap. Default: True.
        box_type (str): The box type used to wrap the bboxes. If ``box_type`` is None, gt_bboxes will keep being np.ndarray. Defaults to 'hbox'.
        reduce_zero_label (bool): Whether reduce all label value by 1. Usually used for datasets where 0 is background label. Defaults to False.
        ignore_index (int): The label index to be ignored. Valid only if reduce_zero_label is true. Defaults is 255.
        imdecode_backend (str): The image decoding backend type. The backend argument for :func:``mmcv.imfrombytes``. See :fun:``mmcv.imfrombytes`` for details. Defaults to 'cv2'.
        backend_args (dict, optional): Arguments to instantiate the corresponding backend. Defaults to None.
    """

    def __init__(
            self,
            with_mask: bool = False,
            poly2mask: bool = True,
            box_type: str = 'hbox',
            reduce_zero_label: bool = False,
            ignore_index: int = 255,
            max_depth_path: str = None,
            **kwargs) -> None:
        super(LoadDualAnnotations, self).__init__(**kwargs)
        self.with_mask = with_mask
        self.poly2mask = poly2mask
        self.box_type = box_type
        self.reduce_zero_label = reduce_zero_label
        self.ignore_index = ignore_index
        self.max_depth_path = max_depth_path
    def _load_bboxes(self, results: dict) -> None:
        """Private function to load bounding box annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        """
        gt_bboxes = []
        gt_ignore_flags = []
        for instance in results.get('instances', []):
            gt_bboxes.append(instance['bbox'])
            gt_ignore_flags.append(instance['ignore_flag'])
        if self.box_type is None:
            results['gt_bboxes'] = np.array(gt_bboxes, dtype=np.float32).reshape((-1, 4))
        else:
            _, box_type_cls = get_box_type(self.box_type)
            results['gt_bboxes'] = box_type_cls(gt_bboxes, dtype=torch.float32)
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)

    def _load_labels(self, results: dict) -> None:
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        """
        gt_bboxes_labels = []
        for instance in results.get('instances', []):
            gt_bboxes_labels.append(instance['bbox_label'])
        results['gt_bboxes_labels'] = np.array(gt_bboxes_labels, dtype=np.int64)

    def _poly2mask(self, mask_ann: Union[list, dict], img_h: int, img_w: int) -> np.ndarray:
        """Private function to convert masks represented with polygon to bitmaps.

        Args:
            mask_ann (list | dict): Polygon mask annotation input.
            img_h (int): The height of output mask.
            img_w (int): The width of output mask.

        Returns:
            np.ndarray: The decode bitmap mask of shape (img_h, img_w).
        """
        if isinstance(mask_ann, list):
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def _process_masks(self, results: dict) -> list:
        """Process gt_masks and filter invalid polygons.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            list: Processed gt_masks.
        """
        gt_masks = []
        gt_ignore_flags = []
        for instance in results.get('instances', []):
            gt_mask = instance['mask']
            if isinstance(gt_mask, list):
                gt_mask = [np.array(polygon) for polygon in gt_mask if len(polygon) % 2 == 0 and len(polygon) >= 6]
                if len(gt_mask) == 0:
                    instance['ignore_flag'] = 1
                    gt_mask = [np.zeros(6)]
            elif not self.poly2mask:
                instance['ignore_flag'] = 1
                gt_mask = [np.zeros(6)]
            elif isinstance(gt_mask, dict) and not (gt_mask.get('counts') is not None and gt_mask.get('size') is not None and isinstance(gt_mask['counts'], (list, str))):
                instance['ignore_flag'] = 1
                gt_mask = [np.zeros(6)]
            gt_masks.append(gt_mask)
            gt_ignore_flags.append(instance['ignore_flag'])
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)
        return gt_masks

    def _load_masks(self, results: dict) -> None:
        """Private function to load mask annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        """
        h, w = results['ori_shape']
        gt_masks = self._process_masks(results)
        if self.poly2mask:
            gt_masks = BitmapMasks([self._poly2mask(mask, h, w) for mask in gt_masks], h, w)
        else:
            gt_masks = PolygonMasks([mask for mask in gt_masks], h, w)
        results['gt_masks'] = gt_masks

       
    def _load_seg_map(self, results: dict) -> None:
        """Private function to load depth map annotations.

        Args:
            results (dict): Result dict from :obj:`mmcv.BaseDataset`.

        Returns:
            dict: The dict contains loaded depth map annotations.
        """
        
        # Path to the max depth information file
        max_depth_file = '/opt/data/private/fcf/Public_dataset/HazyDet-365K/data/HazyDet365K/Depth/max_depth.txt'
        
        # Read max depth information
        max_depth_map = {}
        with open(max_depth_file, 'r') as f:
            # Skip the header line
            for line in f.readlines()[1:]:
                parts = line.strip().split()
                if len(parts) == 2:
                    image_name, max_depth = parts
                    max_depth_map[image_name] = float(max_depth)
        
        # Load depth map
        img_bytes = fileio.get(results['seg_map_path'], backend_args=self.backend_args)
        
        # Read the image using OpenCV to handle PNG format
        depth_map = mmcv.imfrombytes(img_bytes, flag='unchanged', backend='cv2').astype(np.float32)
        
        # Extract the image file prefix name (without extension) from the path
        image_name = os.path.splitext(os.path.basename(results['seg_map_path']))[0]
        
        if image_name not in max_depth_map:
            raise ValueError(f"Max depth not found for image: {image_name}")
        
        max_depth = max_depth_map[image_name]

        # Convert relative depth to absolute depth
        absolute_depth_map = depth_map / 65535.0 * max_depth
               
        results['gt_seg_map'] = absolute_depth_map

        return results


    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label and semantic segmentation.
        """
        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        if self.with_mask:
            self._load_masks(results)
        if self.with_seg:
            self._load_seg_map(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str