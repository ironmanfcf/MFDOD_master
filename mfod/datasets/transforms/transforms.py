# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
import math
import warnings
from typing import List, Optional, Sequence, Tuple, Union,Dict
import mmengine
import cv2
import mmcv
import numpy
import numpy as np
from mmcv.image import imresize
from mmcv.image.geometric import _scale_size
from mmcv.transforms import BaseTransform



from mmdet.structures.bbox import HorizontalBoxes, autocast_box_type

from mmcv.transforms import Compose

Number = Union[int, float]

from mmdet.datasets.transforms import Resize as MMDET_Resize
from mmdet.datasets.transforms import RandomFlip as MMDET_RandomFlip
from mmdet.datasets.transforms import Pad as MMDET_Pad
from mmcv.transforms import MultiScaleFlipAug
from mmcv.transforms import RandomResize as MMCV_RandomResize

from mfod.registry import TRANSFORMS


def _fixed_scale_size(
    size: Tuple[int, int],
    scale: Union[float, int, tuple],
) -> Tuple[int, int]:
    """Rescale a size by a ratio.

    Args:
        size (tuple[int]): (w, h).
        scale (float | tuple(float)): Scaling factor.

    Returns:
        tuple[int]: scaled size.
    """
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    w, h = size
    # don’t need o.5 offset
    return int(w * float(scale[0])), int(h * float(scale[1]))


def rescale_size(old_size: tuple,
                 scale: Union[float, int, tuple],
                 return_scale: bool = False) -> tuple:
    """Calculate the new size to be rescaled to.

    Args:
        old_size (tuple[int]): The old size (w, h) of image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image size.

    Returns:
        tuple[int]: The new rescaled image size.
    """
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f'Invalid scale {scale}, must be positive.')
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
    else:
        raise TypeError(
            f'Scale must be a number or tuple of int, but got {type(scale)}')
    # only change this
    new_size = _fixed_scale_size((w, h), scale_factor)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size


def imrescale(
    img: np.ndarray,
    scale: Union[float, Tuple[int, int]],
    return_scale: bool = False,
    interpolation: str = 'bilinear',
    backend: Optional[str] = None
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """Resize image while keeping the aspect ratio.

    Args:
        img (ndarray): The input image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image.
        interpolation (str): Same as :func:`resize`.
        backend (str | None): Same as :func:`resize`.

    Returns:
        ndarray: The rescaled image.
    """
    h, w = img.shape[:2]
    new_size, scale_factor = rescale_size((w, h), scale, return_scale=True)
    rescaled_img = imresize(
        img, new_size, interpolation=interpolation, backend=backend)
    if return_scale:
        return rescaled_img, scale_factor
    else:
        return rescaled_img


"""
@author: changfeng feng
"""
@TRANSFORMS.register_module()
class PairedImagesResize(MMDET_Resize):
    """Resize paired images (RGB and IR) & bbox & seg.

    This transform resizes the input images according to ``scale`` or
    ``scale_factor``. Bboxes, masks, and seg map are then resized
    with the same scale factor.
    if ``scale`` and ``scale_factor`` are both set, it will use ``scale`` to
    resize.

    Required Keys:

    - img
    - img_ir
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:

    - img
    - img_ir
    - img_shape
    - img_shape_ir
    - gt_bboxes
    - gt_masks
    - gt_seg_map

    Added Keys:

    - scale
    - scale_factor
    - keep_ratio
    - homography_matrix

    Args:
        scale (int or tuple): Images scales for resizing. Defaults to None
        scale_factor (float or tuple[float]): Scale factors for resizing.
            Defaults to None.
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Defaults to False.
        clip_object_border (bool): Whether to clip the objects
            outside the border of the image. In some dataset like MOT17, the gt
            bboxes are allowed to cross the border of images. Therefore, we
            don't need to clip the gt bboxes in these cases. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def _resize_img_ir(self, results: dict) -> None:
        """Resize infrared images with ``results['scale']``."""
        if 'img_ir' in results:
            if self.keep_ratio:
                results['img_ir'], scale_factor = mmcv.imrescale(
                    results['img_ir'], results['scale'], return_scale=True, backend=self.backend, interpolation=self.interpolation)
                results['scale_factor_ir'] = (scale_factor, scale_factor)
            else:
                results['img_ir'], w_scale, h_scale = mmcv.imresize(
                    results['img_ir'], results['scale'], return_scale=True, backend=self.backend, interpolation=self.interpolation)
                results['scale_factor_ir'] = (w_scale, h_scale)
            results['img_shape_ir'] = results['img_ir'].shape[:2]

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes and semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'img_ir', 'gt_bboxes', 'gt_seg_map',
            'scale', 'scale_factor', 'height', 'width', and 'keep_ratio' keys
            are updated in result dict.
        """
        if self.scale:
            results['scale'] = self.scale
        else:
            img_shape = results['img'].shape[:2]
            results['scale'] = _scale_size(img_shape[::-1], self.scale_factor)
        self._resize_img(results)
        self._resize_img_ir(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        self._record_homography_matrix(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.scale}, '
        repr_str += f'scale_factor={self.scale_factor}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'clip_object_border={self.clip_object_border}), '
        repr_str += f'backend={self.backend}), '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str




"""
@author: changfeng feng 
"""
@TRANSFORMS.register_module()
class PairedImagesRandomResize(MMCV_RandomResize):
    def __init__(
            self,
            scale: Union[Tuple[int, int], Sequence[Tuple[int, int]]],
            ratio_range: Tuple[float, float] = None,
            resize_type: str = 'PairedImagesResize',  
            **resize_kwargs,
    ) -> None:
        self.scale = scale
        self.ratio_range = ratio_range

        self.resize_cfg = dict(type=resize_type, **resize_kwargs)
        # create a empty Reisize object
        self.resize = TRANSFORMS.build({'scale': 0, **self.resize_cfg})



"""
@author: changfeng feng
"""
@TRANSFORMS.register_module()
class PairedImagesRandomFlip(MMDET_RandomFlip):
    """Flip the image & bbox & mask & segmentation map. Added or Updated keys:
    flip, flip_direction, img, gt_bboxes, and gt_seg_map. There are 3 flip
    modes:

     - ``prob`` is float, ``direction`` is string: the image will be
         ``direction``ly flipped with probability of ``prob`` .
         E.g., ``prob=0.5``, ``direction='horizontal'``,
         then image will be horizontally flipped with probability of 0.5.
     - ``prob`` is float, ``direction`` is list of string: the image will
         be ``direction[i]``ly flipped with probability of
         ``prob/len(direction)``.
         E.g., ``prob=0.5``, ``direction=['horizontal', 'vertical']``,
         then image will be horizontally flipped with probability of 0.25,
         vertically with probability of 0.25.
     - ``prob`` is list of float, ``direction`` is list of string:
         given ``len(prob) == len(direction)``, the image will
         be ``direction[i]``ly flipped with probability of ``prob[i]``.
         E.g., ``prob=[0.3, 0.5]``, ``direction=['horizontal',
         'vertical']``, then image will be horizontally flipped with
         probability of 0.3, vertically with probability of 0.5.


    Required Keys:
    - img
    - img_ir
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:
    - img
    - img_ir
    - gt_bboxes
    - gt_masks
    - gt_seg_map

    Added Keys:
    - flip
    - flip_direction
    - homography_matrix

    Args:
         prob (float | list[float], optional): The flipping probability.
             Defaults to None.
         direction(str | list[str]): The flipping direction. Options
             If input is a list, the length must equal ``prob``. Each
             element in ``prob`` indicates the flip probability of
             corresponding direction. Defaults to 'horizontal'.
    """

    @autocast_box_type()
    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes, and semantic segmentation map."""
        # flip image
        results['img'] = mmcv.imflip(
            results['img'], direction=results['flip_direction'])
        results['img_ir'] = mmcv.imflip(
            results['img_ir'], direction=results['flip_direction'])

        img_shape = results['img'].shape[:2]

        # flip bboxes
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'].flip_(img_shape, results['flip_direction'])

        # flip masks
        if results.get('gt_masks', None) is not None:
            results['gt_masks'] = results['gt_masks'].flip(
                results['flip_direction'])

        # flip segs
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = mmcv.imflip(
                results['gt_seg_map'], direction=results['flip_direction'])

        # record homography matrix for flip
        self._record_homography_matrix(results)

"""
@author：changfeng feng
"""
@TRANSFORMS.register_module()
class PairedImagesPad(MMDET_Pad):
    """Pad the image & segmentation map.

    There are three padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number. and (3)pad to square. Also,
    pad to square and pad to the minimum size can be used as the same time.

    Required Keys:

    - img
    - img_ir
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:

    - img
    - img_ir
    - img_shape
    - img_shape_ir
    - gt_masks
    - gt_seg_map

    Added Keys:

    - pad_shape
    - pad_fixed_size
    - pad_size_divisor

    Args:
        size (tuple, optional): Fixed padding size.
            Expected padding shape (width, height). Defaults to None.
        size_divisor (int, optional): The divisor of padded size. Defaults to
            None.
        pad_to_square (bool): Whether to pad the image into a square.
            Currently only used for YOLOX. Defaults to False.
        pad_val (Number | dict[str, Number], optional) - Padding value for if
            the pad_mode is "constant".  If it is a single number, the value
            to pad the image is the number and to pad the semantic
            segmentation map is 255. If it is a dict, it should have the
            following keys:

            - img: The value to pad the image.
            - seg: The value to pad the semantic segmentation map.
            Defaults to dict(img=0, seg=255).
        padding_mode (str): Type of padding. Should be: constant, edge,
            reflect or symmetric. Defaults to 'constant'.

            - constant: pads with a constant value, this value is specified
              with pad_val.
            - edge: pads with the last value at the edge of the image.
            - reflect: pads with reflection of image without repeating the last
              value on the edge. For example, padding [1, 2, 3, 4] with 2
              elements on both sides in reflect mode will result in
              [3, 2, 1, 2, 3, 4, 3, 2].
            - symmetric: pads with reflection of image repeating the last value
              on the edge. For example, padding [1, 2, 3, 4] with 2 elements on
              both sides in symmetric mode will result in
              [2, 1, 1, 2, 3, 4, 4, 3]
    """

    def _pad_img_ir(self, results: dict) -> None:
        """Pad infrared images according to ``results['pad_shape']``."""
        if 'img_ir' in results:
            pad_val = self.pad_val.get('img', 0) if isinstance(self.pad_val, dict) else self.pad_val
            results['img_ir'] = mmcv.impad(
                results['img_ir'], shape=results['pad_shape'], pad_val=pad_val)
            results['img_shape_ir'] = results['img_ir'].shape

    def transform(self, results: dict) -> dict:
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        self._pad_img_ir(results)
        self._pad_seg(results)
        self._pad_masks(results)
        return results


"""
@author: changfeng feng
"""
@TRANSFORMS.register_module()
class PairedImageMultiScaleFlipAug(MultiScaleFlipAug):
    """Test-time augmentation with multiple scales and flipping.

    An example configuration is as followed:

    .. code-block::

        dict(
            type='MultiScaleFlipAug',
            scales=[(1333, 400), (1333, 800)],
            flip=True,
            transforms=[
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=1),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ])

    ``results`` will be resized using all the sizes in ``scales``.
    If ``flip`` is True, then flipped results will also be added into output
    list.

    For the above configuration, there are four combinations of resize
    and flip:

    - Resize to (1333, 400) + no flip
    - Resize to (1333, 400) + flip
    - Resize to (1333, 800) + no flip
    - resize to (1333, 800) + flip

    The four results are then transformed with ``transforms`` argument.
    After that, results are wrapped into lists of the same length as below:

    .. code-block::

        dict(
            inputs=[...],
            data_samples=[...]
        )

    Where the length of ``inputs`` and ``data_samples`` are both 4.

    Required Keys:

    - Depending on the requirements of the ``transforms`` parameter.

    Modified Keys:

    - All output keys of each transform.

    Args:
        transforms (list[dict]): Transforms to be applied to each resized
            and flipped data.
        scales (tuple | list[tuple] | None): Images scales for resizing.
        scale_factor (float or tuple[float]): Scale factors for resizing.
            Defaults to None.
        allow_flip (bool): Whether apply flip augmentation. Defaults to False.
        flip_direction (str | list[str]): Flip augmentation directions,
            options are "horizontal", "vertical" and "diagonal". If
            flip_direction is a list, multiple flip augmentations will be
            applied. It has no effect when flip == False. Defaults to
            "horizontal".
        resize_cfg (dict): Base config for resizing. Defaults to
            ``dict(type='Resize', keep_ratio=True)``.
        flip_cfg (dict): Base config for flipping. Defaults to
            ``dict(type='RandomFlip')``.
    """

    def __init__(
        self,
        transforms: List[dict],
        scales: Optional[Union[Tuple, List[Tuple]]] = None,
        scale_factor: Optional[Union[float, List[float]]] = None,
        allow_flip: bool = False,
        flip_direction: Union[str, List[str]] = 'horizontal',
        resize_cfg: dict = dict(type='Resize', keep_ratio=True),
        flip_cfg: dict = dict(type='RandomFlip')
    ) -> None:

        self.transforms = Compose(transforms)  # type: ignore

        if scales is not None:
            self.scales = scales if isinstance(scales, list) else [scales]
            self.scale_key = 'scale'
            assert mmengine.is_list_of(self.scales, tuple)
        else:
            # if ``scales`` and ``scale_factor`` both be ``None``
            if scale_factor is None:
                self.scales = [1.]  # type: ignore
            elif isinstance(scale_factor, list):
                self.scales = scale_factor  # type: ignore
            else:
                self.scales = [scale_factor]  # type: ignore

            self.scale_key = 'scale_factor'

        self.allow_flip = allow_flip
        self.flip_direction = flip_direction if isinstance(
            flip_direction, list) else [flip_direction]
        assert mmengine.is_list_of(self.flip_direction, str)
        if not self.allow_flip and self.flip_direction != ['horizontal']:
            warnings.warn(
                'flip_direction has no effect when flip is set to False')
        self.resize_cfg = resize_cfg.copy()
        self.flip_cfg = flip_cfg

    def transform(self, results: dict) -> Dict:
        """Apply test time augment transforms on results.

        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
            dict: The augmented data, where each value is wrapped
            into a list.
        """

        data_samples = []
        flip_args = [(False, '')]
        if self.allow_flip:
            flip_args += [(True, direction)
                          for direction in self.flip_direction]
        for scale in self.scales:
            for flip, direction in flip_args:
                _resize_cfg = self.resize_cfg.copy()
                _resize_cfg.update({self.scale_key: scale})
                _resize_flip = [_resize_cfg]

                if flip:
                    _flip_cfg = self.flip_cfg.copy()
                    _flip_cfg.update(prob=1.0, direction=direction)
                    _resize_flip.append(_flip_cfg)
                else:
                    results['flip'] = False
                    results['flip_direction'] = None
                resize_flip = Compose(_resize_flip)

                _results = resize_flip(results.copy())

                packed_results = self.transforms(_results)  # type: ignore
                #
                inputs = packed_results['inputs']
                inputs_ir = packed_results['inputs_ir']
                data_samples = packed_results['data_samples']  # type: ignore
                # print(len(inputs_lwir))

        return {'inputs':inputs,'inputs_ir':inputs_ir ,'data_samples':data_samples}

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}'
        repr_str += f', scales={self.scales}'
        repr_str += f', allow_flip={self.allow_flip}'
        repr_str += f', flip_direction={self.flip_direction})'
        return repr_str

