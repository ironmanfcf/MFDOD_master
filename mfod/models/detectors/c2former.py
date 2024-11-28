
"""
authors: changfeng feng

Implementation of `C²Former: calibrated and complementary transformer for RGB-infrared object detection.`__

__ https://ieeexplore.ieee.org/document/10472947/
"""

from mfod.registry import MODELS
from mmrotate.models.detectors import RefineSingleStageDetector
import torch

from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig

from typing import List, Sequence, Tuple, Union, Dict
from mmdet.structures import DetDataSample, OptSampleList, SampleList
from torch import Tensor
from mmdet.models.utils import unpack_gt_instances



ForwardResults = Union[Dict[str, torch.Tensor], List[DetDataSample],
                       Tuple[torch.Tensor], torch.Tensor]

@MODELS.register_module()
class C2Former(RefineSingleStageDetector):


    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head_init: OptConfigType = None,
                 bbox_head_refine: List[OptConfigType] = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head_init=bbox_head_init,
            bbox_head_refine=bbox_head_refine,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg
        )

    def forward(self,
                inputs: torch.Tensor,
                inputs_ir: torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            inputs_ir (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, inputs_ir, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, inputs_ir, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, inputs_ir, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')


    def loss(self, batch_inputs: Tensor, batch_inputs_ir: Tensor,                 
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_inputs_ir (Tensor): Input images of shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs,batch_inputs_ir)

        losses = dict()
        outs = self.bbox_head_init(x)
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs
        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)
        init_losses = self.bbox_head_init.loss_by_feat(*loss_inputs)
        keys = init_losses.keys()
        for key in list(keys):
            if 'loss' in key and 'init' not in key:
                init_losses[f'{key}_init'] = init_losses.pop(key)
        losses.update(init_losses)

        rois = self.bbox_head_init.filter_bboxes(*outs)
        for i in range(self.num_refine_stages):
            weight = self.train_cfg.stage_loss_weights[i]
            x_refine = self.bbox_head_refine[i].feature_refine(x, rois)
            outs = self.bbox_head_refine[i](x_refine)
            loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                                  batch_gt_instances_ignore)
            refine_losses = self.bbox_head_refine[i].loss_by_feat(
                *loss_inputs, rois=rois)
            keys = refine_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'refine' not in key:
                    loss = refine_losses.pop(key)
                    if isinstance(loss, Sequence):
                        loss = [item * weight for item in loss]
                    else:
                        loss = loss * weight
                    refine_losses[f'{key}_refine_{i}'] = loss
            losses.update(refine_losses)

            if i + 1 in range(self.num_refine_stages):
                rois = self.bbox_head_refine[i].refine_bboxes(*outs, rois=rois)

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_inputs_ir: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_inputs_ir (Tensor): Input images of shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 5),
              the last dimension 5 arrange as (x, y, w, h, t).
        """
        x = self.extract_feat(batch_inputs, batch_inputs_ir)
        outs = self.bbox_head_init(x)
        rois = self.bbox_head_init.filter_bboxes(*outs)
        for i in range(self.num_refine_stages):
            x_refine = self.bbox_head_refine[i].feature_refine(x, rois)
            outs = self.bbox_head_refine[i](x_refine)
            if i + 1 in range(self.num_refine_stages):
                rois = self.bbox_head_refine[i].refine_bboxes(*outs, rois)

        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        predictions = self.bbox_head_refine[-1].predict_by_feat(
            *outs, rois=rois, batch_img_metas=batch_img_metas, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, predictions)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_inputs_ir: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_inputs_ir (Tensor): Input images of shape (N, C, H, W).

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x = self.extract_feat(batch_inputs, batch_inputs_ir)
        outs = self.bbox_head_init(x)
        rois = self.bbox_head_init.filter_bboxes(*outs)
        for i in range(self.num_refine_stages):
            x_refine = self.bbox_head_refine[i].feature_refine(x, rois)
            outs = self.bbox_head_refine[i](x_refine)
            if i + 1 in range(self.num_refine_stages):
                rois = self.bbox_head_refine[i].refine_bboxes(*outs, rois)

        return outs

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
        x = self.backbone(batch_inputs, batch_inputs_ir)
        if self.with_neck:
            x = self.neck(x)
        return x
        
        
        
        
"""
# origin code
# """
# import pdb

# from mmrotate. import rbbox2result
# from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
# from .base import RotatedBaseDetector
# from .utils import AlignConvModule
# from mmcv.runner import auto_fp16
# import torch
# import warnings

# from mmrotate.models import RefineSingleStageDetector


# @ROTATED_DETECTORS.register_module()
# class C2Former(RotatedBaseDetector):
#     """Implementation of `C²Former: calibrated and complementary transformer for RGB-infrared object detection`__

#     __ https://ieeexplore.ieee.org/document/9377550
#     """

#     def __init__(self,
#                  backbone,
#                  neck=None,
#                  fam_head=None,
#                  align_cfgs=None,
#                  odm_head=None,
#                  train_cfg=None,
#                  test_cfg=None,
#                  pretrained=None,
#                  init_cfg=None):
#         super(C2Former, self).__init__(init_cfg)

#         if pretrained:
#             warnings.warn('DeprecationWarning: pretrained is deprecated, '
#                           'please use "init_cfg" instead')
#             backbone.pretrained = pretrained

#         self.backbone = build_backbone(backbone)
#         if neck is not None:
#             self.neck = build_neck(neck)
#         if train_cfg is not None:
#             fam_head.update(train_cfg=train_cfg['fam_cfg'])
#         fam_head.update(test_cfg=test_cfg)
#         self.fam_head = build_head(fam_head)

#         self.align_conv_type = align_cfgs['type']
#         self.align_conv_size = align_cfgs['kernel_size']
#         self.feat_channels = align_cfgs['channels']
#         self.featmap_strides = align_cfgs['featmap_strides']

#         if self.align_conv_type == 'AlignConv':
#             self.align_conv = AlignConvModule(self.feat_channels,
#                                               self.featmap_strides,
#                                               self.align_conv_size)

#         if train_cfg is not None:
#             odm_head.update(train_cfg=train_cfg['odm_cfg'])
#         odm_head.update(test_cfg=test_cfg)
#         self.odm_head = build_head(odm_head)

#         self.train_cfg = train_cfg
#         self.test_cfg = test_cfg

#     def extract_feat(self, img, img_tir, img_meta):
#         """Directly extract features from the backbone+neck."""
#         x = self.backbone(img, img_tir)

#         if self.with_neck:
#             x = self.neck(x)

#         return x

#     def forward_dummy(self, img, img_tir):
#         """Used for computing network flops.

#         See `mmedetection/tools/get_flops.py`
#         """
#         x = self.extract_feat(img, img_tir)
#         outs = self.fam_head(x)
#         rois = self.fam_head.refine_bboxes(*outs)
#         # rois: list(indexed by images) of list(indexed by levels)
#         align_feat = self.align_conv(x, rois)
#         outs = self.odm_head(align_feat)

#         return outs

#     def forward_train(self,
#                       img,
#                       img_tir,
#                       img_metas,
#                       gt_bboxes,
#                       gt_labels,
#                       gt_bboxes_ignore=None):
#         """Forward function of S2ANet."""
#         losses = dict()
#         x = self.extract_feat(img, img_tir, img_metas)

#         outs = self.fam_head(x)

#         loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
#         loss_base = self.fam_head.loss(
#             *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
#         for name, value in loss_base.items():
#             losses[f'fam.{name}'] = value

#         rois = self.fam_head.refine_bboxes(*outs)
#         # rois: list(indexed by images) of list(indexed by levels)
#         align_feat = self.align_conv(x, rois)
#         outs = self.odm_head(align_feat)
#         loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
#         loss_refine = self.odm_head.loss(
#             *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore, rois=rois)
#         for name, value in loss_refine.items():
#             losses[f'odm.{name}'] = value

#         return losses

#     def simple_test(self, img, img_tir, img_meta, rescale=False):
#         """Test function without test time augmentation.

#         Args:
#             imgs (list[torch.Tensor]): List of multiple images
#             img_metas (list[dict]): List of image information.
#             rescale (bool, optional): Whether to rescale the results.
#                 Defaults to False.

#         Returns:
#             list[list[np.ndarray]]: BBox results of each image and classes. \
#                 The outer list corresponds to each image. The inner list \
#                 corresponds to each class.
#         """
#         x = self.extract_feat(img, img_tir, img_meta)
#         outs = self.fam_head(x)
#         rois = self.fam_head.refine_bboxes(*outs)
#         # rois: list(indexed by images) of list(indexed by levels)
#         align_feat = self.align_conv(x, rois)
#         outs = self.odm_head(align_feat)

#         bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
#         bbox_list = self.odm_head.get_bboxes(*bbox_inputs, rois=rois)
#         bbox_results = [
#             rbbox2result(det_bboxes, det_labels, self.odm_head.num_classes)
#             for det_bboxes, det_labels in bbox_list
#         ]
#         return bbox_results

#     def aug_test(self, imgs, img_metas, **kwargs):
#         """Test function with test time augmentation."""
#         raise NotImplementedError

#     def forward_test(self, imgs, imgs_tir,  img_metas, **kwargs):
#         """
#         Args:
#             imgs (List[Tensor]): the outer list indicates test-time
#                 augmentations and inner Tensor should have a shape NxCxHxW,
#                 which contains all images in the batch.
#             img_metas (List[List[dict]]): the outer list indicates test-time
#                 augs (multiscale, flip, etc.) and the inner list indicates
#                 images in a batch.
#         """
#         for var, name in [(imgs, 'imgs'), (imgs_tir, 'imgs_tir'), (img_metas, 'img_metas')]:
#             if not isinstance(var, list):
#                 raise TypeError(f'{name} must be a list, but got {type(var)}')

#         num_augs = len(imgs)
#         if num_augs != len(img_metas):
#             raise ValueError(f'num of augmentations ({len(imgs)}) '
#                              f'!= num of image meta ({len(img_metas)})')

#         # NOTE the batched image size information may be useful, e.g.
#         # in DETR, this is needed for the construction of masks, which is
#         # then used for the transformer_head.
#         for img, img_meta in zip(imgs, img_metas):
#             batch_size = len(img_meta)
#             for img_id in range(batch_size):
#                 img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

#         if num_augs == 1:
#             # proposals (List[List[Tensor]]): the outer list indicates
#             # test-time augs (multiscale, flip, etc.) and the inner list
#             # indicates images in a batch.
#             # The Tensor should have a shape Px4, where P is the number of
#             # proposals.
#             if 'proposals' in kwargs:
#                 kwargs['proposals'] = kwargs['proposals'][0]
#             return self.simple_test(imgs[0], imgs_tir[0], img_metas[0], **kwargs)
#         else:
#             assert imgs[0].size(0) == 1, 'aug test does not support ' \
#                                          'inference with batch size ' \
#                                          f'{imgs[0].size(0)}'
#             # TODO: support test augmentation for predefined proposals
#             assert 'proposals' not in kwargs
#             return self.aug_test(imgs, imgs_tir, img_metas, **kwargs)

#     @auto_fp16(apply_to=('img', 'img_tir', ))
#     def forward(self, img, img_tir, img_metas, return_loss=True, **kwargs):
#         """Calls either :func:`forward_train` or :func:`forward_test` depending
#         on whether ``return_loss`` is ``True``.

#         Note this setting will change the expected inputs. When
#         ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
#         and List[dict]), and when ``resturn_loss=False``, img and img_meta
#         should be double nested (i.e.  List[Tensor], List[List[dict]]), with
#         the outer list indicating test time augmentations.
#         """
#         if torch.onnx.is_in_onnx_export():
#             assert len(img_metas) == 1
#             return self.onnx_export(img[0], img_metas[0])

#         if return_loss:
#             return self.forward_train(img, img_tir, img_metas, **kwargs)
#         else:
#             return self.forward_test(img, img_tir, img_metas, **kwargs)

