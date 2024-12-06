# Copyright (c) OpenMMLab. All rights reserved.
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

from ..fusion.e2emfd_fusion import FusionNet
from ..losses import DetcropPixelLoss
import matplotlib.pyplot as plt


from mfod.registry import MODELS

# from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
# from .base import RotatedBaseDetector
from mmdet.models import TwoStageDetector


def cal_line_length(point1, point2):
    """Calculate the length of line.

    Args:
        point1 (List): [x,y]
        point2 (List): [x,y]

    Returns:
        length (float)
    """
    return math.sqrt(
        math.pow(point1[0] - point2[0], 2) +
        math.pow(point1[1] - point2[1], 2))
def get_best_begin_point_single(coordinate):
    """Get the best begin point of the single polygon.

    Args:
        coordinate (List): [x1, y1, x2, y2, x3, y3, x4, y4, score]

    Returns:
        reorder coordinate (List): [x1, y1, x2, y2, x3, y3, x4, y4, score]
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combine = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
               [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
               [[x3, y3], [x4, y4], [x1, y1], [x2, y2]],
               [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combine[i][0], dst_coordinate[0]) \
                     + cal_line_length(combine[i][1], dst_coordinate[1]) \
                     + cal_line_length(combine[i][2], dst_coordinate[2]) \
                     + cal_line_length(combine[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
    return np.hstack(
        (np.array(combine[force_flag]).reshape(8)))
def get_best_begin_point(coordinates):
    """Get the best begin points of polygons.

    Args:
        coordinate (ndarray): shape(n, 9).

    Returns:
        reorder coordinate (ndarray): shape(n, 9).
    """
    coordinates = list(map(get_best_begin_point_single, coordinates.tolist()))
    # coordinates = list(get_best_begin_point_single(coordinates.tolist()))
    coordinates = np.array(coordinates)
    return coordinates
def obb2poly_np_le90(obboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle,score]

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3,score]
    """
    try:
        center, w, h, theta = np.split(obboxes, (2, 3, 4), axis=-1)
    except:  # noqa: E722
        results = np.stack([0., 0., 0., 0., 0., 0., 0., 0., 0.], axis=-1)
        return results.reshape(1, -1)
    Cos, Sin = np.cos(theta), np.sin(theta)
    vector1 = np.concatenate([w / 2 * Cos, w / 2 * Sin], axis=-1)
    vector2 = np.concatenate([-h / 2 * Sin, h / 2 * Cos], axis=-1)
    point1 = center - vector1 - vector2
    point2 = center + vector1 - vector2
    point3 = center + vector1 + vector2
    point4 = center - vector1 + vector2
    polys = np.concatenate([point1, point2, point3, point4], axis=-1)
    polys = get_best_begin_point(polys)
    return polys

@MODELS.register_module()
class E2EMFD(TwoStageDetector):
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
        super(E2EMFD, self).__init__(backbone=backbone,
                                     neck=neck,
                                     rpn_head=rpn_head,
                                     roi_head=roi_head,
                                     train_cfg=train_cfg,
                                     test_cfg=test_cfg,
                                     data_preprocessor=data_preprocessor,
                                     init_cfg=init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained

        self.backbone = MODELS.build(backbone)

        if neck is not None:
            self.neck = MODELS.build(neck)

        self.fusion = FusionNet(block_num=3, feature_out=False)
        self.criterion_fuse = DetcropPixelLoss()

        # 添加对 data_preprocessor 的处理
        if data_preprocessor is not None:
            self.data_preprocessor = MODELS.build(data_preprocessor)
        else:
            self.data_preprocessor = None

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = MODELS.build(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            # 移除 pretrained 参数
            self.roi_head = MODELS.build(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        OD_RGB, OD_IR, RGB_bri, RGB_clr, F_rgb, F_ir = img
        # inputs = torch.cat([F_ir[:,0:1,:,:], F_rgb], dim=1)
        # vis_weight = None
        # inf_weight = None
        x_rgb = self.backbone(OD_RGB)
        x_ir = self.backbone(OD_IR)

        if self.with_neck:
            x_rgb = self.neck(x_rgb)
            x_ir = self.neck(x_ir)
        features = list()
        for i in range(5):
            feature = x_rgb[i]+x_ir[i]
            features.append(feature)
        return features

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat((img, img))
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 6).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def change_hsv2rgb(self, fus_img, visimage_clr):
        bri = fus_img.detach().cpu().numpy() * 255
        bri = bri.reshape([fus_img.size()[2], fus_img.size()[3]])
        # bri = np.where(bri < 0, 0, bri)
        # bri = np.where(bri > 255, 255, bri)
        min_value = bri.min()
        max_value = bri.max() 
        scale = 255 / (max_value - min_value) 
        bri = (bri - min_value) * scale
        bri = np.clip(bri, 0, 255)
        im1 = Image.fromarray(bri.astype(np.uint8))

        clr = visimage_clr.cpu().numpy().squeeze().transpose(1, 2, 0)
        clr = np.concatenate((clr, bri.reshape(fus_img.size()[2], fus_img.size()[3], 1)), axis=2)
        clr[:, :, 2] = im1
        clr = cv2.cvtColor(clr.astype(np.uint8), cv2.COLOR_HSV2RGB) 
        
        return clr

    def forward_fusion(self, img):
        OD_RGB, OD_IR, RGB_bri, RGB_clr, F_rgb, F_ir = img
        device = 'cuda'
        vi_rgb = OD_RGB.data[0].to(device)
        visimage_bri = RGB_bri.data[0].to(device)
        visimage_clr = RGB_clr.data[0].to(device)
        ir_image = F_ir.data[0][:,0:1,:,:].to(device)
        vi_image = F_rgb.data[0].to(device)
        ir_image = ir_image.to(torch.float32)
        vi_image = vi_image.to(torch.float32)
        visimage_bri = visimage_bri.to(torch.float32)
        ir_rgb = OD_IR.data[0].to(device)
        inputs = torch.cat([ir_image, vi_image], dim=1)
        start = time.time()

        x_rgb = self.backbone(vi_rgb)          #vi_image是/255的，vi_rgb是用的normalizer1
        x_ir = self.backbone(ir_rgb) #3个通道     ir_rgb是用的normalizer1   ir_image是/255的

        if self.with_neck:
            x_rgb = self.neck(x_rgb)
            x_ir = self.neck(x_ir)

        features = list()
        for i in range(4):
            feature = x_rgb[i]+x_ir[i]
            features.append(feature)
        _, res_weight = self.fusion(features,inputs) #这个测试需要修改
        end = time.time()
        time_per_img = end-start
        
        # greater = torch.gt(res_weight[:, 0:1, :, :], res_weight[:, 1:, :, :])  #ir权重大于bri的话在对应位置返回true
        # num_greater = torch.sum(greater).item()
        # total = res_weight[:, 1:, :, :].numel()

        # if num_greater > int(total * 0.8):
        #     fus_img = ir_image.to(torch.float32) + res_weight[:, 1:, :, :] * visimage_bri.to(torch.float32)#/ 255.
        # elif (total - num_greater) > int(total * 0.8):
        #     fus_img = res_weight[:, 0:1, :, :] * ir_image.to(torch.float32) + visimage_bri.to(torch.float32)#/ 255.
        # else:
        #     fus_img = res_weight[:, 0:1, :, :] * ir_image.to(torch.float32) + res_weight[:, 1:, :, :] * visimage_bri.to(torch.float32)
        fus_img = res_weight[:, 0:1, :, :] * ir_image + res_weight[:, 1:, :, :] * visimage_bri

        fusion_img = self.change_hsv2rgb(fus_img, visimage_clr)
      
        return fus_img, fusion_img, time_per_img, vi_image.squeeze(0)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)
        fusion_feature = [x1 for x1 in x[:4]]
        OD_RGB, OD_IR, RGB_bri, RGB_clr, F_rgb, F_ir = img
        F_ir = F_ir.to(torch.float32)
        F_rgb = F_rgb.to(torch.float32)
        RGB_bri = RGB_bri.to(torch.float32)
        inputs = torch.cat([F_ir[:,0:1,:,:], F_rgb], dim=1)
        vis_weight = None
        inf_weight = None
        _, res_weight = self.fusion(fusion_feature, inputs)
        
        # greater = []
        # num_greater = []
        # total = []
        # for i in range(2):   #2是batch size数
        #     greater.append(torch.gt(res_weight[i:i+1, 0:1, :, :], res_weight[i:i+1, 1:, :, :]))  #ir权重大于bri的话在对应位置返回true
        #     num_greater.append(torch.sum(greater[i]).item())
        #     total.append(res_weight[i:i+1, 1:, :, :].numel())
        num_greater = None
        total = None
        
        # fus_img_temp = []
        # for i in range(2):
        #     if num_greater[i] > int(total[i] * 0.8):
        #         fus_img_temp.append(F_ir[i:i+1, 0:1, :, :] + res_weight[i:i+1, 1:, :, :] * RGB_bri[i:i+1, :, :, :])#/ 255.
        #     elif (total[i] - num_greater[i]) > int(total[i] * 0.8):
        #         fus_img_temp.append(res_weight[i:i+1, 0:1, :, :] * F_ir[i:i+1, 0:1, :, :] + RGB_bri[i:i+1, :, :, :])#/ 255.
        #     else:
        #         fus_img_temp.append(res_weight[i:i+1, 0:1, :, :] * F_ir[i:i+1, 0:1, :, :] + res_weight[i:i+1, 1:, :, :] * RGB_bri[i:i+1, :, :, :])
        # fus_img = torch.cat([fus_img_temp[0], fus_img_temp[1]], dim=0)
        fus_img = res_weight[:, 0:1, :, :] * F_ir[:,0:1,:,:] + res_weight[:, 1:, :, :] * RGB_bri  #origin
        mask_list = []
        for per_image in gt_bboxes:
            mask = torch.zeros(712,840)
            n_cx_cy_w_h_a = per_image.cpu().numpy()
            points = obb2poly_np_le90(n_cx_cy_w_h_a)
            points = torch.tensor(points).to(img[0].device).to(torch.float32)
            for k in range(points.shape[0]):
            # for k in range(per_image.shape[0]):
                #horizontal
                # mask[int(per_image[k][1] - 0.5 * per_image[k][3]):int(per_image[k][1] + 0.5*per_image[k][3]),\
                #      int(per_image[k][0] - 0.5 * per_image[k][2]):int(per_image[k][0] + 0.5*per_image[k][2])]=1
                #horizontal_full_wrap
                max_x, _ = torch.max(torch.stack([points[k][0], points[k][2], points[k][4], points[k][6]]), dim=0)
                min_x, _ = torch.min(torch.stack([points[k][0], points[k][2], points[k][4], points[k][6]]), dim=0)
                max_y, _ = torch.max(torch.stack([points[k][1], points[k][3], points[k][5], points[k][7]]), dim=0)
                min_y, _ = torch.min(torch.stack([points[k][1], points[k][3], points[k][5], points[k][7]]), dim=0)
                mask[int(min_y) : int(max_y), int(min_x) : int(max_x)]=1
            mask_list.append(mask)
       
        mask = torch.stack(mask_list).unsqueeze(1).to(img[0].device)
        pad1 = int((fus_img.shape[-2]-mask.shape[-2]))
        pad2 = int((fus_img.shape[-1]-mask.shape[-1]))
        mask = F.pad(mask, (0,pad2,0,pad1))

        # ### 做mask的可视化
        # plt.imshow(mask.cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8))
        # plt.savefig("mask.png")
        # plt.imshow((fus_img*255).detach().cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8))
        # plt.savefig("fus_img.png")
        # plt.imshow((F_ir*255).detach().cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8))
        # plt.savefig("ir_image.png")
        # plt.imshow((F_ir[:,0:1,:,:]*255).detach().cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8))
        # plt.savefig("ir_image_dan.png")
        # plt.imshow((RGB_bri*255).detach().cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8))
        # plt.savefig("RGB_bri.png")
        # # 将叠加图像叠加到背景图像上
        # background_image = (F_ir*255).detach().cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8)
        # overlay_image = (mask*255).cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8)
        # alpha = 0.5  # 设置叠加图像的透明度
        # x_offset = 0  # 设置叠加图像的水平偏移量
        # y_offset = 0  # 设置叠加图像的垂直偏移量
        # background_image[y_offset:y_offset+overlay_image.shape[0], x_offset:x_offset+overlay_image.shape[1]] = \
        #     alpha * overlay_image + (1 - alpha) * background_image[y_offset:y_offset+overlay_image.shape[0], x_offset:x_offset+overlay_image.shape[1]]
        # # 使用 Matplotlib 显示叠加后的图像
        # plt.imshow(background_image)
        # plt.axis('off')
        # plt.savefig("ir_overlay_mask.png")
            
        SSIM_loss,grad_loss,pixel_loss = self.criterion_fuse(fus_img, RGB_bri, F_ir[:,0:1,:,:], \
                                                             mask, num_greater, total)
        
        losses = dict()
        losses_new = dict()
        # losses['SSIM_loss'] = 10*SSIM_loss 
        # losses['grad_loss'] = 10*grad_loss 
        # losses['pixel_loss'] = 10*pixel_loss 

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)
        losses_new['fusion_loss'] = 10*SSIM_loss + 10*grad_loss + 10*pixel_loss 
        losses_new['detection_loss'] = losses['loss_rpn_cls'][0] + losses['loss_rpn_cls'][1] + losses['loss_rpn_cls'][2] + losses['loss_rpn_cls'][3] + \
                losses['loss_rpn_cls'][4] +  \
                losses['loss_rpn_bbox'][0] + losses['loss_rpn_bbox'][1] + losses['loss_rpn_bbox'][2] + losses['loss_rpn_bbox'][3] + losses['loss_rpn_bbox'][4] + \
                losses['loss_cls'] + losses['loss_bbox']
        losses_new['acc'] = losses['acc']
        return losses_new
    
    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
    
    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                if isinstance(img, list) or isinstance(img, tuple):
                    img_meta[img_id]['batch_input_shape'] = tuple(img[0].size()[-2:])
                else:
                    img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)
