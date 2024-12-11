# Copyright (c) OpenMMLab. All rights reserved.
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
from mmengine.utils import is_list_of

from mmdet.structures import DetDataSample, OptSampleList, SampleList
from mmdet.utils import InstanceList, OptConfigType, OptMultiConfig

from ..fusion.e2emfd_fusion import FusionNet
from ..losses import DetcropPixelLoss
import matplotlib.pyplot as plt
from mmengine.optim import OptimWrapper
from collections import OrderedDict

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

def unpack_gt_instances(batch_data_samples: SampleList) -> tuple:
    """Unpack ``gt_instances``, ``gt_instances_ignore`` and ``img_metas`` based
    on ``batch_data_samples``

    Args:
        batch_data_samples (List[:obj:`DetDataSample`]): The Data
            Samples. It usually includes information such as
            `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

    Returns:
        tuple:

            - batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            - batch_gt_instances_ignore (list[:obj:`InstanceData`]):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            - batch_img_metas (list[dict]): Meta information of each image,
                e.g., image size, scaling factor, etc.
    """
    batch_gt_instances = []
    batch_gt_instances_ignore = []
    batch_img_metas = []
    for data_sample in batch_data_samples:
        batch_img_metas.append(data_sample.metainfo)
        batch_gt_instances.append(data_sample.gt_instances)
        if 'ignored_instances' in data_sample:
            batch_gt_instances_ignore.append(data_sample.ignored_instances)
        else:
            batch_gt_instances_ignore.append(None)

    return batch_gt_instances, batch_gt_instances_ignore, batch_img_metas

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
        self.fusion = FusionNet(block_num=3, feature_out=False)
        self.criterion_fuse = DetcropPixelLoss()       

    def forward(self,
                inputs: torch.Tensor,
                inputs_ir: torch.Tensor,
                inputs_vis_bri: torch.Tensor,
                inputs_vis_clr: torch.Tensor,
                inputs_F_vis: torch.Tensor,
                inputs_F_ir: torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') :
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
                return self.loss(inputs, inputs_ir, inputs_vis_bri, inputs_vis_clr,inputs_F_vis,inputs_F_ir, data_samples)
            elif mode == 'predict':
                return self.predict(inputs, inputs_ir, inputs_vis_bri, inputs_vis_clr,inputs_F_vis,inputs_F_ir, data_samples)
            elif mode == 'tensor':
                return self._forward(inputs, inputs_ir, inputs_vis_bri, inputs_vis_clr,inputs_F_vis,inputs_F_ir, data_samples)
            else:
                raise RuntimeError(f'Invalid mode "{mode}". '
                                'Only supports loss, predict and tensor mode')

    def extract_feat(self,
                batch_inputs: torch.Tensor,
                batch_inputs_ir: torch.Tensor,
                batch_inputs_vis_bri: torch.Tensor,
                batch_inputs_vis_clr: torch.Tensor,
                batch_inputs_F_vis: torch.Tensor,
                batch_inputs_F_ir: torch.Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        OD_RGB, OD_IR, RGB_bri, RGB_clr, F_rgb, F_ir = batch_inputs, \
                                                     batch_inputs_ir,\
                                                     batch_inputs_vis_bri,\
                                                    batch_inputs_vis_clr,\
                                                    batch_inputs_F_vis,\
                                                    batch_inputs_F_ir
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


    def _forward(self,
                batch_inputs: torch.Tensor,
                batch_inputs_ir: torch.Tensor,
                batch_inputs_vis_bri: torch.Tensor,
                batch_inputs_vis_clr: torch.Tensor,
                batch_inputs_F_vis: torch.Tensor,
                batch_inputs_F_ir: torch.Tensor,
                batch_data_samples: SampleList) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """

        
        results = ()
        x = self.extract_feat(batch_inputs,
                            batch_inputs_ir,
                            batch_inputs_vis_bri,
                            batch_inputs_vis_clr,
                            batch_inputs_F_vis,
                            batch_inputs_F_ir) 

        if self.with_rpn:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        roi_outs = self.roi_head.forward(x, rpn_results_list,
                                         batch_data_samples)
        results = results + (roi_outs, )
        return results

    def loss(self,
                batch_inputs: torch.Tensor,
                batch_inputs_ir: torch.Tensor,
                batch_inputs_vis_bri: torch.Tensor,
                batch_inputs_vis_clr: torch.Tensor,
                batch_inputs_F_vis: torch.Tensor,
                batch_inputs_F_ir: torch.Tensor,
                batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        x = self.extract_feat(batch_inputs,
                            batch_inputs_ir,
                            batch_inputs_vis_bri,
                            batch_inputs_vis_clr,
                            batch_inputs_F_vis,
                            batch_inputs_F_ir) 
        fusion_feature = [x1 for x1 in x[:4]]
        OD_RGB, OD_IR, RGB_bri, RGB_clr, F_rgb, F_ir = batch_inputs, \
                                                     batch_inputs_ir,\
                                                     batch_inputs_vis_bri,\
                                                    batch_inputs_vis_clr,\
                                                    batch_inputs_F_vis,\
                                                    batch_inputs_F_ir
        F_ir = F_ir.to(torch.float32)
        F_rgb = F_rgb.to(torch.float32)
        RGB_bri = RGB_bri.to(torch.float32)
        inputs = torch.cat([F_ir[:,0:1,:,:], F_rgb], dim=1)                                                    
        _, res_weight = self.fusion(fusion_feature, inputs)                                                    



        ##获取标注
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,batch_img_metas) = outputs    
        ##ORPPR(Object Region Pixel Phylogenetic Tree)在mmcv2.x上的实现
                                                         
        num_greater = None
        total = None 
        
        fus_img = res_weight[:, 0:1, :, :] * F_ir[:,0:1,:,:] + res_weight[:, 1:, :, :] * RGB_bri  #origin
        mask_list = []
        
        H,W = F_ir.shape[-2], F_ir.shape[-1]       
        
        for batch_data_sample in batch_data_samples:
            per_image = batch_data_sample.gt_instances.bboxes
            #根据输入图像大小改编mask大小
            mask = torch.zeros(H, W)
            n_cx_cy_w_h_a = per_image.cpu().numpy()
            points = obb2poly_np_le90(n_cx_cy_w_h_a)
            points = torch.tensor(points).to(batch_inputs.device).to(torch.float32)
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
       
        mask = torch.stack(mask_list).unsqueeze(1).to(batch_inputs.device)
        pad1 = int((fus_img.shape[-2]-mask.shape[-2]))
        pad2 = int((fus_img.shape[-1]-mask.shape[-1]))
        mask = F.pad(mask, (0,pad2,0,pad1))
        
        
        SSIM_loss,grad_loss,pixel_loss = self.criterion_fuse(fus_img, RGB_bri, F_ir[:,0:1,:,:], \
                                                             mask, num_greater, total)
        
        losses = dict()
        losses_new = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)
        ####
        losses_new['fusion_loss'] = 10*SSIM_loss + 10*grad_loss + 10*pixel_loss 
        losses_new['detection_loss'] = losses['loss_rpn_cls'][0] + losses['loss_rpn_cls'][1] + losses['loss_rpn_cls'][2] + losses['loss_rpn_cls'][3] + \
                losses['loss_rpn_cls'][4] +  \
                losses['loss_rpn_bbox'][0] + losses['loss_rpn_bbox'][1] + losses['loss_rpn_bbox'][2] + losses['loss_rpn_bbox'][3] + losses['loss_rpn_bbox'][4] + \
                losses['loss_cls'] + losses['loss_bbox']
        losses_new['acc'] = losses['acc']
        return losses_new

    def predict(self,
                batch_inputs: torch.Tensor,
                batch_inputs_ir: torch.Tensor,
                batch_inputs_vis_bri: torch.Tensor,
                batch_inputs_vis_clr: torch.Tensor,
                batch_inputs_F_vis: torch.Tensor,
                batch_inputs_F_ir: torch.Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(batch_inputs,
                            batch_inputs_ir,
                            batch_inputs_vis_bri,
                            batch_inputs_vis_clr,
                            batch_inputs_F_vis,
                            batch_inputs_F_ir) 

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples


    def forward_fusion(self,  batch_inputs: torch.Tensor,
                batch_inputs_ir: torch.Tensor,
                batch_inputs_vis_bri: torch.Tensor,
                batch_inputs_vis_clr: torch.Tensor,
                batch_inputs_F_vis: torch.Tensor,
                batch_inputs_F_ir: torch.Tensor):
        
        OD_RGB, OD_IR, RGB_bri, RGB_clr, F_rgb, F_ir = batch_inputs, \
                                                     batch_inputs_ir,\
                                                     batch_inputs_vis_bri,\
                                                    batch_inputs_vis_clr,\
                                                    batch_inputs_F_vis,\
                                                    batch_inputs_F_ir
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


    def train_step(self, data: Union[dict, tuple, list],
                   shared_parameter, 
                   task_specific_params,
                   iter,
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Implements the default model training process including
        preprocessing, model forward propagation, loss calculation,
        optimization, and back-propagation.

        During non-distributed training. If subclasses do not override the
        :meth:`train_step`, :class:`EpochBasedTrainLoop` or
        :class:`IterBasedTrainLoop` will call this method to update model
        parameters. The default parameter update process is as follows:

        1. Calls ``self.data_processor(data, training=False)`` to collect
           batch_inputs and corresponding data_samples(labels).
        2. Calls ``self(batch_inputs, data_samples, mode='loss')`` to get raw
           loss
        3. Calls ``self.parse_losses`` to get ``parsed_losses`` tensor used to
           backward and dict of loss tensor used to log messages.
        4. Calls ``optim_wrapper.update_params(loss)`` to update model.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        # Enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses = self._run_forward(data, mode='loss')  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore

        
#---------------------------modifications------------------------
        del parsed_losses['acc']
        
        # if iter % 1000 ==0 and iter>=1000: #从第1000个iter开始对齐，每隔1000对齐一次
        if iter == 2:   
            
            # runner.model.zero_grad()
            # self.balancer.step_with_model(
            #     losses = runner.outputs['loss'],
            #     shared_params = shared_parameter,
            #     task_specific_params = task_specific_params,
            #     last_shared_layer_params = None,
            #     iter=runner.iter
            # )
            
            optim_wrapper.gmta_update_params(parsed_losses,
                                             shared_parameter, 
                                             task_specific_params,
                                             iter)
        else:
            # (runner.outputs['loss']['fusion_loss'] + runner.outputs['loss']['detection_loss']).backward()  
            optim_wrapper.update_params(parsed_losses['fusion_loss']+parsed_losses['detection_loss'])
        # optim_wrapper.update_params(parsed_losses)
        return log_vars


    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore
        ###
        # 这里的方法与E2EMFD源代码中一致，送入优化器的损失不是求和后的损失，
        # 而是任务分离的损失，源代码中直接重写了BaseDetector的代码，导致运行其他方法时会出错。
        # return loss, log_vars  # type: ignore
        return losses, log_vars  # type: ignore    