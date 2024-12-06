_base_ = [
    '../_base_/default_runtime.py'
]

angle_version = 'le90'
model = dict(
    type='mfod.FrenquenceDet',
    data_preprocessor=dict(
        type='mfod.DataPreprocessor',
        mean_ir= [123.675, 116.28, 103.53],
        std_ir=[58.395, 57.12, 57.375],        
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),
    backbone=dict(
        type='mfod.LSKNet',
        embed_dims=[64, 128, 320, 512],
        drop_rate=0.1,
        drop_path_rate=0.1,
        depths=[2,2,4,2],
        # download from https://github.com/zcablii/LSKNet
        init_cfg=dict(type='Pretrained', checkpoint="/opt/data/private/fcf/MFOD_master/pretrained_weights/lsk_s_backbone-e9d2e551.pth"),
        norm_cfg=dict(type='BN', requires_grad=True)),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='OrientedRPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='mmdet.AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
            use_box_type=True),
        bbox_coder=dict(
            type='MidpointOffsetCoder',
            angle_version=angle_version,
            target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss',
            beta=0.1111111111111111,
            loss_weight=1.0)),
    roi_head=dict(
        type='mmdet.StandardRoIHead',
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='mmdet.Shared2FCBBoxHead',
            predict_box_type='rbox',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=5,
            reg_predictor_cfg=dict(type='mmdet.Linear'),
            cls_predictor_cfg=dict(type='mmdet.Linear'),
            bbox_coder=dict(
                type='DeltaXYWHTRBBoxCoder',
                angle_version=angle_version,
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_bbox=dict(
                type='mmdet.SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='mmdet.MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBbox2HBboxOverlaps2D')),
            sampler=dict(
                type='mmdet.RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='mmdet.MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                iou_calculator=dict(type='RBboxOverlaps2D'),
                gpu_assign_thr=800,
                ignore_iof_thr=-1),
            sampler=dict(
                type='mmdet.RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms_rotated', iou_threshold=0.1),
            max_per_img=2000)))


dataset_type = 'mfod.MMDroneVehicleDataset'
data_root = '/opt/data/private/fcf/MFOD_master/data/dronevehicle/'
backend_args = None

train_pipeline = [
    dict(type='mfod.LoadPairedImageFromFile'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),    
    dict(type='mfod.PairedImagesResize', scale=(712,840), keep_ratio=True),
    dict(
        type='mfod.PairedImagesRandomFlip',
        prob=0.25,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mfod.PackedPairedDataDetInputs')
]

val_pipeline = [
    dict(type='mfod.LoadPairedImageFromFile', backend_args=backend_args),
    dict(type='mfod.PairedImagesResize', scale=(712,840), keep_ratio=True),    
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mfod.PackedPairedDataDetInputs',
        meta_keys=('img_id', 'img_path', 'img_ir_path',
                                  'ori_shape', 'img_shape','img_shape_ir', 'scale_factor'))
]

test_pipeline = [
    dict(type='mfod.LoadPairedImageFromFile', backend_args=backend_args),
    dict(type='mfod.PairedImagesResize', scale=(712,840), keep_ratio=True),  
    dict(
        type='mfod.PackedPairedDataDetInputs',
        meta_keys=('img_id', 'img_path', 'img_ir_path',
                                  'ori_shape', 'img_shape','img_shape_ir', 'scale_factor'))
]

train_dataloader = dict(
    # batch_size=4,  #too large for 24GB GPU
    # num_workers=8,
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'train/labels/',
        # data_prefix=dict(img_path=data_root + 'train/rgb/images/'),
        data_prefix=dict(img_path='train/rgb/images/', img_ir_path='train/ir/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'test/ir/labels/',
        data_prefix=dict(img_path='test/rgb/images/', img_ir_path='test/ir/images/'),
        test_mode=True,
        pipeline=val_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='DOTAMetric', metric='mAP')
test_evaluator = val_evaluator

# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    clip_grad=dict(max_norm=35, norm_type=2))


vis_backends = [dict(type='LocalVisBackend'),
                dict(
                    init_kwargs=dict(
                        group='ir_rgb',
                        name='frequence_lsk_fpn_dronevehicle_le90',
                        project='DroneVehicle'),
                    type='WandbVisBackend')]
visualizer = dict(
    type='RotLocalVisualizer', vis_backends=vis_backends, name='visualizer')