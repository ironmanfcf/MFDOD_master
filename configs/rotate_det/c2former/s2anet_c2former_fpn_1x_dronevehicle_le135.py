
_base_ = [
    '../_base_/default_runtime.py'
]

angle_version = 'le135'
model = dict(
    type='mfod.C2Former',
    data_preprocessor=dict(
        type='mfod.PairedDetDataPreprocessor',
        mean_ir= [123.675, 116.28, 103.53],
        std_ir=[58.395, 57.12, 57.375],        
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),
    backbone=dict(
        type='mfod.C2FormerResNet',
        fmap_size=(128, 160),
        dims_in=[256, 512, 1024, 2048],
        dims_out=[96, 192, 384, 768],
        num_heads=[3, 6, 12, 24],
        cca_strides=[3, 3, 3, 3],
        groups=[1, 2, 3, 6],
        offset_range_factor=[2, 2, 2, 2],
        no_offs=[False, False, False, False],
        attn_drop_rate=0.0,
        drop_rate=0.0,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        pretrained='/opt/data/private/fcf/MFOD_master/pretrained_weights/c2former_resnet50-2stream.pth',
    ),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head_init=dict(
        type='S2AHead',
        num_classes=5,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        anchor_generator=dict(
            type='FakeRotatedAnchorGenerator',
            angle_version=angle_version,
            scales=[4],
            ratios=[1.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHTRBBoxCoder',
            angle_version=angle_version,
            norm_factor=1,
            edge_swap=False,
            proj_xy=True,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
            use_box_type=False),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='mmdet.SmoothL1Loss', beta=0.11, loss_weight=1.0)),
    bbox_head_refine=[
        dict(
            type='S2ARefineHead',
            num_classes=5,
            in_channels=256,
            stacked_convs=2,
            feat_channels=256,
            frm_cfg=dict(
                type='AlignConv',
                feat_channels=256,
                kernel_size=3,
                strides=[8, 16, 32, 64, 128]),
            anchor_generator=dict(
                type='PseudoRotatedAnchorGenerator',
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHTRBBoxCoder',
                angle_version=angle_version,
                norm_factor=1,
                edge_swap=False,
                proj_xy=True,
                target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
            loss_cls=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(
                type='mmdet.SmoothL1Loss', beta=0.11, loss_weight=1.0))
    ],
    train_cfg=dict(
        init=dict(
            assigner=dict(
                type='mmdet.MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBboxOverlaps2D')),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        refine=[
            dict(
                assigner=dict(
                    type='mmdet.MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.4,
                    min_pos_iou=0,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='RBboxOverlaps2D')),
                allowed_border=-1,
                pos_weight=-1,
                debug=False)
        ],
        stage_loss_weights=[1.0]),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000))


dataset_type = 'mfod.MMDroneVehicleDataset'
data_root = '/opt/data/private/fcf/MFOD_master/data/clip_dronevehicle/'
backend_args = None

train_pipeline = [
    dict(type='mfod.LoadPairedImageFromFile'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),    
    dict(type='mfod.PairedImagesResize', scale=(512,640), keep_ratio=True),
    dict(
        type='mfod.PairedImagesRandomFlip',
        prob=0.25,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mfod.PackedPairedDataDetInputs')
]

val_pipeline = [
    dict(type='mfod.LoadPairedImageFromFile', backend_args=backend_args),
    dict(type='mfod.PairedImagesResize', scale=(512,640), keep_ratio=True),    
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mfod.PackedPairedDataDetInputs',
        meta_keys=('img_id', 'img_path', 'img_ir_path',
                                  'ori_shape', 'img_shape','img_shape_ir', 'scale_factor'))
]

test_pipeline = [
    dict(type='mfod.LoadPairedImageFromFile', backend_args=backend_args),
    dict(type='mfod.PairedImagesResize', scale=(512,640), keep_ratio=True),  
    dict(
        type='mfod.PackedPairedDataDetInputs',
        meta_keys=('img_id', 'img_path', 'img_ir_path',
                                  'ori_shape', 'img_shape','img_shape_ir', 'scale_factor'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
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
        ann_file=data_root + 'test/labels/',
        data_prefix=dict(img_path='test/rgb/images/', img_ir_path='test/ir/images/'),
        test_mode=True,
        pipeline=val_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='DOTAMetric', metric='mAP')
test_evaluator = val_evaluator

# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
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
        end=24,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001),
    # optimizer=dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))