
_base_ = [
     '../_base_/schedules/schedule_2x.py',
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
        num_classes=9,
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
            num_classes=15,
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


dataset_type = 'mfod.MMVEDAIDataset'
data_root = '/opt/data/private/fcf/MFOD_master/data/VEDAI/VEDAI1024/'
backend_args = None

train_pipeline = [
    dict(type='mfod.LoadPairedImageFromFile'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),    
    dict(type='mfod.PairedImagesResize', scale=(512, 512), keep_ratio=True),
    dict(
        type='mfod.PairedImagesRandomFlip',
        prob=0.5,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mfod.PackedPairedDataDetInputs')
]

val_pipeline = [
    dict(type='mfod.LoadPairedImageFromFile', backend_args=backend_args),
    dict(type='mfod.PairedImagesResize', scale=(512, 512), keep_ratio=True),    
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mfod.PackedPairedDataDetInputs',
        meta_keys=('img_id', 'img_path', 'img_ir_path',
                                  'ori_shape', 'img_shape','img_shape_ir', 'scale_factor'))
]

test_pipeline = [
    dict(type='mfod.LoadPairedImageFromFile', backend_args=backend_args),
    dict(type='mfod.PairedImagesResize', scale=(512, 512), keep_ratio=True),  
    dict(
        type='mfod.PackedPairedDataDetInputs',
        meta_keys=('img_id', 'img_path', 'img_ir_path',
                                  'ori_shape', 'img_shape','img_shape_ir', 'scale_factor'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'train/ir/labels/',
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
        ann_file=data_root + 'test/rgb/labels/',
        data_prefix=dict(img_path='test/rgb/images/', img_ir_path='test/ir/images/'),
        test_mode=True,
        pipeline=val_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='DOTAMetric', metric='mAP')
test_evaluator = val_evaluator

# img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# train_pipeline = [
#     dict(type='LoadPairedImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='RResize', img_scale=(512, 512)),
#     dict(
#         type='RRandomFlip',
#         flip_ratio=[0.25, 0.25, 0.25],
#         direction=['horizontal', 'vertical', 'diagonal'],
#         version=angle_version),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='PairedImageDefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'img_tir', 'gt_bboxes', 'gt_labels'])
# ]

# test_pipeline = [
#     dict(type='LoadPairedImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(512, 512),
#         flip=False,
#         transforms=[
#             dict(type='RResize'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='PairedImageDefaultFormatBundle'),
#             dict(type='Collect', keys=['img', 'img_tir'])
#         ])
# ]


# data = dict(
#     samples_per_gpu=1,
#     workers_per_gpu=2,
#     train=dict(pipeline=train_pipeline, version=angle_version),
#     val=dict(version=angle_version),
#     test=dict(version=angle_version))

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)