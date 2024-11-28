# dataset settings
dataset_type = 'mfod.MMDroneVehicleDataset'
data_root = '/opt/data/private/fcf/MFOD_master/data/dronevehicle/'
backend_args = None

train_pipeline = [
    dict(type='mfod.LoadPairedImageFromFile'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),    
    dict(type='mfod.PairedImagesResize', scale=(712, 840), keep_ratio=True),
    dict(
        type='mfod.PairedImagesRandomFlip',
        prob=0.5,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mfod.PackedPairedDataDetInputs')
]

val_pipeline = [
    dict(type='mfod.LoadPairedImageFromFile', backend_args=backend_args),
    dict(type='mfod.PairedImagesResize', scale=(712, 840), keep_ratio=True),    
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mfod.PackedPairedDataDetInputs',
        meta_keys=('img_id', 'img_path', 'img_ir_path',
                                  'ori_shape', 'img_shape','img_shape_ir', 'scale_factor'))
]

test_pipeline = [
    dict(type='mfod.LoadPairedImageFromFile', backend_args=backend_args),
    dict(type='mfod.PairedImagesResize', scale=(712, 840), keep_ratio=True),  
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
        data_prefix=dict(img_path='train/rgb/images/', img_ir_path='train/ir/images/'),
        test_mode=True,
        pipeline=val_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='DOTAMetric', metric='mAP')
test_evaluator = val_evaluator