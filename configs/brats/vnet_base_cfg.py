# dataset settings
dataset_type = 'BraTS2018Dataset'
data_root = '/opt/data/private/project/charelchen.cj/workDir/dataset/MICCAI_BraTS17_Data_Training/'
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (2336, 3504)
crop_size = (64, 64, 64)
train_pipeline = [
    dict(type='LoadImageFromNIIFile'),
    dict(type='LoadAnnotationsFromNIIFile'),
    # dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='ExtractDataFromObj'),
    dict(type='RandomCropMedical', crop_size=crop_size, cat_max_ratio=0.85),
    dict(type='NormalizeMedical', norm_type='full_volume_mean', instensity_min_val=0.1, instensity_max_val=99.8),
    dict(type='ConcatImage'),
    # dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    # dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle3D'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=img_scale,
#         # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img'])
#         ])
# ]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16,
    train=dict(
        type='RepeatDataset',
        times=40000,
        # type='BraTS2018Dataset',
        # times=40000,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='HGG',
            ann_dir='HGG',
            pipeline=train_pipeline)),
    )


# model settings
norm_cfg = dict(type='BN3d', requires_grad=True)
conv_cfg = dict(type='Conv3d')
act_cfg = dict(type='RELU')
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='VNet',
        in_channels=4
        ),
    decode_head=dict(
        type='FCNHead3D',
        in_index=-1,
        in_channels=32,
        num_convs=1,
        channels=32,
        num_classes=5,
        norm_cfg=norm_cfg,
        conv_cfg=conv_cfg,
        # in_channels=64,
        # in_index=4,
        # channels=64,
        # num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        # num_classes=2,
        # norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5)),
    # auxiliary_head=dict(
    #     type='FCNHead3D',
    #     in_channels=64,
    #     in_index=-2,
    #     channels=64,
    #     num_convs=1,
    #     concat_input=False,
    #     dropout_ratio=0.1,
    #     num_classes=5,
    #     conv_cfg=conv_cfg,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=256, stride=170))

# optimizer
optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=100, norm_type=2))
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='mIoU')

# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
