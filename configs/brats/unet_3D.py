
# dataset settings
dataset_type = 'BraTS2018Dataset'
data_root = '/opt/data/private/project/charelchen.cj/workDir/dataset/MICCAI_BraTS17_Data_Training/'
crop_size = (128, 128, 64)
train_pipeline = [
    dict(type='LoadImageFromNIIFile'),
    dict(type='LoadAnnotationsFromNIIFile'),
    # dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='ExtractDataFromObj'),
    dict(type="IgnoreBlackArea", set_label=255),
    dict(type='RandomCropMedical', crop_size=crop_size, cat_max_ratio=0.9, ignore_index=255),
    dict(type='NormalizeMedical', norm_type='mean', instensity_min_val=0.5, instensity_max_val=99.5),
    dict(type='ConcatImage'),
    # dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),
    # dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle3D'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
        dict(type='LoadImageFromNIIFile'),
        dict(type='ExtractDataFromObj'),
        dict(type='NormalizeMedical', norm_type='mean', instensity_min_val=0.5, instensity_max_val=99.5),
        dict(type='ConcatImage'),
        dict(type='ToTensor', keys=['img']),
        dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type='RepeatDataset',
        times=40000,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='HGG',
            ann_dir='HGG',
            pipeline=train_pipeline)),
    test=dict(
            type='BraTS2018Dataset',
            data_root=data_root,
            img_dir='HGG',
            ann_dir='HGG',
            pipeline=test_pipeline),
    val=dict(
            type='BraTS2018Dataset',
            data_root=data_root,
            img_dir='HGG',
            ann_dir='HGG',
            pipeline=test_pipeline),
    )


# model settings
norm_cfg = dict(type='BN3d', requires_grad=True)
conv_cfg = dict(type='Conv3d')
act_cfg = dict(type='RELU')
# model settings
# norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='UNet3D',
        in_channels=4,
        base_channels=32,
        num_stages=5,
        norm_cfg=norm_cfg,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=conv_cfg,
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv',
                          conv_cfg=conv_cfg,
                          upsample_cfg=dict(
                              scale_factor=2, mode='trilinear', align_corners=False)
                          ),
        norm_eval=False),
    decode_head=dict(
        type='FCNHead3D',
        in_channels=32,
        in_index=4,
        channels=32,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=5,
        norm_cfg=norm_cfg,
        conv_cfg=conv_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=[0.1, 1, 1, 1, 1])),
    auxiliary_head=dict(
        type='FCNHead3D',
        in_channels=64,
        in_index=3,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=5,
        norm_cfg=norm_cfg,
        conv_cfg=conv_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.5)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=[128, 128, 64], stride=[96, 96, 48], three_dim_input=True))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(type='Fp16OptimizerHook',
                        loss_scale=512.,
                        )
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='mIoU')

# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
# work_dirs =
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

# optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
