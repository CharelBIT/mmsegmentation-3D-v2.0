
# dataset settings
dataset_type = 'BraTS2018Dataset'
data_root = '/opt/data/private/project/charelchen.cj/workDir/dataset/MICCAI_BraTS17_Data_Training'
crop_size = (64, 64, 64)
train_pipeline = [
    dict(type='LoadImageFromNIIFile'),
    dict(type='LoadAnnotationsFromNIIFile'),
    # dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='ExtractDataFromObj'),
    dict(type='NormalizeMedical', norm_type='full_volume_mean', instensity_min_val=0.5, instensity_max_val=99.5),
    dict(type='RandomCropMedicalWithForeground', crop_size=crop_size, fore_cat_ratio=0.1),
    dict(type='ConcatImage'),
    dict(type='DefaultFormatBundle3D'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromNIIFile'),
        dict(type='ExtractDataFromObj'),
        dict(type='NormalizeMedical', norm_type='full_volume_mean', instensity_min_val=0.5, instensity_max_val=99.5),
        dict(type='ConcatImage'),
        dict(type='ToTensor', keys=['img']),
        dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16,
    train=dict(
        type='RepeatDataset',
        times=40000,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='total',
            ann_dir='total',
            split='total_trainlist.txt',
            pipeline=train_pipeline)),
    test=dict(
            type='BraTS2018Dataset',
            data_root=data_root,
            split='total_testlist.txt',
            img_dir='total',
            ann_dir='total',
            pipeline=test_pipeline),
    val=dict(
            type='BraTS2018Dataset',
            data_root=data_root,
            split='total_testlist.txt',
            img_dir='total',
            ann_dir='total',
            pipeline=test_pipeline),
    )


# model settings
norm_cfg = dict(type='BN3d', requires_grad=True)
conv_cfg = dict(type='Conv3d')
act_cfg = dict(type='RELU')
num_classes = 4
use_sigmoid=True
background_as_first_channel = False
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
        kernel_size=5,
        channels=num_classes,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        conv_cfg=conv_cfg,
        act_cfg=dict(type="ELU"),
        # in_channels=64,
        # in_index=4,
        # channels=64,
        # num_convs=1,
        concat_input=False,
        dropout_ratio=0.,
        # num_classes=2,
        # norm_cfg=norm_cfg,
        align_corners=False,
        loss_cfg=[dict(type="MedicalDiceLoss",
                       classes=num_classes,
                       use_sigmoid=use_sigmoid,
                       class_weight=[1., 1., 1., 1.],
                       loss_weight=1.,
                       background_as_first_channel=background_as_first_channel),
                  dict(type="CrossEntropyLoss",
                       use_sigmoid=use_sigmoid,
                       loss_weight=1.,
                       class_weight=[1., 1., 1., 1.],
                       background_as_first_channel=background_as_first_channel,
                       ), ]),
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
    test_cfg=dict(mode='slide', crop_size=[64, 64, 64], stride=[48, 48, 48], three_dim_input=True))

# optimizer
eval_options = dict(background_as_first_channel=background_as_first_channel)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# optimizer_config = dict(type='Fp16OptimizerHook',
#                         loss_scale=512.,
#                         grad_clip=dict(max_norm=100, norm_type=2))
# learning policy
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=False)
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=1000,
    warmup_ratio=1.0 / 3,
    step=[10000, 15000])
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=5000, metric='mIoU')

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
# resume_from = 'work_dirs/vnet_vnetsegmentor/latest.pth'
resume_from=None
workflow = [('train', 1)]
cudnn_benchmark = True

# optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
