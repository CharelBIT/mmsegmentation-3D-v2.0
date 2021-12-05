
# dataset settings
dataset_type = 'WUHANZL_ProstateDataset'
data_root = '/opt/data/private/project/charelchen.cj/workDir/dataset/wuzl_prostate/data_resample_100_1x1x1'
crop_size = (64, 64, 64)
train_pipeline = [
    dict(type='LoadImageFromNIIFile'),
    dict(type='LoadAnnotationsFromNIIFile'),
    # dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='ExtractDataFromObj'),
    # dict(type="IgnoreBlackArea", set_label=255),
    # dict(type='RandomCropMedical', crop_size=crop_size, cat_max_ratio=0.9, ignore_index=255),
    dict(type='NormalizeMedical', norm_type='full_volume_mean', instensity_min_val=0.5, instensity_max_val=99.5),
    dict(type='RandomCropMedicalWithForeground', crop_size=crop_size, fore_cat_ratio=0.1),
    dict(type='BinaryCateogry', ignore_index=255),
    dict(type='ConcatImage'),
    dict(type='DefaultFormatBundle3D'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
        dict(type='LoadImageFromNIIFile'),
        dict(type='ExtractDataFromObj'),
        dict(type='NormalizeMedical', norm_type='full_volume_mean', instensity_min_val=0.5, instensity_max_val=99.5),
        dict(type='BinaryCateogry', ignore_index=255),
        dict(type='ConcatImage'),
        dict(type='ToTensor', keys=['img']),
        dict(type='Collect', keys=['img'],
             extend_meta_keys=['image_id', 'img_affine_matrix'])
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16,
    train=dict(
        type='RepeatDataset',
        times=40000,
        dataset=dict(
            type=dataset_type,
            # data_root=data_root,
            img_dir=data_root,
            ann_dir=data_root,
            mode='T1WI',
            split='/opt/data/private/project/charelchen.cj/workDir/dataset/wuzl_prostate/trainlist_100.txt',
            pipeline=train_pipeline)),
    test=dict(
            type=dataset_type,
            # data_root=data_root,
            split='/opt/data/private/project/charelchen.cj/workDir/dataset/wuzl_prostate/testlist_100.txt',
            # split='/gruntdata/workDir/dataset/wuzl_prostate/one_test.txt',
            img_dir=data_root,
            ann_dir=data_root,
            mode='T1WI',
            pipeline=test_pipeline),
    val=dict(
            type=dataset_type,
            # data_root=data_root,
            split='/opt/data/private/project/charelchen.cj/workDir/dataset/wuzl_prostate/testlist_100.txt',
            # split='/gruntdata/workDir/dataset/wuzl_prostate/one_test.txt',
            img_dir=data_root,
            ann_dir=data_root,
            mode='T1WI',
            pipeline=test_pipeline),
    )


# model settings
# norm_cfg = dict(type='BN3d', requires_grad=True)
# conv_cfg = dict(type='Conv3d')
# act_cfg = dict(type='RELU')
# model settings
# norm_cfg = dict(type='SyncBN', requires_grad=True)
background_as_first_channel = False
use_sigmoid=True
classes = 1
model = dict(
    type='VNetSegmentor',
    pretrained=None,
    in_channels=1,
    classes=classes,
    resize_mode="trilinear",
    loss_cfg=[dict(type="MedicalDiceLoss",
                classes=classes,
                use_sigmoid=use_sigmoid,
                class_weight=[1.],
                loss_weight=1.,
                background_as_first_channel=background_as_first_channel),
              dict(type="FocalLoss",
                   use_sigmoid=use_sigmoid,
                   loss_weight=1.,
                   avg_factor='NUM_POS'
                   # class_weight=[1., 1., 1., 1.],
                   # background_as_first_channel=background_as_first_channel,
                   ), ],
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=[32, 32, 32],
                  three_dim_input=True, threshold=0.3)
    )
eval_options = dict(background_as_first_channel=background_as_first_channel)
# optimizer
optimizer = dict(type='Adam', lr=0.01)
optimizer_config = dict()
# optimizer_config = dict(type='Fp16OptimizerHook',
#                         loss_scale=512.,
#                         )
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
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000)

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
