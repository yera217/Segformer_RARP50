norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(type='mit_b1', style='pytorch'),
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=10,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        decoder_params=dict(embed_dim=256),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, 
            class_weight=[1.0, 1.4, 1.4, 1.13, 1.6, 2.1, 2.7, 1.2, 2.8, 1.16])),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'RARP50Dataset'
data_root = '/media/deep/Transcend/sar-rarp-dataset/'
img_norm_cfg = dict(
    mean=[60.99641042, 26.03402802, 26.96294992],
    std=[43.8721898, 30.58696675, 33.15588164],
    to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(1333, 750)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='Normalize',
        mean=[60.99641042, 26.03402802, 26.96294992],
        std=[43.8721898, 30.58696675, 33.15588164],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 750),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(
                type='Normalize',
                mean=[60.99641042, 26.03402802, 26.96294992],
                std=[43.8721898, 30.58696675, 33.15588164],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type='RARP50Dataset',
        data_root='/media/deep/Transcend/sar-rarp-dataset/',
        img_dir='traindata/rgb/training',
        ann_dir='traindata/segmentation/training',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='Resize', img_scale=(1333, 750)),
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(
                type='Normalize',
                mean=[60.99641042, 26.03402802, 26.96294992],
                std=[43.8721898, 30.58696675, 33.15588164],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='RARP50Dataset',
        data_root='/media/deep/Transcend/sar-rarp-dataset/',
        img_dir='traindata/rgb/val_small',
        ann_dir='traindata/segmentation/val_small',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 750),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[60.99641042, 26.03402802, 26.96294992],
                        std=[43.8721898, 30.58696675, 33.15588164],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='RARP50Dataset',
        data_root='/media/deep/Transcend/sar-rarp-dataset/',
        img_dir='traindata/rgb/val_small',
        ann_dir='traindata/segmentation/val_small',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 750),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[60.99641042, 26.03402802, 26.96294992],
                        std=[43.8721898, 30.58696675, 33.15588164],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'work_dirs/mitb1_pretrained_dwsampled_rarp50_160k.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=1e-02,
    betas=(0.9, 0.999),
    weight_decay=1e-02,
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0))))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='mIoU')
work_dir = './work_dirs/segformer.b1.512x512.rarp50.160k.finetune.class_weight2'
gpu_ids = range(0, 1)


