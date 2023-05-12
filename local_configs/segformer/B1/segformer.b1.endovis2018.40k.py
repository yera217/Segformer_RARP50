norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_pretrained_endovis2018_48k.pth',
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
            class_weight=[1.2536, 45.4545, 27.7778, 8.3333, 250, 100, 200, 500, 666.667, 555.555])),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'RARP50Dataset'
data_root = '/media/deep/Transcend/sar-rarp-dataset/'
img_norm_cfg = dict(
    mean = [85.7148, 40.4981, 39.0133], 
    std = [51.8542, 39.9905,41.0522],
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
        mean = [85.7148, 40.4981, 39.0133], 
        std = [51.8542, 39.9905,41.0522],
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
                mean = [85.7148, 40.4981, 39.0133], 
                std = [51.8542, 39.9905,41.0522],
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
        img_dir=['traindata/video_1.zip/rgb',
                'traindata/video_2.zip/rgb',
                'traindata/video_3.zip/rgb',
                'traindata/video_4.zip/rgb',
                'traindata/video_6.zip/rgb',
                'traindata/video_7.zip/rgb',
                'traindata/video_8.zip/rgb',
                'traindata/video_9.zip/rgb',
                'traindata/video_10.zip/rgb',
                'traindata/video_11_2.zip/rgb',
                'traindata/video_11.zip/rgb',
                'traindata/video_13.zip/rgb',
                'traindata/video_15_1.zip/rgb',
                'traindata/video_15_2.zip/rgb',
                'traindata/video_16.zip/rgb',
                'traindata/video_17_1.zip/rgb',
                'traindata/video_17_2.zip/rgb',
                'traindata/video_18.zip/rgb',
                'traindata/video_20.zip/rgb',
                'traindata/video_21.zip/rgb',
                'traindata/video_22.zip/rgb',
                'traindata/video_23.zip/rgb',
                'traindata/video_24.zip/rgb',
                'traindata/video_25.zip/rgb',
                'traindata/video_26.zip/rgb',
                'traindata/video_27.zip/rgb',
                'traindata/video_28.zip/rgb',
                'traindata/video_29_2.zip/rgb',
                'traindata/video_30.zip/rgb',
                'traindata/video_31.zip/rgb',
                'traindata/video_32.zip/rgb',
                'traindata/video_33.zip/rgb',
                'traindata/video_34.zip/rgb',
                'traindata/video_36.zip/rgb',
                'traindata/video_37.zip/rgb',
                'traindata/video_38.zip/rgb',
                'traindata/video_39.zip/rgb',
                'traindata/video_40.zip/rgb'],
        ann_dir=['traindata/video_1.zip/segmentation',
                'traindata/video_2.zip/segmentation',
                'traindata/video_3.zip/segmentation',
                'traindata/video_4.zip/segmentation',
                'traindata/video_6.zip/segmentation',
                'traindata/video_7.zip/segmentation',
                'traindata/video_8.zip/segmentation',
                'traindata/video_9.zip/segmentation',
                'traindata/video_10.zip/segmentation',
                'traindata/video_11_2.zip/segmentation',
                'traindata/video_11.zip/segmentation',
                'traindata/video_13.zip/segmentation',
                'traindata/video_15_1.zip/segmentation',
                'traindata/video_15_2.zip/segmentation',
                'traindata/video_16.zip/segmentation',
                'traindata/video_17_1.zip/segmentation',
                'traindata/video_17_2.zip/segmentation',
                'traindata/video_18.zip/segmentation',
                'traindata/video_20.zip/segmentation',
                'traindata/video_21.zip/segmentation',
                'traindata/video_22.zip/segmentation',
                'traindata/video_23.zip/segmentation',
                'traindata/video_24.zip/segmentation',
                'traindata/video_25.zip/segmentation',
                'traindata/video_26.zip/segmentation',
                'traindata/video_27.zip/segmentation',
                'traindata/video_28.zip/segmentation',
                'traindata/video_29_2.zip/segmentation',
                'traindata/video_30.zip/segmentation',
                'traindata/video_31.zip/segmentation',
                'traindata/video_32.zip/segmentation',
                'traindata/video_33.zip/segmentation',
                'traindata/video_34.zip/segmentation',
                'traindata/video_36.zip/segmentation',
                'traindata/video_37.zip/segmentation',
                'traindata/video_38.zip/segmentation',
                'traindata/video_39.zip/segmentation',
                'traindata/video_40.zip/segmentation'],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='Resize', img_scale=(1333, 750)),
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(
                type='Normalize',
                mean = [85.7148, 40.4981, 39.0133], 
                std = [51.8542, 39.9905,41.0522],
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
                        mean = [85.7148, 40.4981, 39.0133], 
                        std = [51.8542, 39.9905,41.0522],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='RARP50Dataset',
        data_root='/media/deep/Transcend/sar-rarp-dataset/',
        img_dir='traindata/rgb/val',
        ann_dir='traindata/segmentation/val',
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
                        mean = [85.7148, 40.4981, 39.0133], 
                        std = [51.8542, 39.9905,41.0522],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'pretrained/mit_pretrained_endovis2018_48k.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=1e-04,
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
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=500, metric='mIoU')
work_dir = './work_dirs/segformer.b1.endovis2018.40k'
gpu_ids = range(0, 1)


