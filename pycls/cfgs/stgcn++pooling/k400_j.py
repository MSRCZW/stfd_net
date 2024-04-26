model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='newSTGCN_group',
        gcn_adaptive='init',
        gcn_with_res=True,
        num_stages=10,
        inflate_stages=[5, 8],
        down_stages=[5, 8],
        tcn_type='mstcn',
        graph_cfg=dict(layout='coco', mode='spatial')),
    pbackbone = dict(
        type='Part_STGCN_group',
        gcn_adaptive='init',
        gcn_with_res=True,
        num_stages=10,
        inflate_stages=[5, 8],
        tcn_type='mstcn',
        focus_size = 0,
        graph_cfg=dict(layout='coco', mode='spatial')),
    cls_head=dict(type='GCNHead_GRU', num_classes=400, in_channels=256),
    prime_cls_head=dict(type='GCNHead', num_classes=400, in_channels=256),
    p1_cls_head=dict(type='GCNHead', num_classes=400, in_channels=256),
    p2_cls_head=dict(type='GCNHead', num_classes=400, in_channels=256),
)


dataset_type = 'PoseDataset'
ann_file = './data/k400_preprocessed.pkl'
train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='RandomScale', scale=0.1),
    dict(type='GenSkeFeat', dataset='coco', feats=['bm']),
    dict(type='UniformSample', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]


val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['bm']),
    dict(type='UniformSample', clip_len=100, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['bm']),
    dict(type='UniformSample', clip_len=100, num_clips=10, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
memcached = False
valid_ratio = 0.1
box_thr = 0.7
data = dict(
    videos_per_gpu=128,
    workers_per_gpu=16,
    test_dataloader=dict(videos_per_gpu=160),
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='train',memcached = memcached,
                     box_thr=box_thr, valid_ratio=valid_ratio)),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='val',memcached = memcached),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='val',memcached = memcached))


# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 25
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

# runtime settings
log_level = 'INFO'
GE_frozen_stages = [1]
work_dir = './work_dirs/k400/b1'