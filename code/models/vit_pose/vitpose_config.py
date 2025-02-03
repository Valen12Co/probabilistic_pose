evaluation = dict()
target_type = 'GaussianHeatmap'

channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=list(range(17)),
    inference_channel=list(range(17)))

# model settings
model = dict(
    type='TopDown',
    pretrained='/mnt/vita/scratch/vita-students/users/perret/probabilistic_pose/code/models/vit_pose/pretrained_weigths/vitpose-b-multi-coco.pth',
    backbone=dict(
        type='ViT',
        img_size=(256, 256),
        patch_size=16,
        embed_dim=64,
        depth=12,
        num_heads=12,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.3,
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=64,
        num_deconv_layers=2,
        num_deconv_filters=(64, 64),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1,),
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=None),
    train_cfg=dict(),
    test_cfg=dict())