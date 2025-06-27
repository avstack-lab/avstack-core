_base_ = [
    "../../../third_party/mmsegmentation/configs/" + "_base_/datasets/fov_bev.py",
    "../../../third_party/mmsegmentation/configs/" + "_base_/default_runtime.py",
    "../../../third_party/mmsegmentation/configs/" + "_base_/schedules/schedule_5k.py",
]

# model settings
crop_size = (256, 256)
norm_cfg = dict(type="SyncBN", requires_grad=True)
data_preprocessor = dict(type="SegDataPreProcessor")
model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type="UNet",
        in_channels=1,
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=norm_cfg,
        act_cfg=dict(type="ReLU"),
        upsample_cfg=dict(type="InterpConv"),
        norm_eval=False,
    ),
    decode_head=dict(
        type="FCNHead",
        in_channels=64,
        in_index=4,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    auxiliary_head=dict(
        type="FCNHead",
        in_channels=128,
        in_index=3,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="slide", crop_size=crop_size),
)


# scheduling configuration
# train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=36)
# val_cfg = dict(type='ValLoop')
# test_cfg = dict(type='TestLoop')
