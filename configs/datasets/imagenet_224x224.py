dataset_type = "ImageNet"
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="ResizeEdge",
        scale=256,
        edge="short",
        backend="pillow",
        interpolation="bicubic",
    ),
    dict(type="CenterCrop", crop_size=224),
    dict(type="ClsPackInputs"),
]
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root="/data02/datasets/imagenet",
        split="val",
        pipeline=test_pipeline,
        test_mode=False,
        lazy_init=False,
    ),
    sampler=dict(type="DefaultSampler", shuffle=False),
)
test_dataloader = val_dataloader
val_evaluator = dict(type="Accuracy", topk=(1, 5))
test_evaluator = val_evaluator
