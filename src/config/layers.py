
fc_small = (
    1024,
    "out",
    1024,
    "out",
    1024,
)

fc_medium = (
    1024,
    1024,
    "out",
    2048,
    2048,
    "out",
    4094,
    4094,
    "out",
    4094,
)

cnn_small = (
    ("conv", 5, 5, 32, "relu"),
    ("conv", 5, 5, 32, None),
    ("bn",),
    ("relu",),
    ("out",),

    ("pool", 2, 2),
    ("conv", 3, 3, 32, "relu"),
    ("conv", 3, 3, 32, None),
    ("bn",),
    ("relu",),
    ("pool", 2, 2),
    ("out",),
)

cnn_small_transposed = lambda out_channels=1: (
    ("upsample", 2),
    ("tconv", 3, 3, 32, "relu"),
    ("tconv", 3, 3, 32, None),
    ("bn",),
    ("relu",),

    ("upsample", 2),
    ("tconv", 5, 5, 32, "relu"),
    ("tconv", 5, 5, out_channels, None),
)

cnn_medium = (
    ("conv", 3, 3, 64, "relu"),
    ("conv", 3, 3, 64, None),
    ("bn",),
    ("relu",),
    ("out",),

    ("conv", 3, 3, 64, "relu"),
    ("conv", 3, 3, 64, None),
    ("bn",),
    ("relu",),
    ("out",),

    ("pool", 2, 2),

    ("conv", 3, 3, 64, "relu"),
    ("conv", 3, 3, 64, "relu"),
    ("conv", 3, 3, 64, None),
    ("bn",),
    ("relu",),
    ("out",),
)

cnn_medium_transposed = lambda out_channels=1: (
    ("tconv", 3, 3, 64, "relu"),
    ("tconv", 3, 3, 64, "relu"),
    ("tconv", 3, 3, 64, None),

    ("upsample", 2),
    ("bn",),
    ("relu",),
    ("tconv", 3, 3, 64, "relu"),
    ("tconv", 3, 3, 64, None),

    ("bn",),
    ("relu",),
    ("tconv", 3, 3, 64, "relu"),
    ("tconv", 3, 3, out_channels, None),
)

cnn_1d_medium = (
    ("conv1d", dict(out_channels=32, kernel_size=11, stride=1, dilation=2)),
    ("relu",),
    ("conv1d", dict(out_channels=32, kernel_size=11, stride=1, dilation=2)),
    ("relu",),
    # ("conv1d", dict(out_channels=32, kernel_size=11, stride=1, dilation=2)),
    # ("relu",),
    ("pool1d", dict(kernel_size=2, stride=2)),
    ("out",),

    ("conv1d", dict(out_channels=32, kernel_size=5, stride=1)),
    ("relu",),
    ("conv1d", dict(out_channels=32, kernel_size=5, stride=1)),
    ("relu",),
    ("conv1d", dict(out_channels=32, kernel_size=5, stride=1)),
    # ("relu",),
    # ("pool1d", dict(kernel_size=2, stride=2)),
    ("out",),
)
