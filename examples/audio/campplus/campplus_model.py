from collections import OrderedDict

import torch

from campplus_components import (
    CAMDenseTDNNBlock,
    DenseLayer,
    FCM,
    StatsPool,
    TDNNLayer,
    TransitLayer,
    get_nonlinear,
)


class CAMPPlus(torch.nn.Module):
    def __init__(
        self,
        feat_dim=80,
        embedding_size=192,
        growth_rate=32,
        bn_size=4,
        init_channels=128,
        config_str="batchnorm-relu",
        memory_efficient=True,
        output_level="segment",
        **kwargs,
    ):
        super().__init__()

        self.head = FCM(feat_dim=feat_dim)
        channels = self.head.out_channels
        self.output_level = output_level

        self.xvector = torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        "tdnn",
                        TDNNLayer(
                            channels,
                            init_channels,
                            5,
                            stride=2,
                            dilation=1,
                            padding=-1,
                            config_str=config_str,
                        ),
                    ),
                ]
            )
        )
        channels = init_channels
        for i, (num_layers, kernel_size, dilation) in enumerate(
            zip((12, 24, 16), (3, 3, 3), (1, 2, 2))
        ):
            block = CAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=growth_rate,
                bn_channels=bn_size * growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.xvector.add_module("block%d" % (i + 1), block)
            channels = channels + num_layers * growth_rate
            self.xvector.add_module(
                "transit%d" % (i + 1),
                TransitLayer(channels, channels // 2, bias=False, config_str=config_str),
            )
            channels //= 2

        self.xvector.add_module("out_nonlinear", get_nonlinear(config_str, channels))

        if self.output_level == "segment":
            self.xvector.add_module("stats", StatsPool())
            self.xvector.add_module(
                "dense", DenseLayer(channels * 2, embedding_size, config_str="batchnorm_")
            )
        else:
            assert (
                self.output_level == "frame"
            ), "`output_level` should be set to 'segment' or 'frame'. "

    def forward(self, x, feat_mask=None, mask=None):
        # x: (B, T, F) fbank features. For fixed padded inference, pass:
        #   feat_mask: (B, 1, T) at fbank/head time resolution
        #   mask: (B, 1, ceil(T / 2)) after the first TDNN stride-2 layer
        x = x.permute(0, 2, 1)  # (B, T, F) => (B, F, T)
        x = self.head(x, mask=feat_mask)
        if feat_mask is not None:
            x = x * feat_mask

        for name, module in self.xvector.named_children():
            if isinstance(module, CAMDenseTDNNBlock):
                x = module(x, mask=mask)
            elif isinstance(module, StatsPool):
                x = module(x, mask=mask)
            else:
                x = module(x)

            if mask is not None and x.dim() == 3 and x.shape[-1] == mask.shape[-1]:
                x = x * mask

        if self.output_level == "frame":
            x = x.transpose(1, 2)
        return x  # (B, embedding_size) when output_level == 'segment'
