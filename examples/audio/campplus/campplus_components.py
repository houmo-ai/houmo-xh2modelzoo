import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp


# Controls how CAMLayer.seg_pooling upsamples segments for ONNX export.
#   "interpolate" -> Resize op, correct for dynamic time (general onnx)
#   "expand"      -> Expand/Reshape, converter-safe & exact at FIXED shape (hmonnx)
# Set this from the export script before building the model.
SEG_POOLING_ONNX_MODE = "interpolate"


class BasicResBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=(stride, 1), padding=1, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias=False,
                ),
                torch.nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x, mask=None):
        out = F.relu(self.bn1(self.conv1(x)))
        if mask is not None:
            out = out * mask
        out = self.bn2(self.conv2(out))
        if mask is not None:
            out = out * mask
        shortcut = self.shortcut(x)
        if mask is not None:
            shortcut = shortcut * mask
        out += shortcut
        out = F.relu(out)
        if mask is not None:
            out = out * mask
        return out


class FCM(torch.nn.Module):
    def __init__(self, block=BasicResBlock, num_blocks=[2, 2], m_channels=32, feat_dim=80):
        super(FCM, self).__init__()
        self.in_planes = m_channels
        self.conv1 = torch.nn.Conv2d(1, m_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(m_channels)

        self.layer1 = self._make_layer(block, m_channels, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, m_channels, num_blocks[0], stride=2)

        self.conv2 = torch.nn.Conv2d(
            m_channels, m_channels, kernel_size=3, stride=(2, 1), padding=1, bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(m_channels)
        self.out_channels = m_channels * (feat_dim // 8)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return torch.nn.Sequential(*layers)

    def forward(self, x, mask=None):
        mask2d = None if mask is None else mask.unsqueeze(2)
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        if mask2d is not None:
            out = out * mask2d
        for block in self.layer1:
            out = block(out, mask=mask2d)
        for block in self.layer2:
            out = block(out, mask=mask2d)
        out = F.relu(self.bn2(self.conv2(out)))
        if mask2d is not None:
            out = out * mask2d

        shape = out.shape
        out = out.reshape(shape[0], shape[1] * shape[2], shape[3])
        return out


def get_nonlinear(config_str, channels):
    nonlinear = torch.nn.Sequential()
    for name in config_str.split("-"):
        if name == "relu":
            nonlinear.add_module("relu", torch.nn.ReLU(inplace=True))
        elif name == "prelu":
            nonlinear.add_module("prelu", torch.nn.PReLU(channels))
        elif name == "batchnorm":
            nonlinear.add_module("batchnorm", torch.nn.BatchNorm1d(channels))
        elif name == "batchnorm_":
            nonlinear.add_module("batchnorm", torch.nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError("Unexpected module ({}).".format(name))
    return nonlinear


def statistics_pooling(x, dim=-1, keepdim=False, unbiased=True, eps=1e-2, mask=None):
    if mask is None:
        mean = x.mean(dim=dim)
        std = x.std(dim=dim, unbiased=unbiased)
    else:
        den = mask.sum(dim=dim).clamp_min(1.0)
        mean = (x * mask).sum(dim=dim) / den
        centered = (x - mean.unsqueeze(dim)) * mask
        if unbiased:
            var_den = (den - 1.0).clamp_min(1.0)
        else:
            var_den = den
        var = (centered * centered).sum(dim=dim) / var_den
        std = torch.sqrt(var.clamp_min(eps * eps))
    stats = torch.cat([mean, std], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


class StatsPool(torch.nn.Module):
    def forward(self, x, mask=None):
        return statistics_pooling(x, mask=mask)


class TDNNLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
    ):
        super(TDNNLayer, self).__init__()
        if padding < 0:
            assert (
                kernel_size % 2 == 1
            ), "Expect equal paddings, but got even kernel size ({})".format(kernel_size)
            padding = (kernel_size - 1) // 2 * dilation
        self.linear = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        x = self.linear(x)
        x = self.nonlinear(x)
        return x


class CAMLayer(torch.nn.Module):
    def __init__(
        self, bn_channels, out_channels, kernel_size, stride, padding, dilation, bias, reduction=2
    ):
        super(CAMLayer, self).__init__()
        self.linear_local = torch.nn.Conv1d(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.linear1 = torch.nn.Conv1d(bn_channels, bn_channels // reduction, 1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.linear2 = torch.nn.Conv1d(bn_channels // reduction, out_channels, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask
        y = self.linear_local(x)
        if mask is None:
            global_context = x.mean(-1, keepdim=True)
            seg_context = self.seg_pooling(x)
        else:
            den = mask.sum(-1, keepdim=True).clamp_min(1.0)
            global_context = (x * mask).sum(-1, keepdim=True) / den
            seg_context = self.seg_pooling(x, mask=mask)
        context = global_context + seg_context
        context = self.relu(self.linear1(context))
        m = self.sigmoid(self.linear2(context))
        out = y * m
        if mask is not None:
            out = out * mask
        return out

    def seg_pooling(self, x, seg_len=100, stype="avg", mask=None):
        if stype == "avg":
            # count_include_pad=False so the partial last window (from ceil_mode)
            # is divided by its valid element count, matching torch's own
            # avg_pool1d result. Without it torch.onnx emits count_include_pad=1,
            # making ONNX Runtime divide by the full kernel -> wrong for any T
            # whose tdnn output is not an exact multiple of seg_len.
            if mask is None:
                seg = F.avg_pool1d(
                    x, kernel_size=seg_len, stride=seg_len, ceil_mode=True,
                    count_include_pad=False,
                )
            else:
                num = F.avg_pool1d(
                    x * mask, kernel_size=seg_len, stride=seg_len, ceil_mode=True,
                    count_include_pad=False,
                )
                den = F.avg_pool1d(
                    mask, kernel_size=seg_len, stride=seg_len, ceil_mode=True,
                    count_include_pad=False,
                ).clamp_min(1.0 / float(seg_len))
                seg = num / den
        elif stype == "max":
            seg = F.max_pool1d(x, kernel_size=seg_len, stride=seg_len, ceil_mode=True)
        else:
            raise ValueError("Wrong segment pooling type.")
        # Upsample each segment value back over its seg_len frames. Two
        # numerically-identical implementations, selected by SEG_POOLING_ONNX_MODE:
        #
        # "interpolate" (default): integer-factor nearest upsampling -> exports to
        #   a Resize op with dynamic time. Correct for ANY input length. Use for the
        #   general dynamic-axis ONNX (CPU / onnxruntime deployment).
        #
        # "expand": the original FunASR form (Expand + Reshape). It unpacks
        #   seg.shape into python ints, so torch.onnx bakes the traced length as a
        #   constant -> only valid when exported at a FIXED shape. Use this for the
        #   HMONNX/NPU path (fixed seq_len), where it (a) is exact and (b) matches
        #   the converter-safe op set of the shipped CosyVoice3 campplus.onnx
        #   (Expand/Reshape, no Resize).
        if SEG_POOLING_ONNX_MODE == "expand":
            shape = seg.shape
            seg = seg.unsqueeze(-1).expand(*shape, seg_len).reshape(*shape[:-1], -1)
        elif SEG_POOLING_ONNX_MODE == "interpolate":
            seg = F.interpolate(seg, scale_factor=float(seg_len), mode="nearest",
                                recompute_scale_factor=False)
        else:
            raise ValueError(f"Unknown SEG_POOLING_ONNX_MODE: {SEG_POOLING_ONNX_MODE}")
        seg = seg[..., : x.shape[-1]]
        if mask is not None:
            seg = seg * mask
        return seg


class CAMDenseTDNNLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bn_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
        memory_efficient=False,
    ):
        super(CAMDenseTDNNLayer, self).__init__()
        assert kernel_size % 2 == 1, "Expect equal paddings, but got even kernel size ({})".format(
            kernel_size
        )
        padding = (kernel_size - 1) // 2 * dilation
        self.memory_efficient = memory_efficient
        self.nonlinear1 = get_nonlinear(config_str, in_channels)
        self.linear1 = torch.nn.Conv1d(in_channels, bn_channels, 1, bias=False)
        self.nonlinear2 = get_nonlinear(config_str, bn_channels)
        self.cam_layer = CAMLayer(
            bn_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def bn_function(self, x):
        return self.linear1(self.nonlinear1(x))

    def forward(self, x, mask=None):
        if self.training and self.memory_efficient:
            x = cp.checkpoint(self.bn_function, x)
        else:
            x = self.bn_function(x)
        x = self.cam_layer(self.nonlinear2(x), mask=mask)
        return x


class CAMDenseTDNNBlock(torch.nn.ModuleList):
    def __init__(
        self,
        num_layers,
        in_channels,
        out_channels,
        bn_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=False,
        config_str="batchnorm-relu",
        memory_efficient=False,
    ):
        super(CAMDenseTDNNBlock, self).__init__()
        for i in range(num_layers):
            layer = CAMDenseTDNNLayer(
                in_channels=in_channels + i * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                config_str=config_str,
                memory_efficient=memory_efficient,
            )
            self.add_module("tdnnd%d" % (i + 1), layer)

    def forward(self, x, mask=None):
        for layer in self:
            x = torch.cat([x, layer(x, mask=mask)], dim=1)
        return x


class TransitLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, config_str="batchnorm-relu"):
        super(TransitLayer, self).__init__()
        self.nonlinear = get_nonlinear(config_str, in_channels)
        self.linear = torch.nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.nonlinear(x)
        x = self.linear(x)
        return x


class DenseLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, config_str="batchnorm-relu"):
        super(DenseLayer, self).__init__()
        self.linear = torch.nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.nonlinear = get_nonlinear(config_str, out_channels)

    def forward(self, x):
        if len(x.shape) == 2:
            x = self.linear(x.unsqueeze(dim=-1)).squeeze(dim=-1)
        else:
            x = self.linear(x)
        x = self.nonlinear(x)
        return x
