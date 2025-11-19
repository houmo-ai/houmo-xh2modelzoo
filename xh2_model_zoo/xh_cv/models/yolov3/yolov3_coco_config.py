from hmquant.configs.api_config import *

# yolov5 for coco example
yolov3_coco_input_config = InputConfig(
    data_format="RGB",
    first_layer_weight_denorm_mean=[0, 0, 0],
    first_layer_weight_denorm_std=[1, 1, 1],
    resizer_crop={"top": 0, "left": 0, "height": 640, "width": 640},
    resizer_resize={
        "height": 640,
        "width": 640,
        "align_corners": False,
        "method": "bilinear",
    },
    # toYUV_format="YUV444"
)

yolov3_coco_input_config_int = InputConfig(
    data_format="Int8Feature",
    quantize={"quanted": False, "quant_type": "int8", "quant_scale": None, "quant_method": "Min_Max"},
    first_layer_weight_denorm_mean=[0, 0, 0],
    first_layer_weight_denorm_std=[1, 1, 1],
)


class AvoidLSBRandomnessConv2d(KLCIMDConv2d):
    def quant_forward(self, x):
        assert x.quant_param.dtype == "int8"
        x_q = x.quant_param.quant_tensor(x, simulate=False)
        # explicitly set LSB to 0
        x_flip = (x_q >> 1) << 1
        x_rand_sim = x.quant_param.integer_to_float(x_flip)
        x_rand_sim = x + (x_rand_sim - x).detach()  # STE
        x_rand_sim.quant_param = x.quant_param

        return super().quant_forward(x_rand_sim)

    def advanced_quant_forward(self, x):
        assert NotImplementedError()


class AvoidLSBRandomnessConfig(KLConfig):
    def qconv_class(self, *args, **kwargs):
        module = AvoidLSBRandomnessConv2d(*args, **kwargs, w_bit=8, o_bit=8, bias_bit=16, w_channelwise=True)
        return module


config = APIConfig(
    inputs_config={"ALL": yolov3_coco_input_config},
    quant_config=KLConfig(),
    graph_optimization={
        "save_fx_model": False,
        "auto_quant_flag": True,
        "fuse_conv_relu": False,
        "return_fuse_onnx": False,
    },
)
