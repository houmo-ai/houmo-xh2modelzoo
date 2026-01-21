from pydantic import ConfigDict
import argparse,os.path as osp,torch
from pathlib import Path
from xhquant import export
from xhquant.api import DeviceType, xhquant_init, QuantScheme, get_root_logger, Config, release_quanted_model_unused_parameters  # isort:skip
from xh_model_zoo.utils import MemoryTracker, TimeProfiler
from xh_model_zoo.xh_llm import LLMConverter
from xh_model_zoo.xh_llm.models.deepseek_ocr.deepseekv2_converter import DeepseekV2ConverterConfig
import math

class vision_processor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.sam_model = model.sam_model
        self.vision_model = model.vision_model
        self.projector = model.projector
        self.image_newline = model.image_newline
        self.view_seperator = model.view_seperator

    def forward(self, image_ori):
        images_in_this_batch=[]

        global_features_1 = self.sam_model(image_ori)
        global_features_2 = self.vision_model(image_ori, global_features_1) 
        global_features = torch.cat((global_features_2[:, 1:], global_features_1.flatten(2).permute(0, 2, 1)), dim=-1) 
        global_features = self.projector(global_features)


        return global_features

class vision_processor_gundam(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.sam_model = model.sam_model
        self.vision_model = model.vision_model
        self.projector = model.projector
        self.image_newline = model.image_newline
        self.view_seperator = model.view_seperator

    def forward(self, patches,image_ori):
        images_in_this_batch=[]

        local_features_1 = self.sam_model(patches)

        local_features_2 = self.vision_model(patches, local_features_1)  
        # vit_time = time.time()
        local_features = torch.cat((local_features_2[:, 1:], local_features_1.flatten(2).permute(0, 2, 1)), dim=-1) 
        local_features = self.projector(local_features)


        global_features_1 = self.sam_model(image_ori)
        global_features_2 = self.vision_model(image_ori, global_features_1) 
        global_features = torch.cat((global_features_2[:, 1:], global_features_1.flatten(2).permute(0, 2, 1)), dim=-1) 
        global_features = self.projector(global_features)


        return local_features, global_features


def export_hmonnx(model, inputs, onnx_name, handles_wrap, args):

    import tempfile
    import onnx
    from xhquant.utils.onnxsim_large_model.simplify_large_onnx import simplify_large_onnx
    from xhquant.api import (  # isort:skip
        Config,
        DeviceType,
        ConfigDict,
        get_root_logger,
        create_quant_config,
        is_ssfp_quant_config,
        CacheTensor,
        convert_fx_model_to_quanted_model, 
        convert_onnx_to_hmonnx, 
        convert_quanted_model_to_hmonnx,
    )
    from xh_model_zoo.xh_llm.models.builder import wrap_llm_model
    from xhquant.api.quant_type import QuantScheme


    if args.export_mode == "Tiny":
        input_resolution=512
    args.out_hmonnx_file = args.out_dirs + f"{onnx_name}_XH2a_{args.export_mode}_{args.quant_type}"
    logger = get_root_logger()
    logger.info(f"********************* start export {onnx_name} model *********************")
    model.eval()
    model.cpu()
    wrap_cfg = ConfigDict(
        input_resolution=input_resolution,
    )
    handles_wrap(model)
    wraped_model = wrap_llm_model(model, wrap_cfg)
    export_onnx(wraped_model, inputs, onnx_name, args)
    # quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=args.quant_type)
    # quant_config = create_quant_config(quant_scheme)
    # quanted_model = convert_fx_model_to_quanted_model(
    #         wraped_model,
    #         inputs,
    #         quant_scheme.target_device,
    #         quant_config=quant_config,
    #     )
    # quanted_model = release_quanted_model_unused_parameters(quanted_model)
    # convert_quanted_model_to_hmonnx(quanted_model, inputs, out_hmonnx_file)

    # logger.info(f"Export {onnx_name} model to {out_hmonnx_file}")


def export_onnx(model, inputs, onnx_name, args):
    import tempfile
    import onnx
    from xhquant.utils.onnxsim_large_model.simplify_large_onnx import simplify_large_onnx
    from xhquant.api import (  # isort:skip
        Config,
        DeviceType,
        ConfigDict,
        get_root_logger,
        create_quant_config,
        is_ssfp_quant_config,
        CacheTensor,
        convert_fx_model_to_quanted_model, 
        convert_onnx_to_hmonnx, 
        convert_quanted_model_to_hmonnx,
    )
    logger = get_root_logger()
    logger.info(f"********************* start export {onnx_name} model *********************")
    model.eval()
    model.cpu()
    # wrap_cfg = dict(
    #     max_size=args.input_resolution,
    # )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_onnx_file = str(Path(tmp_dir) / f"{onnx_name}.onnx")
        if len(inputs) == 1:
            torch.onnx.export(
                model,
                inputs[0],
                tmp_onnx_file,
                export_params=True,
                opset_version=18,
                do_constant_folding=True,
                input_names=[],
                output_names=[],
                verbose=False,
            )
        else:
            torch.onnx.export(
                model,
                tuple(inputs),
                tmp_onnx_file,
                export_params=True,
                opset_version=18,
                do_constant_folding=True,
                input_names=[],
                output_names=[],
                verbose=False,
            )        
        onnx_model = onnx.load(tmp_onnx_file, load_external_data=True)
        logger.info(f"simplify onnx model............")
        onnx_model, check = simplify_large_onnx(onnx_model)
        out_hmonnx_file = args.out_hmonnx_file + f"/{onnx_name}.onnx"
        quant_config = dict(
            # inputs=dict(
            #     image_ori=dict(
            #         quantizer=dict(
            #             qspec=dict(fake_dtype="float16"),
            #         )
            #     ),
            # )
        )
        quant_config = ConfigDict(quant_config)

        convert_onnx_to_hmonnx(
            onnx_model,
            inputs,
            DeviceType.XH2a,
            out_hmonnx_file,
            quant_config,
        )
    logger.info(f"Export {onnx_name} model to {out_hmonnx_file}")

