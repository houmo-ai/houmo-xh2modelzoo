import onnx

onnx_path = "/data01/home/xuchen/xh2/xh2_model_zoo/data/model_zoo2/houmo/vit/vit_small_patch16_224.onnx"
save_path=  "/data01/home/xuchen/xh2/xh2_model_zoo/data/model_zoo2/houmo/vit/vit_small_patch16_224_extract.onnx"

input_names = ['images']
output_names = ['/Gather_output_0']

onnx.utils.extract_model(onnx_path, save_path, input_names, output_names)