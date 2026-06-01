import logging
import os
from pathlib import Path
import numpy as np

from post_decode import post_decode
from utils import OrtInferSession 

logger = logging.getLogger()
def hi_load_model(model_dir, quantize=False,dev_id = -1,intra_op_num_threads=1,model_name="model"):
    if not Path(model_dir).exists():
        raise f"model dir [{model_dir}] does not exist."
    
    non_quant_model_path = os.path.join(model_dir, f"{model_name}.onnx")
          
       
    #downgrade to find onnx model if bmodel not exist
    if quantize:
        #for cpu, use non-quant model if it exists
        if os.path.exists(non_quant_model_path) and not OrtInferSession.is_cuda_available():
            model_path = non_quant_model_path
        else:
            model_path =  os.path.join(model_dir, f"{model_name}_quant.onnx")
            if not os.path.exists(model_path):
                model_path = non_quant_model_path
    else:
        model_path =  non_quant_model_path
        
    if not os.path.exists(model_path):
        print(f"ONNX model does not exist. Please check the model path. {model_path}")
        raise f"ONNX model does not exist. Please check the model path."
    
    return OrtInferSession(model_path,dev_id,intra_op_num_threads)       




if __name__ == "__main__":
    # decode_out = np.load("decoder_out.npy")
    # token_nums = np.load("pre_token_length.npy")
    # preds = post_decode(decode_out, token_nums)
    # print(''.join(preds[0]))
    
    model_dir = "weights/huachuang"
    encoder = hi_load_model(model_dir, model_name="encoder")
    predictor = hi_load_model(model_dir, model_name="predictor")
    decoder = hi_load_model(model_dir,model_name="decoder")
    
    speech = np.load("weights/huachuang/inputs/speech.npy")
    # speech_len = np.load("speech_length.npy")
    speech_len = np.array([speech.shape[1]], dtype=np.int32)

    enc,mask = encoder([speech, speech_len]) 
        
    # predictor 是动态图，算出来的shape是[1,X,512], 理论上能算出  (1, 92, 512)    
    pre_acoustic_embeds, pre_token_length = predictor([enc,mask])
       
    if pre_token_length[0] == 1:
        logger.info("No words detected.")
        exit(0)
    
    #目前是固定shape，所以保护一下，可以直接load样本中的数据跳过predictor
    if pre_token_length[0] != 92:    
        logger.error("Unexpected token length.")
        exit(0)
        
    # bias_embed = np.load("bias_embed.npy")
        
    [decoder_out] = decoder([enc,speech_len,pre_acoustic_embeds,pre_token_length])       
    
    preds = post_decode(decoder_out, pre_token_length)
    print(''.join(preds[0]))
