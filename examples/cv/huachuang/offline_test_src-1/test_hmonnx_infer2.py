import logging
import os
from pathlib import Path
import numpy as np
import torch
from post_decode import post_decode
from utils import OrtInferSession 
from xhquant.api import HMONNXGoldenInference

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


def hmonnx_load_model(model_path, model_name="model"):
    return HMONNXGoldenInference(model_path).to("cuda")


if __name__ == "__main__":
    # decode_out = np.load("decoder_out.npy")
    # token_nums = np.load("pre_token_length.npy")
    # preds = post_decode(decode_out, token_nums)
    # print(''.join(preds[0]))
    
    model_dir = "weights/huachuang"
    encoder_hmonnx_path = "work_dirs/encoder_sim_XH2a_quanted_debug/encoder_sim_XH2a.onnx"
    encoder = hmonnx_load_model(encoder_hmonnx_path)
    predictor = hi_load_model(model_dir, model_name="predictor")
    # decoder_hmonnx_path = "/home/jiangyong.yu/xh2_work/export/work_dirs/decoder_sim_XH2a_quanted_debug/decoder_sim_XH2a.onnx"
    decoder = hi_load_model("/home/jiangyong.yu/.cache/modelscope/hub/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", model_name="decoder_sim")
    
    speech = np.load("weights/huachuang/inputs/speech.npy")
    # speech_len = np.load("speech_length.npy")
    speech_mask = torch.ones(speech.shape[0], speech.shape[1], dtype=torch.float16)
    speech_len = np.array([speech.shape[1]], dtype=np.int32)
    speech = torch.from_numpy(speech).cuda().half()
    enc  = encoder(speech, speech_mask).float().detach().cpu().numpy() 

    np.save("weights/huachuang/inputs/enc.npy", enc)
    np.save("weights/huachuang/inputs/enc_mask.npy", speech_mask.unsqueeze(0).float().detach().cpu().numpy())
        
    # predictor 是动态图，算出来的shape是[1,X,512], 理论上能算出  (1, 92, 512)    
    pre_acoustic_embeds, pre_token_length = predictor([enc, speech_mask.unsqueeze(0).float().detach().cpu().numpy() ])
       
    pre_acoustic_embeds_pad = np.zeros((1, 100, 512), dtype=np.float32)
    pre_acoustic_embeds_pad[:pre_acoustic_embeds.shape[0], :pre_acoustic_embeds.shape[1], :] = pre_acoustic_embeds
    np.save("weights/huachuang/inputs/pre_acoustic_embeds.npy", pre_acoustic_embeds_pad)

    pre_token_mask = np.zeros((1, 100), dtype=np.float32)
    pre_token_mask[:, :pre_token_length.item()] = 1
    np.save("weights/huachuang/inputs/pre_token_mask.npy", pre_token_mask)


    if pre_token_length[0] == 1:
        logger.info("No words detected.")
        exit(0)
    
    #目前是固定shape，所以保护一下，可以直接load样本中的数据跳过predictor
    if pre_token_length[0] != 92:    
        logger.error("Unexpected token length.")
        exit(0)
        
    # bias_embed = np.load("bias_embed.npy")
    # enc = torch.from_numpy(enc).half()
    # enc_mask = torch.from_numpy(speech_mask.unsqueeze(0).float().detach().cpu().numpy()).half()
    # pre_acoustic_embeds = torch.from_numpy(pre_acoustic_embeds_pad).half()
    # pre_token_mask = torch.from_numpy(pre_token_mask).half()
    
    decoder_out = decoder([enc, speech_mask.unsqueeze(0).float().detach().cpu().numpy(), pre_acoustic_embeds_pad, pre_token_mask])[0]       
    decoder_out = decoder_out[:, :pre_token_length.item()]
    preds = post_decode(decoder_out, pre_token_length)
    print(''.join(preds[0]))
