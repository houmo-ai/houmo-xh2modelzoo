from xhquant.api import (  # type: ignore # isort:skip
    Config,
    DeviceType,
    ConfigDict,
    convert_fx_model_to_quanted_model,
    convert_quanted_model_to_hmonnx,
    get_root_logger,
    create_quant_config,
    is_ssfp_quant_config,
    HMONNXGoldenInference,
)
from modelscope import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F

hf_model_dir = '/data02/datasets/paddle_gte'
tokenizer = AutoTokenizer.from_pretrained(hf_model_dir)
model = AutoModel.from_pretrained(hf_model_dir, trust_remote_code=True)
model.cuda()
device = "cuda:0"

word_embedding = model.embeddings.word_embeddings
token_type_embedding = model.embeddings.token_type_embeddings

prefill_onnx_file = "work_dirs/paddle_gte-XH2a-batch_1-2k-w4a8h1_sefp/hmonnx/prefill/paddle_gte-XH2a-2k-w4a8h1_sefp_prefill.onnx"

session = HMONNXGoldenInference(prefill_onnx_file)
session.to(device)
session.save_golden = False

input_texts = [
    "what is the capital of China?",
    "how to implement quick sort in python?",
    "北京",
    "快排算法介绍"
]
b_outputs = []
target_length = 256

for text in input_texts:
    data_dict = tokenizer(text, max_length=8192, padding=True, truncation=True, return_tensors='pt')
    input_ids_t = data_dict['input_ids'].cuda()
    current_length = input_ids_t.shape[1] 
    pad_amount = (0, target_length - current_length)  

    input_ids_padded = F.pad(input_ids_t, pad_amount, mode='constant', value=1)

    attention_mask = torch.zeros_like( data_dict['attention_mask'] ).cuda()
    attention_mask_padded = torch.ones( (1, target_length - current_length), device=device)* -torch.inf
    attention_mask_padded = torch.concat([attention_mask, attention_mask_padded], dim=1).unsqueeze(0).unsqueeze(0)

    word_embeds = word_embedding(input_ids_padded) # [4, 11, 768]
    
    token_type_ids = torch.zeros_like(input_ids_padded)
    token_embeds = token_type_embedding(token_type_ids) # [4, 11, 768]
    # attention_mask = torch.zeros( (1,1,1,input_ids_t.shape[-1]), device=device)
    position_ids = torch.arange(256, device=device).unsqueeze(0)

    inputs_list = (word_embeds.half(), token_embeds.half(), attention_mask_padded.half(), position_ids.to(torch.int32))

    outputs = session(*inputs_list)
    outputs = outputs[:, 0][:768]
    b_outputs.append(outputs)

out_data  = torch.concat(b_outputs, dim=0)
embeddings = F.normalize(out_data, p=2, dim=1)
scores = (embeddings[:1] @ embeddings[1:].T)
print(scores.tolist())   