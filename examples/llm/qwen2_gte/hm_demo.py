from sentence_transformers import SentenceTransformer
from xhquant.api import (
    DeviceType,
    HMONNXGoldenInference,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    get_root_logger,
    ptq_quantize,
    to_frontend_graph,
    to_quant_graph,
    xhquant_init,
    HMONNXInference,
    CacheTensor,
)
import torch
import torch.nn.functional as F

model = SentenceTransformer("/data02/datasets/gte_qwen2_1.5b_inst", trust_remote_code=True)
# In case you want to reduce the maximum length:
model.max_seq_length = 8192

queries = [
    "how much protein should a female eat",
    "summit define",
]
documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
]

# 0 ='Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: how much protein should a female eat'
# 1 ='Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: summit define'

query_prefix = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "

queries_process = [query_prefix + query for query in queries]

# load hmonnx
device = torch.device("cuda")
prefill = HMONNXInference(str("work_dirs/gte_qwen2_1.5b_inst-XH2a/hmonnx/golden/prefill/hmquant_gte_qwen2_1.5b_inst-XH2a-batch_1-2k-w8a8h1_sefp_prefill_with_act.onnx"))
# decoder = HMONNXInference(str("work_dirs/gte_qwen2_1.5b_inst-XH2a/hmonnx/golden/decoder/hmquant_gte_qwen2_1.5b_inst-XH2a-batch_1-2k-w8a8h1_sefp_decoder_with_act.onnx"))
prefill.to(device)
# decoder.to(device)
prefill.exec_device = device
# decoder.exec_device = device
token_embedding = model[0].auto_model.embed_tokens

# prepare cache
head_dim = model[0].auto_model.config.hidden_size // model[0].auto_model.config.num_attention_heads
num_hidden_layers = model[0].auto_model.config.num_hidden_layers
num_decoder_layers = num_hidden_layers
kv_cache_shape = [
    1,
    model[0].auto_model.config.num_key_value_heads,
    2048,
    head_dim,
]        
past_key_caches = []
past_value_caches = []
for _ in range(num_decoder_layers):
    past_key_caches.append(CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)))
    past_value_caches.append(CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)))


# query ===========================
query_emb = []
for query_single in queries_process:
    inputs_ids = model.tokenize([query_single])['input_ids']

    if inputs_ids.shape[1] > 256:
        num_infer = inputs_ids.shape[1] // 256
        for i in range(num_infer):
            if i != num_infer - 1:
                inputs_ids_single = inputs_ids[:, i * 256: (i + 1) * 256]
            else:
                inputs_ids_single = inputs_ids[:, i * 256:]

            inputs_ids_single = inputs_ids_single.to(device)

            seq_length = inputs_ids_single.shape[1]
            inputs_embeds = token_embedding(inputs_ids_single)
            if inputs_ids_single.shape[1] != 256:
                inputs_embeds = F.pad(inputs_embeds, (0, 0, 0, 256 - inputs_ids_single.shape[1]))

            past_seq_length = torch.tensor([i * 256], dtype=torch.int32, device="cuda")
            current_input_length = torch.tensor([seq_length], dtype=torch.int32, device="cuda")
            position_id = torch.arange(past_seq_length, past_seq_length + seq_length, dtype=torch.long, device="cuda").unsqueeze(0)

            inputs = [
                inputs_embeds,
                past_seq_length,
                current_input_length,
                position_id,
            ]
            for k_data in past_key_caches:
                inputs.append(k_data.half())

            for v_data in past_value_caches:
                inputs.append(v_data.half())

            outputs = prefill(inputs)
    else:
        seq_length = inputs_ids.shape[1]
        # inputs_ids = F.pad(inputs_ids, (0, 256 - inputs_ids.shape[1])).to("cuda")
        inputs_embeds = token_embedding(inputs_ids.to("cuda"))
        if inputs_ids.shape[1] != 256:
            inputs_embeds = F.pad(inputs_embeds, (0, 0, 0, 256 - inputs_ids.shape[1])).to("cuda")

        past_seq_length = torch.tensor([0], dtype=torch.int32, device="cuda")
        current_input_length = torch.tensor([seq_length], dtype=torch.int32, device="cuda")
        position_id = torch.arange(past_seq_length.item(), 256, dtype=torch.int32, device="cuda").unsqueeze(0)

        inputs = [
            inputs_embeds.half(),
            past_seq_length,
            current_input_length,
            position_id,
        ]
        for k_data in past_key_caches:
            inputs.append(k_data.half())

        for v_data in past_value_caches:
            inputs.append(v_data.half())


        outputs = prefill(*inputs)
    
    query_emb.append(outputs)




# document ============================
past_key_caches_doc = []
past_value_caches_doc = []
for _ in range(num_decoder_layers):
    past_key_caches_doc.append(CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)))
    past_value_caches_doc.append(CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)))

doc_emb = []
for doc_single in documents:
    inputs_ids = model.tokenize([doc_single])['input_ids']

    if inputs_ids.shape[1] > 256:
        num_infer = inputs_ids.shape[1] // 256
        for i in range(num_infer):
            if i != num_infer - 1:
                inputs_ids_single = inputs_ids[:, i * 256: (i + 1) * 256]
            else:
                inputs_ids_single = inputs_ids[:, i * 256:]

            inputs_ids_single = inputs_ids_single.to(device)

            seq_length = inputs_ids_single.shape[1]
            inputs_embeds = token_embedding(inputs_ids_single)
            if inputs_ids_single.shape[1] != 256:
                inputs_embeds = F.pad(inputs_embeds, (0, 0, 0, 256 - inputs_ids_single.shape[1]))

            past_seq_length = torch.tensor([i * 256], dtype=torch.int32, device="cuda")
            current_input_length = torch.tensor([seq_length], dtype=torch.int32, device="cuda")
            position_id = torch.arange(past_seq_length, past_seq_length + seq_length, dtype=torch.long, device="cuda").unsqueeze(0)

            inputs = [
                inputs_embeds,
                past_seq_length,
                current_input_length,
                position_id,
            ]
            for k_data in past_key_caches_doc:
                inputs.append(k_data.half())

            for v_data in past_value_caches_doc:
                inputs.append(v_data.half())

            outputs = prefill(inputs)

    else:
        seq_length = inputs_ids.shape[1]
        # inputs_ids = F.pad(inputs_ids, (0, 256 - inputs_ids.shape[1])).to("cuda")
        inputs_embeds = token_embedding(inputs_ids.to("cuda"))
        if inputs_ids.shape[1] != 256:
            inputs_embeds = F.pad(inputs_embeds, (0, 0, 0, 256 - inputs_ids.shape[1])).to("cuda")

        past_seq_length = torch.tensor([0], dtype=torch.int32, device="cuda")
        current_input_length = torch.tensor([seq_length], dtype=torch.int32, device="cuda")
        position_id = torch.arange(past_seq_length.item(), 256, dtype=torch.int32, device="cuda").unsqueeze(0)

        inputs = [
            inputs_embeds.half(),
            past_seq_length,
            current_input_length,
            position_id,
        ]
        for k_data in past_key_caches_doc:
            inputs.append(k_data.half())

        for v_data in past_value_caches_doc:
            inputs.append(v_data.half())


        outputs = prefill(*inputs)
    
    doc_emb.append(outputs)



# query_embeddings = model.encode(queries, prompt_name="query")
# document_embeddings = model.encode(documents)

query_emb_c = torch.concat(query_emb, axis=0)
doc_emb_c = torch.concat(doc_emb, axis=0)

scores = (query_emb_c @ doc_emb_c.T) * 100  
# tensor([[71.8125, 22.0312],
#         [12.2969, 62.9375]], device='cuda:0', dtype=torch.float16)

# [[78.49691772460938, 17.04286766052246], [14.924494743347168, 75.37962341308594]]
print(scores.tolist())
