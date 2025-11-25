from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import sys
import os
import random


prefix_name = sys.argv[1]
model_path = sys.argv[2]
gen_seqlen=int(sys.argv[3])

prefix = os.path.join('gen_data/',prefix_name,str(gen_seqlen))
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path,device_map='auto')

n_vocab = random.sample(range(model.config.vocab_size), 512) 

if not os.path.exists("gen_data"):
    os.mkdir("gen_data")

for i in n_vocab:
    input_ids = torch.tensor([[i]]).cuda()
    print("generating: ", i)
    random_len = random.randint(3, 6)
    outputs1 = model.generate(input_ids, do_sample=False, max_length=random_len)
    outputs = model.generate(outputs1, do_sample=False, num_beams=4, max_length=gen_seqlen,repetition_penalty=1.5,length_penalty=1)
    gen_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(gen_text)
    text_dict = {"text" : gen_text[0]}
    with open(prefix+".jsonl", "a") as f:
        f.write(json.dumps(text_dict))
        f.write('\n')


