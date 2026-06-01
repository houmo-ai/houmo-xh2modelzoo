import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re
import pickle

model_path = "/data02/users/cc_work/model312/BERT"
label_encoder_path = "/data02/users/cc_work/model312/BERT/label_encoder.pkl"
device = "cuda:0"
page_pattern = re.compile(r'cur_page:\d+', re.IGNORECASE)
max_length = 512

model = BertForSequenceClassification.from_pretrained(
    model_path,
    local_files_only=True
)

# 移动到设备并设置为评估模式
model.to(device)
model.eval()

tokenizer = BertTokenizer.from_pretrained(
    model_path,
    local_files_only=True
)

with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)


def process_text_content(content: str) -> str:
    """处理文本内容：移除页码字符串，移除所有空白字符，截取前100个字符"""
    global page_pattern
    # 移除页码字符串
    content = page_pattern.sub('', content)
    # 移除所有空白字符（换行符、空格、制表符等）
    content = re.sub(r'\s+', '', content)
    # 只截取前100个字符
    content = content[:100]
    return content

text = "受害者是几点死亡的，死亡的具体原因是谋杀"
text = process_text_content(text)

# 文本预处理
encoding = tokenizer(
    text,
    truncation=True,
    padding='max_length',
    max_length=max_length,
    return_tensors='pt'
)

# 移动到设备
encoding = {k: v.to(device) for k, v in encoding.items()}

# 预测
with torch.no_grad():
    outputs = model(**encoding)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][predicted_class].item()

# 获取所有类别的概率
all_probs = probabilities[0].cpu().numpy()
prob_dict = {}
for i, prob in enumerate(all_probs):
    label = label_encoder.inverse_transform([i])[0]
    prob_dict[label] = float(prob)

# 解码预测标签
predicted_label = label_encoder.inverse_transform([predicted_class])[0]