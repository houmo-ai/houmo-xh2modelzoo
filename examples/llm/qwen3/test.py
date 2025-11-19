# from auto_round import AutoRoundConfig ##must import for auto-round format
from modelscope import AutoModelForCausalLM,AutoTokenizer
quantized_model_dir = "/data02/datasets/Qwen2.5-1.5B-Instruct-int4-sym-inc"
tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir)

model = AutoModelForCausalLM.from_pretrained(
    quantized_model_dir,
    torch_dtype='auto',
    device_map="auto",
    ##revision="14dbc8" ## AutoGPTQ format
)

##import habana_frameworks.torch.core as htcore ## uncommnet it for HPU
##import habana_frameworks.torch.hpu as hthpu ## uncommnet it for HPU
##model = model.to(torch.bfloat16).to("hpu") ## uncommnet it for HPU

prompt = "There is a girl who likes adventure,"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=200,  ##change this to align with the official usage
    do_sample=False  ##change this to align with the official usage
)
generated_ids = [
output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)

prompt = "There is a girl who likes adventure,"
##INT4:
"""That's great! What kind of adventures do you like to go on? Do you prefer outdoor activities or indoor ones? Maybe we could come up with some ideas together!
"""

##BF16:
"""That's great! Adventure can be an exciting and fulfilling experience for many people. What kind of adventures do you like to go on? Do you enjoy hiking, camping, or exploring new places? Or maybe you prefer more extreme activities like skydiving or bungee jumping? Whatever your interests may be, there are plenty of opportunities out there for someone who loves adventure.
"""

prompt = "9.11和9.8哪个数字大"  
#INT4: 
"""
9.11 和 9.8 都是小数，它们的大小比较如下：

- 9.11 大于 9.8

具体来说：
- 9.11 的十位和个位都是 9，十分位是 1。
- 9.8 的十位和个位都是 9，十分位也是 8。

由于 1 > 8，在相同的小数部分相同时，较大的数字在十位上。因此，9.11 比 9.8 更大。
"""

##BF16: 
"""9.11 和 9.8 都是小数，比较它们的大小需要从左到右逐位进行比较。

首先看整数部分：
- 9.11 的整数部分是 9。
- 9.8 的整数部分也是 9。

因为两者的整数部分相同，所以继续比较小数部分：

- 9.11 的小数部分是 0.11。
- 9.8 的小数部分是 0.8。

现在我们来比较这两个小数点后的数字：
- 0.11 和 0.8

显然，0.11 小于 0.8。因此，9.11 比 9.8 大。

所以，答案是：9.11 > 9.8。
"""


prompt = "Once upon a time,"
##INT4: 
"""I'm sorry, but I don't understand what you're asking. Could you please provide more context or clarify your question?"""

##BF16:
"""I'm ready to listen! What's the story you'd like me to hear?"""


prompt = "请简短介绍一下阿里巴巴公司"
##INT4:
"""阿里巴巴集团是一家中国领先的电子商务和数字技术公司，成立于1999年。它最初是淘宝网的运营者，后来发展成为全球最大的电商平台之一，并且在云计算、金融科技等领域也取得了显著成就。

阿里巴巴旗下的主要业务包括：

1. **淘宝**：一个在线购物平台，提供各种商品和服务。
2. **天猫**：另一个大型电商平台，专注于销售品牌商品。
3. **阿里云**：提供云计算服务，帮助企业和个人实现数字化转型。
4. **蚂蚁金服**：提供金融服务，如支付宝支付系统、余额宝等。
5. **菜鸟网络**：负责物流配送，支持电商交易和快递服务。
6. **Lazada** 和 **Global Markets**：分别是中国和东南亚地区的电商平台。

阿里巴巴通过这些业务为消费者提供了便捷的购物体验，同时也为企业和个人提供了多种商业解决方案和技术支持。此外，阿里巴巴还积极参与社会公益事业，致力于推动数字经济的发展和社会的可持续性。。"""

##BF16:
"""阿里巴巴集团是一家中国领先的电子商务和数字技术公司，成立于1999年。它以B2C（企业对消费者）和B2B（企业对企业）为主要业务模式，提供包括淘宝、天猫、速卖通等在内的在线购物平台，并通过支付宝为用户提供支付服务。阿里巴巴还涉足云计算、人工智能等领域，致力于推动数字经济的发展。
"""