from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/data/bljj_houmo/houmo-examples-xh2/models/llm/qwen3/qwen3-8b_8k_W5A16_gguf"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input


system_prompt = """你是一个笔录精简专家，给你一段“庭审对话”，请忽略上下文信息，仅基于所给的“庭审对话内容”做规整、精简，具体要求如下：
--“庭审对话”开头和结尾的内容禁止删除！！且“庭审对话”中的审判人和原告被告之间的问答信息需要全部保留！
--规整结果需要完全来源于对话本身，不要去推测对话本身是否有内容缺失，不要进行任何的内容扩写;
--不要篡改对话中的任何信息，不要删减、篡改对话中提到的金额和日期；
--对话中以'<被代>：'、'<原代>：、'<审>：、'<书记员>：'等开头的表示说话人，规整结果不能改变说话人，不能删除说话人所包含的“&nbsp”等符号；
--禁止将原对话中A说的话规整为B说的话，例如“庭审对话”为<A>：xxx...，那么这句话的规整结果不可以是<A>:xxxxx...,<B>:xxxxx...；
--禁止对原对话的内容自行做总结推理，任何时候都禁止将A说的话修改为B说的话",
--如果你无法理解或精简所给的“庭审对话”，请直接输出原对话内容",
下面是“庭审对话”内容：""" 

user_prompt1 = """<审判长>：那是驳回回避申请，不是起诉。 
原代：不是我说的，就是驳回其就书面的驳回申请，他没给我书面的。对，没给我回避的？驳回申请。我现在继续要求给我出示驳回申请书，因为你口头说他其中的内容并不清楚，我现在坚决要回避，到现在还要回避，现在主要的问题，我还要说传票，我对传票有极大的异议，问题太严重了。"""

user_prompt2 = """<审判长>：给你记录给你，你调好该上哪怀疑上，哪怀疑，该上哪举报上哪举报去。庭审继续。你的诉讼请求和事实理由，你说不是之前有没有变化。
<原代>：我现在对你进行回避的理由非常充实，法院。
<审判长>：你充实也没用了。
<原代>：受理我不认可。
<审判长>：经决定已经决定不予准许了，没用。"""

user_prompt3 = """<审判长>：本案由武清区人民法院担任审判长、与审判员、人民陪审员共同组成合议庭，书记员担任法庭记录。如果你们认为法庭的组成人员与本案有直接和间接的利害关系，可以提出事实理由申请回避。原告申请吗？
<原代>：我问一下，陪审员，和上次不是一人。从早上来作为人大。
<审判长>：全部已经都一样。
<原代>：二写的人民陪审员跟上我这边还是一样的。是吗？和上次一个人。说上诉人是一个人吗？这就这个名字。上次。
<审判长>：跟上一庭不一定一样。
<原代>：不是上一庭也是这个人人名。
<审判长>：不对。
<原代>：我没有异议。
<审判长>：被告。
<被告>：没有。"""

user_prompt4 = """<审判长>：第5个是情况说明，说明6月10日，还有6月11日、6月20号，接北京*所接派出所，对真实性、合法性、关联性、证明目的认可吗？这情况说明两份的。
<原代>：他说情况说明对吧。
<审判长>：两份，6月23号的对这两份是什么意见。
<原代>：这个上面写的很清楚，说接到通报后均已通报，属地进京将该人接盘稳控。北京市公安局分局的章上边，这个情况说明写的是后该人被天津市公安局**分局接回，北京警察并没有说是我违法乱纪，让他们把我拘留，只是说让他们接回。
<审判长>：对这个有异议，真实性有异议吗。
<原代>：真实性没异议。
<审判长>：对真实性没有异议，对他的合法性、关联性呢。
<原代>：我刚才不说了吗。
<审判长>：证明目的都不认可是吧。"""

user_prompt5 = """<审判长>：问一下原告，在被告向你出示的处罚告知笔录里边，你手写的是说你去北京社区旅游，但是当庭你陈述你这次去北京是去反映被告方的违法乱纪问题，是反映问题去了，两次陈述不一致，你解释一下是什么原因。
<原代>：这个不重要我认为。
<审判长>：主要就是你得向法庭说明一下，我们得想了解这个情况。
<原代>：他们写我并没有签字，这个不重要。
<审判长>：是你自己写的？你看一下处罚告知笔录里是你自己手写的，说你是去旅游的。
<原代>：对。
<审判长>：我去中山公园旅游，我不清楚的大门在哪，这是你在告知笔录里是这么说的吗。
<原代>：这个并不重要。
<审判长>：你解释一下，刚才你说是去那反映这几个人的违法乱纪的事，然后你在笔录里边又说是去旅游，这两个是怎么回事？你到底是去旅游，还是说反映违法乱纪的事？
<原代>：这个都不重要，这是我反映违法乱纪的事，我旅游这个不重要，还是那句话，我有违法犯罪，北京警察会处理我的。"""

user_prompt6 = """<审判长>：不是，首先你认为王 * 忠嘴这块没有伤。
<原代>：不是，他当场跟我打架前，在一公安局出前，没有伤。你到走了以后，到救护车上是你自己磕的还是你自个儿咬的？是吧？那么情况下，你这个认定不认定不了这个伤是是得就是说外力也引起，是吧？公安机关他有现场的视频，如果他有现场视频来说，现场有伤，我要是按照治安处罚法来说，殴打他人致他人轻微伤的就 5 日拘留了。"""


user_prompt_list = [user_prompt1, user_prompt2, user_prompt3, user_prompt4, user_prompt5, user_prompt6]

for user_prompt in user_prompt_list:
    messages = [{"role": "system", "content": system_prompt},{"role": "user", "content": user_prompt}]


    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("thinking content:", thinking_content)
    print("content:", content)
