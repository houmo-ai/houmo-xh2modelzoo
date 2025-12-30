 # 1. quant
 # 2. mix_search
import dataclasses
import attrs


mix_search = True
quant_config  = dict(
    mix_search = mix_search,
)


@dataclasses.dataclass
class Qwen3LegacyQuantConfig:
    model_name: str
    context_length: int
    input_sequence_length: int
    quant_type: str
    quant_weight: str
    mix_search: str
    #****************# 
    # 别管
    #****************# 


# 首先根据config如果是混合精度需要生成混合精度表，生成完成以后送入到 GPTQModel 之后

# quant_model 本身支持混合精度，但是不方便导出给 GPTQModel.  
    # 如果要实现混合精度
    # method 1
    # nn.Linear + 上一个字符串. 表示它的位置.  然后生成配置表
    # 这个表送给 GPTQModel.
    # method 2
    # 不借助quant_model, 用其他方式，比方说海森启发的方式
    # 同样GPTQModel需要支持混合精度的配置 
    # 3. GPTQModel的混合精度配置
        # 支持正则表达式的配置
    
    # XH2Model Zoo 需要做什么？
        # 1. mix_search.py 需要能接入混合精度，并输出表
        # 2. 如果是加载 GPTQModel 的混合精度模型，不读取config, 而是读取 GPTQModel config 的 bit 信息
    
    # APIs 
        # 1. mix_search 输出配置表
        # 2. 量化的 API, 根据类型走不同的量化方法
    
# export_honnx

# golden, 再说
    # 基于 hmonnx 进行 demo, 这个针对 LLMs 写一个 API?    (写一个文件更方便)

