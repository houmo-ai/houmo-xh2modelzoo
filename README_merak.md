# xhmodelmerak

## M50采用天璇架构(merak),适用于XH2a/YueHui系列芯片

## 安装

```bash
pip install -e . 
或者 poetry install --only-root
或者 uv pip install -e . --link-mode=copy
```

## ModelZoo

### 1. [设计文档](https://houmo.feishu.cn/wiki/J5Vrw7DCHiJzhekbEJvcwquQnzf)

## xhquant_llm 迁移指南

### 1. 配置

xhquant_llm中的配置：

```text
    model = dict(
        type="XHQwen3Model",
        hf_model=hf_model_dir,
        wrap_cfg=dict(
            max_sequence_length=2048,
            input_sequence_length=256,
            use_cache=True,
            num_logits_to_keep=1,
            kv_cache=dict(
                cache_axis=2,
            ),
            enable_rope=True,
        ),  # wrap模型时，需要传入的配置参数
        quant_config=quant_config,
        frontend_type=frontend_type,
        export_cfg=dict(
            input_names=[
                "inputs_embeds",
                "past_seq_length",
                "current_input_length",
            ],
            output_names=["logits"],
        ),
    )
```

xhmodel_merak中的配置

```
model = dict(
    model_type="Qwen3ForCausalLM",
    model_name="Qwen3-1.7B_w4a8",
    context_max_length=2048,
    prefill_chunk_length=256,
    use_cache=True,
    num_logits_to_keep=1,
    quant_scheme=dict(
        quant_type="w8a8h1_sefp",  # Node默认量化类型
        nodes=dict(
            lm_head=dict(
                quant_type="w8a8h1_sefp",
            )
        ),
        ops={},
    ),
    # 内部调试参数
    only_first_block=False,  # 仅包裹第一层，调试用
)
```

llm 模型配置文件命名规则

```text
cfg_name = f"{target_device}_{model_name}_{quant_type}_{prefill_chunk_length}_{context_length // 1024}k
例如 xh2a_qwen3_30b_a3b_w8a8h1_sefp_256_2k.py
```

### 2. 量化设置

```text
两种方式
1. quant_scheme=dict(
        quant_type="w8a8h1_sefp",  # Node默认量化类型
        nodes=dict(
            lm_head=dict(
                quant_type="w8a8h1_sefp",
            )
        ),
        ops_cfg={},
    ),
2.  quant_scheme=dict(
        # quant_type="w4a8_ssfp",  # Node默认量化类型
        w_scheme=dict(
            bits=4,
            fp_mode="ssfp",
        ),
        act_scheme=dict(
            bits=8,
            fp_mode="sefp",
        ),
        nodes=dict(
            lm_head=dict(
                # quant_type="w8a8h1_sefp",
                w_scheme=dict(
                    bits=8,
                    fp_mode="sefp",
                ),
                act_scheme=dict(
                    bits=8,
                    fp_mode="sefp",
                ),
            )
        ),
        ops={},
    )
```

### 3. transformers库模型改写

#### graph_forward

量化需要的forward接口如果和原生的forward不同，则增加graph_forward，graph_forward在做fx trace时调用,例如

```text
class _Qwen3ForCausalLM(DynamicModule):
    def graph_forward(
        self,
        # position_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        past_seq_length: Tensor | None = None,
        current_input_length: Tensor | None = None,
        past_key_cache: list[Tensor] | None = None,
        past_value_cache: list[Tensor] | None = None,
    ):
```
