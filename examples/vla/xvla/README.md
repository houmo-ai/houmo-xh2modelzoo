# XVLA XH2 导出与评测

## 环境准备

导出该模型需要同时使用xhquanttool和lerobot环境。lerobot环境配置可见 https://github.com/huggingface/lerobot;

模型主体可分为三部分：vision(vit), llm(florence2), action(12层的transformers)

如果需要在服务器上进行评测渲染的话，需要安装对应依赖：
```
conda install -c conda-forge mesa mesalib
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
ls $CONDA_PREFIX/lib | grep OSMesa
```
python要求3.12以上

## 导出HMONNX

### 导出 vision 部分hmonnx

替换相应的模型路径即可。

```bash
python examples/vla/xvla/xvla_export_vision_xh2a_libero.py --model_path models--lerobot--xvla
```
在work_dirs下会生成vision对应的onnx与hmonnx文件，以hmonnx为准

### 导出 llm(florence2) 部分hmonnx

```bash
python examples/vla/xvla/xvla_export_llm_xh2a_libero.py --config examples/vla/xvla/xvla_florence_09b_llm_xh2a_2k_libero_onnx.py
```
在配置中确认`input_sequence_length`是100，这和官方模型的配置对应，官方模型中max pos embed的长度为50，concat image和text后正好为100，text不足的部分会在预处理时pad补足

### 导出 action 部分hmonnx

先从artifactory上下载onnx
```
wget http://10.10.1.53:8082/artifactory/model_zoo2/xvla/soft_prompted_transformer.onnx
```

接着，导出hmonnx
```bash
python examples/vla/xvla/xvla_export_soft_transformer_xh2a_libero.py --onnx_path your onnx path
```

## Eval && Demo

与pi05类似，可以找@志轩

当前测试指标，在action设置为w8a16时，其余部分保持w8a8，可与fp16一致，在10个捡瓶子任务中，均成功（100%）
