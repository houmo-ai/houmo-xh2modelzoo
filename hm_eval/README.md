# HM-Eval 统一评测平台

基于 evalscope 的 xh2modelzoo 统一评测平台，支持 float 和 hmonnx 两类后端，提供 Gradio Web UI 与 CLI 两种运行方式。

HM-Eval 既可以作为个人评测平台使用，也可以作为团队共享评测平台使用：

- 个人自用：只监听本机或通过 VS Code 端口转发访问。
- 团队协作：监听内网地址或通过反向代理发布成稳定公共入口，供同组成员共同提交任务、查看日志与报告。
- 临时外部分享：通过 `--share` 生成一次性的 Gradio 公网链接，适合短期演示或与外部合作方临时联调。

> 重要说明：当前版本没有登录鉴权、用户隔离和权限控制。任何能打开页面的人，都可以看到共享任务、日志、报告，也可以提交新的评测任务。若要作为公共平台使用，建议至少放在受信内网、VPN、SSH 隧道或带鉴权的反向代理后面。

---

## 环境准备

```bash
conda activate xhquant_22
cd /data01/home/chenzx/project/xh2modelzoo
```

推荐环境：Python 3.12、torch 2.8、transformers 5.5、evalscope 1.4.2、gradio 6.12。

---

## 快速开始

### 方案一：作为个人评测平台使用

如果只是自己在服务器上使用，建议监听本机地址：

```bash
conda activate xhquant_22
cd /data01/home/chenzx/project/xh2modelzoo
python -m hm_eval --host 127.0.0.1 --port 7860
```

正确访问地址取决于你是在哪台机器上打开浏览器：

- 如果浏览器就在运行 HM-Eval 的服务器本机：`http://127.0.0.1:7860`
- 如果你通过 VS Code Remote-SSH 在自己电脑上访问：先转发 `7860` 端口，再打开 `http://localhost:7860`
- 如果你通过 SSH 隧道访问：建立 `7860:127.0.0.1:7860` 隧道后，打开 `http://127.0.0.1:7860` 或 `http://localhost:7860`
- 不能从另一台机器直接打开 `http://<服务器IP>:7860`，因为当前进程只监听在 `127.0.0.1`

适用场景：

- 本机直接打开 `http://127.0.0.1:7860`
- 通过 VS Code Remote-SSH 连接服务器后，使用 VS Code 端口转发访问 `http://localhost:7860`
- 通过 SSH 隧道只给自己访问

### 方案二：作为团队共享平台使用

如果希望同一内网中的其他同事也能访问，监听 `0.0.0.0`：

```bash
conda activate xhquant_22
cd /data01/home/chenzx/project/xh2modelzoo
python -m hm_eval --host 0.0.0.0 --port 7860
```

访问方式：

- 服务器本机：`http://127.0.0.1:7860`
- 同一内网其他机器：`http://<服务器IP>:7860`
- 配合反向代理域名：`https://hm-eval.example.com`

### 方案三：临时开放成公网链接

如果你只是想快速给外部合作方一个临时可访问的公共链接，可以启用 Gradio share：

```bash
conda activate xhquant_22
cd /data01/home/chenzx/project/xh2modelzoo
python -m hm_eval --host 0.0.0.0 --port 7860 --share
```

启动后终端会打印类似下面的地址：

```text
https://xxxxxx.gradio.live
```

适用场景：

- 临时演示
- 快速外部联调
- 不方便立刻开防火墙或配反向代理时的短期共享

限制与建议：

- `--share` 生成的是临时隧道链接，不适合长期稳定运行。
- 链接生命周期、带宽与可达性依赖 Gradio share 服务。
- 长时间评测、大模型任务和团队长期协作，仍建议用固定端口 + 反向代理的方式部署。

---

## 启动参数

HM-Eval 的实际启动参数来自 `python -m hm_eval`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | Web UI 监听地址。个人自用建议改成 `127.0.0.1`，共享使用再改成 `0.0.0.0`。 |
| `--port` | `7860` | Web UI 端口。可改为 `7861`、`7862`、`8080` 等，适合多实例并行部署。 |
| `--share` | 关闭 | 启用 Gradio share，生成临时公网访问链接。 |
| `--debug` | 关闭 | 输出更详细的调试日志。 |
| `--cli` | 关闭 | 切换到 CLI 模式，不启动 Gradio。 |
| `--model` | 无 | CLI 模式下指定模型 display name 或 config_id。 |
| `--backend` | `float` | CLI 模式下指定后端，可选 `float` 或 `hmonnx`。 |
| `--datasets` | 无 | CLI 模式下指定一个或多个数据集。 |
| `--limit` | `0` | CLI 模式下每个子集的题数限制，`0` 表示全量。 |
| `--max-tokens` | `512` | CLI 模式下最大生成 tokens；float 与 hmonnx 均按 `max_new_tokens` 理解，不包含 prompt/few-shot 输入长度。 |

---

## 访问方式与适用场景

| 场景 | 推荐启动方式 | 典型访问地址 | 说明 |
|------|--------------|--------------|------|
| 本机个人使用 | `--host 127.0.0.1 --port 7860` | `http://127.0.0.1:7860` | 最安全，只有本机可访问。 |
| VS Code Remote-SSH | `--host 127.0.0.1 --port 7860` | `http://localhost:7860` | 通过 VS Code 端口转发访问，推荐个人日常使用。 |
| 团队内网共享 | `--host 0.0.0.0 --port 7860` | `http://<服务器IP>:7860` | 适合同一办公网段或实验室环境。 |
| 临时公网分享 | `--host 0.0.0.0 --port 7860 --share` | `https://xxxx.gradio.live` | 适合临时给外部合作方访问。 |
| 稳定公网服务 | `--host 127.0.0.1 --port 7860` + 反向代理 | `https://hm-eval.example.com` | 最适合长期协作，便于做 HTTPS、鉴权和访问控制。 |

---

## 作为个人评测平台使用

### VS Code 端口转发

如果你是通过 VS Code Remote-SSH 连接服务器，推荐使用端口转发而不是直接开放端口：

1. 启动服务：

```bash
python -m hm_eval --host 127.0.0.1 --port 7860
```

注意：这条命令只会让服务监听在服务器本机的 `127.0.0.1:7860`。
如果你是在自己的电脑浏览器里访问，就不能直接打开 `http://<服务器IP>:7860`，而应当通过端口转发后访问 `http://localhost:7860`。

2. 打开 VS Code 底部面板的 Ports 标签。
3. 点击 Forward a Port，输入 `7860`。
4. 在本地浏览器打开 `http://localhost:7860`。

这样 HM-Eval 只对你自己可见，不会直接暴露给整台服务器所在网络。

### SSH 隧道

也可以手动做 SSH 隧道：

```bash
ssh -L 7860:127.0.0.1:7860 <user>@<server>
```

然后本地浏览器访问 `http://127.0.0.1:7860`。

---

## 作为公共协作平台使用

### 最小部署方式：直接开放端口

如果你的团队就在同一内网，可以直接开放固定端口：

```bash
conda activate xhquant_22
cd /data01/home/chenzx/project/xh2modelzoo
nohup python -m hm_eval --host 0.0.0.0 --port 7860 > hm_eval/logs/server_7860.log 2>&1 &
echo $!
```

然后确认端口已监听：

```bash
ss -lntp | grep 7860
curl -I http://127.0.0.1:7860
```

如果系统有防火墙或云安全组，还需要放行对应端口。

### 推荐部署方式：反向代理成固定公共入口

更推荐的做法是让 HM-Eval 只监听本机，再由 Nginx、Caddy 或公司已有网关统一暴露给外部：

```bash
python -m hm_eval --host 127.0.0.1 --port 7860
```

Nginx 示例：

```nginx
server {
    listen 80;
    server_name hm-eval.example.com;

    location / {
        proxy_pass http://127.0.0.1:7860;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

这样做的好处：

- 可以挂自己的域名。
- 更容易接 HTTPS。
- 更容易在代理层加白名单、Basic Auth、SSO 或 VPN。
- 端口不必直接暴露在公网。

### 多人协作时的建议

当前 HM-Eval 是共享实例模型：

- 所有人共享同一个任务列表。
- 所有人都能查看共享日志和报告。
- 所有人共用同一台机器上的 GPU 与导出产物。

因此更适合以下两种方式：

1. 同一团队共用一个实例，统一走一个固定域名或端口。
2. 不同团队或不同用途，各自起一个实例，分别占用不同端口，例如 `7860`、`7861`、`7862`。

如果你需要真正的多租户隔离，至少要在 HM-Eval 前面增加鉴权层，或者直接按团队拆成多个独立实例。

---

## 公共部署前必须确认的事项

在把 HM-Eval 当公共平台发布之前，建议逐项确认：

- 模型权重目录对运行用户可读。
- `work_dirs/` 下的 `export_meta_info.json` 及 ONNX 产物对运行用户可读。
- `hm_eval/tasks/`、`hm_eval/outputs/`、`hm_eval/logs/` 具备写权限。
- 服务器 GPU 足够，或团队成员知道如何在页面里手动选择 GPU。
- 端口已经在系统防火墙和安全组中放行。
- 如果平台需要给受限用户访问，已经在外层配置了 VPN、SSH 隧道、Basic Auth、公司网关或其他鉴权机制。

---

## 无法访问时的排查方法

```bash
# 1. 看服务是否真的监听在目标端口
ss -lntp | grep 7860

# 2. 本机连通性测试
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:7860/

# 3. 如果走内网 IP 访问失败，检查防火墙或安全组
iptables -L -n | grep 7860

# 4. 如果只是临时给外部访问，可直接试 share 模式
python -m hm_eval --host 0.0.0.0 --port 7860 --share
```

常见问题：

- 本机能访问，别人不能访问：通常是防火墙、安全组或只绑定了 `127.0.0.1`。
- `--share` 没拿到链接：通常是外网出站受限，或 Gradio share 服务不可达。
- 页面能打开但任务失败：优先检查 GPU、transformers 版本切换、模型路径和导出产物路径。

---

## 后台启动与关闭

### 后台启动

```bash
conda activate xhquant_22
cd /data01/home/chenzx/project/xh2modelzoo
nohup python -m hm_eval --host 0.0.0.0 --port 7860 > hm_eval/logs/server_7860.log 2>&1 &
echo $!
```

### 前台关闭

前台终端直接按 `Ctrl+C`。

### 后台关闭

```bash
ps aux | grep "python -m hm_eval" | grep -v grep
kill <PID>

# 或者按端口关闭
fuser -k 7860/tcp

# 或者一键关闭所有 hm_eval Web 进程
pkill -f "python -m hm_eval"
```

---

## Web UI 功能说明

平台包含两个页面标签：

### 1. 提交评测任务

- 选择模型：从 `hm_eval/model_configs/` 自动发现。
- 选择后端：`float` 或 `hmonnx`。
- 选择数据集：模型推荐数据集自动勾选，支持关键词搜索追加。
- 设置参数：包括每子集题数、最大生成 tokens 和 GPU 选择；最大生成 tokens 在 float 与 hmonnx 下均表示 `max_new_tokens`，不包含 prompt/few-shot 输入长度。
- hmonnx 配置：不再自动扫描任何导出结果，LLM 部分需要填写运行时 meta 的完整绝对路径，支持 `golden_meta_info.json`、`export_meta_info.json` 以及兼容旧导出的 `meta.json`；CMMMU/MMMU 等多模态任务在旧式分离导出物下还需要额外填写 vision 部分的 `export_meta_info.json`。
- 点击提交后，任务会进入后台子进程执行。

### 2. 评测报告

- 集中展示所有任务状态，包括进行中、失败和已完成。
- 点击任务后可直接查看实时日志、失败诊断和完成报告。
- 已完成任务展示 HTML 报告，并支持复制纯文本报告结果。
- 多模态评测的错误样例会在 HTML 报告中展示对应图片，便于定性分析。

---

## HMONNX 的当前使用方式

README 这里特别说明当前实现的真实行为：

- 当前 Web UI 不会在页面里自动执行模型导出。
- 选择 `hmonnx` 后端时，必须先准备好当前仓库导出的运行时 meta，支持 `golden_meta_info.json`、`export_meta_info.json` 或兼容旧导出的 `meta.json`。
- 如果手里是导出入口生成的 `export_meta_info.json`，只要其中包含 `exported_dir` 且能定位到对应的 `golden_meta_info.json`，平台也可以直接使用。
- 页面中的 “最大生成 tokens” 对 hmonnx 也表示生成 token 上限，不表示 prompt+生成的总长度。旧版 `LLMWithMaskONNXModel` 导出物如果遇到 few-shot prompt 超出导出上下文窗口，会自动截断左侧 prompt，为这部分生成预算保留 KV cache 空间。
- 对 CMMMU/MMMU 等多模态任务，旧式分离导出物除 LLM meta 外，还要在页面的 “Vision HMONNX export_meta_info.json 路径” 中填写视觉编码器导出的 `export_meta_info.json`，例如 `work_dirs/gemma4_moe_26b_a4b_it_vision_xh2a_no_upsample_token_448x448/export_meta_info.json`。如果 unified `golden_meta_info.json` 或兼容 `meta.json` 已经内嵌 `visual_config`，只填写 LLM meta 即可。
- 平台不会扫描 `work_dirs/` 或其他目录，必须手动填写 meta 文件的完整绝对路径。
- 如果没有有效的 meta 文件，hmonnx 任务不会提交成功。

### 手动导出示例

```bash
conda activate xhquant_22
cd /data01/home/chenzx/project/xh2modelzoo

python examples_merak/llm/gemma4_moe/gemma4_moe_with_mask_xh_export_hmonnx.py \
  --config configs_merak/xh2a/llm_models/gemma4_moe/26b_a4b_it/gemma4_moe_with_mask_26b_a4b_it_xh2a_w4a8_256_2k.py \
    --valid
```

导出完成后，通常会在 `work_dirs/<配置名>/` 下生成 `export_meta_info.json`，并在导出子目录内生成 `golden_meta_info.json` 和对应 HMONNX 产物。部分旧式 qwen3_next 导出目录则直接提供 `meta.json`。回到 HM-Eval 页面后，填写可直接运行的 meta 路径即可；如果 `export_meta_info.json` 能通过 `exported_dir` 指向对应 `golden_meta_info.json`，也可以直接填写 `export_meta_info.json`。

旧式多模态 HMONNX 还需要提前准备视觉部分导出产物。Gemma4 / Qwen3.5 这类已内嵌 vision 子图的 unified meta 可以只填 LLM meta；如果是旧式分离导出物，提交 CMMMU 任务时同时填写：

- LLM meta：语言模型导出的 `golden_meta_info.json`、兼容旧导出的 `meta.json`，或可定位到它的 `export_meta_info.json`。
- Vision meta：视觉部分导出的 `export_meta_info.json`。

如果 unified `golden_meta_info.json` 或兼容 `meta.json` 的 `model_config` 已包含 `visual_config`，平台会优先从 LLM meta 中加载 vision 运行信息，Vision meta 可以留空。

float 后端不需要填写 vision meta，平台会直接通过官方 HF 模型与 `AutoProcessor` 处理 evalscope 的图文消息。

---

## CLI 模式

如果不需要 Web UI，也可以直接走 CLI：

```bash
# 快速测试（每子集 5 题）
python -m hm_eval --cli \
    --model "Gemma-4-26B-A4B-it" \
    --backend float \
    --datasets mmlu_pro gsm8k \
    --limit 5

# 全量评测
python -m hm_eval --cli \
    --model "Qwen3.5-35B-A3B" \
    --backend float \
    --datasets ceval cmmlu mmlu_pro gsm8k math_500
```

CLI 模式适合：

- 单次调试
- 批处理脚本
- 不需要共享页面时的本地评测

---

## 目录结构

```text
hm_eval/
├── __init__.py           # 版本信息
├── __main__.py           # 入口（Gradio / CLI）
├── app.py                # Gradio Web UI
├── worker.py             # 后台任务执行进程
├── core/
│   ├── normalize.py      # 提示词辅助与可选答案提取工具
│   ├── model_registry.py # YAML 模型配置加载与注册
│   ├── dataset_registry.py # 数据集注册、搜索和 max_tokens hint
│   ├── env_manager.py    # transformers 版本切换、GPU 管理
│   ├── backends.py       # float / hmonnx 后端抽象
│   ├── eval_runner.py    # evalscope 评测执行引擎
│   ├── report.py         # JSON / 文本 / HTML 报告生成
│   └── task_manager.py   # 后台任务管理与持久化
├── model_configs/        # 模型 YAML 配置文件
├── tasks/                # 任务持久化（自动创建）
├── outputs/              # 评测输出目录（自动创建）
└── logs/                 # Web 服务日志目录
```

---

## 添加新模型

在 `hm_eval/model_configs/` 下创建 YAML 文件，例如：

```yaml
config_id: my_model
display_name: My-Model-7B
model_family: qwen3
architecture: Qwen3ForCausalLM
hf_model_dir: /path/to/weights
transformers_version: "5.5.0"

recommended_datasets:
  - cmmmu
  - mmlu_pro
  - gsm8k

backends:
  float:
    type: float
    model_class: Qwen3ForCausalLM
    extra_args:
      torch_dtype: bfloat16
      device_map: auto

  hmonnx:
    type: hmonnx
    onnx_model_type: qwen3_next
    xhquant_config: configs/xxx/config.py
    vision_export_meta_info: /path/to/vision/export_meta_info.json  # 仅多模态模型需要
```

---

## 支持的数据集

| 数据集 | 类型 | 说明 |
|--------|------|------|
| ceval | 选择题 | 中文学科综合 |
| cmmlu | 选择题 | 中文多任务理解 |
| mmlu_pro | 选择题 | MMLU 增强版（A-J） |
| mmlu | 选择题 | 英文多任务理解 |
| arc | 选择题 | 科学推理 |
| hellaswag | 选择题 | 常识推理 |
| winogrande | 选择题 | 代词消解 |
| gsm8k | 数值 | 小学数学 |
| math_500 | 数值 | 高等数学 |
| humaneval | 代码 | Python 代码生成 |
| ifeval | 生成 | 指令遵循 |
| gpqa | 选择题 | 研究生级别 QA |
| bbh | 混合 | Big-Bench Hard |
| truthfulqa | 选择题 | 真实性评估 |

## 功能特性

- **统一后端抽象**: float (HF model.generate) 和 hmonnx (xhmodel_merak AutoLLMHONNXModel，兼容旧版 LLMWithMaskONNXModel / Qwen3NextONNXModel meta)
- **保留模型输出**: 交给 evalscope 的 predictions/reviews 默认保存模型生成文本，便于后续调整后处理或评测策略后复评，不必重新生成
- **per-subset 详细报告**: 每个子集的准确率、正确数、总数
- **数据集搜索**: 关键词搜索内置+evalscope注册的数据集
- **后台任务管理**: 进程隔离执行，状态持久化，日志流式查看
- **transformers 版本管理**: 自动切换 transformers 版本 (Gemma4 / Qwen3.5 / Qwen3.6 均按模型配置切到 5.5.0)
