# XH Model Zoo

## 开发手册

[这里](https://houmo.feishu.cn/wiki/GjWNwICABiBm8Ykn0GAcRKnen6d)

## 依赖项

使用xhquanttool工程的python环境

## 安装

```bash
pip install -v -e . --no-index --no-build-isolation
或者 uv pip install -e . --link-mode=copy
```

## HM-Eval 评测平台使用方式

本仓库内置 `hm_eval` 统一评测平台，用于通过 evalscope 对 LLM 模型执行 float 或 hmonnx 后端评测。完整说明见 [hm_eval/README.md](hm_eval/README.md)。

### 启动 Web UI

```bash
cd /data01/home/chenzx/project/xh2modelzoo
python -m hm_eval --host 127.0.0.1 --port 7860
```

通过 VS Code Remote-SSH 使用时，推荐转发 `7860` 端口后在本地浏览器打开 `http://localhost:7860`。如果要作为内网共享平台使用，可以改为：

```bash
python -m hm_eval --host 0.0.0.0 --port 7860
```

注意：当前平台没有登录鉴权。监听 `0.0.0.0` 或通过反向代理发布前，请确认只暴露在受信网络、VPN、SSH 隧道或带鉴权的网关后面。

### CLI 评测

```bash
python -m hm_eval --cli \
  --model "Gemma-4-26B-A4B-it" \
  --backend float \
  --datasets mmlu_pro gsm8k \
  --limit 5
```

### HMONNX 评测

HMONNX 后端不会在页面内自动导出模型。请先使用 `examples_merak/llm/` 下的导出脚本生成 `golden_meta_info.json`，再在 Web UI 中填写该文件的完整绝对路径。若导出生成的 `export_meta_info.json` 能通过 `exported_dir` 定位到 `golden_meta_info.json`，也可以直接填写 `export_meta_info.json`。

页面和 CLI 中的最大生成 tokens 在 float 与 hmonnx 下均表示 `max_new_tokens`，不包含 prompt/few-shot 输入长度。旧版 HMONNX 导出物如果遇到 few-shot prompt 超出导出上下文窗口，会自动截断左侧 prompt，为生成预算保留 KV cache 空间。

多模态评测（如 CMMMU/MMMU）在 hmonnx 后端下还需要额外填写 vision 部分导出的 `export_meta_info.json`。float 后端会直接加载官方模型与 `AutoProcessor` 处理图文输入；完成后的 HTML 报告会展示错误样例对应图片，便于定性分析。

## 代码提交

```拉取代码
git pull --rebase
```

```bash
在develop分支上提交代码，提交前请确保代码已经通过测试。
git push origin HEAD:refs/for/develop
```

## 量化格式

需要量化的算子：Conv、Linear、MatMul、Gemm

| 量化模式   | 权重位宽 | 激活位宽 | 计算模式 |
| ---------- | -------- | -------- | -------- |
| w8a8-sefp  | 8bit     | 8bit     | sefp     |
| w8a16-sefp | 8bit     | 16bit    | sefp     |
| w4a8-ssfp  | 4bit     | 8bit     | ssfp     |

## Model zoo

<table align="left">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>LLM</b>
      </td>
      <td>
        <b>Multi-Modality</b>
      </td>
      <td>
        <b>AIGC</b>
      </td>
    </tr>
     <tr valign="top">
      <td>
      <ul>
        <li><a href="examples/llm/qwen2_legacy/README.md">Qwen2</a></li>
        <li><a href="examples/llm/qwen3_legacy/README.md">Qwen3</a></li>
      </ul>
      </td>
      <td>
      <ul>
        <li><a href="examples/llm/qwen2-vl/README.md">Qwen2-VL</a></li>
      </ul>
      </td>
      <td>
      <ul>
        <li><a href="examples/aigc/sd3/README.md">SD3</a></li>
        <li><a href="examples/aigc/sd3_custom_a/README.md">SD3 2B Custom A</a></li>
        <li><a href="examples/aigc/sd3_5/README.md">SD3.5</a></li>
      </ul>
      </td>
    </tr>
  </tbody>
</table>
