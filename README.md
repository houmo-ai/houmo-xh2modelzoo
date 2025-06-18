# XH Model Zoo

## 开发手册

这里(<https://houmo.feishu.cn/wiki/GjWNwICABiBm8Ykn0GAcRKnen6d>)

## 依赖项

使用xhquanttool工程的python环境

## 代码提交  

```拉取代码
git pull --rebase
```

```bash
在develop分支上提交代码，提交前请确保代码已经通过测试。  
git push origin HEAD:refs/for/develop
```

## 量化格式

- w8a8-sefp: 8bit 权重, 8bit 激活, 计算模式：sefp
- w4a8-ssfp: 4bit 权重, 8bit 激活, 计算模式：ssfp
- w8a16-sefp: 8bit 权重, 16bit 激活, 计算模式：sefp

## Model zoo

<table align="center">
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
        <li><a href="examples/aigc/sd3_lenovo/README.md">SD3 2B Lenovo</a></li>
        <li><a href="examples/aigc/sd3_5/README.md">SD3.5</a></li>
      </ul>
      </td>
    </tr>
  </tbody>
</table>
