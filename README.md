# XH Model Zoo

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
