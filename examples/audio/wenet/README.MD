# Wenet模型分为两种类型

1. 有最大长度限制的定长模型
   - 路径: `examples/audio/wenet/fix_length/`
   - 支持最大长度为1600的音频输入
   - 包含模型下载、导出和GPU仿真等功能
   - 详细使用说明见 [fix_length/README.md](fix_length/README.md)

2. 任意长度的不定长模型
   - 路径: `examples/audio/wenet/unfix_length/`
   - 支持任意长度的音频输入
   - 包含模型下载、导出和GPU仿真等功能
   - 详细使用说明见 [unfix_length/README.md](unfix_length/README.md)
