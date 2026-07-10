"""
Do not import model-specific packages in this file.

模型迁移规则
1.  yaml配置文件：量化配置放quant字段，导出配置放export字段
2.  type字段位置固定export.model.type。原本的传入MODELS.build()的配置文件直接对应export.model整个字段，
    如果有多个子模型，选一个主模型作为export.model，子模型的可以对应export.modelA / export.modelB。
    其余导出阶段用到的配置可以放在export.xxxx，总的来说除了export.model必要，其他相对自由。
3.  必须能够通过yaml配置文件控制每个子模型的量化导出精度
4.  必须要写芯片架构参数，例如target_device，没有特殊情况，建议放在export.target_device
5.  导出的产物命名要体现量化精度和芯片架构
6.  export接口中使用的workflow_config，需要dump到export接口的output_dir路径下
7.  dump_golden接口必须实现，导出golden功能放在该接口中，export接口不要有任何导出golden相关的功能
8.  原则上，应当保证在相同的配置下，迁移后导出的hmoonx与迁移前完全一致


开发规范
1.  workflow_config不允许写回workflow_config.data
2.  模型workflow.py中保持代码简洁。不要设计复杂、无意义的抽象
3.  原始py配置文件存在继承，迁移到yaml中时务必完整迁移
4.  迁移完成后，不可以依赖xh2_model_zoo或examples中任何文件，做到完整迁移
"""