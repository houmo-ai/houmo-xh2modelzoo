"""Shared benchmark case definitions for Qwen3.6 27B MTP vs DFlash comparisons.

No external dependencies beyond stdlib — safe to import from any environment.
"""

import dataclasses
from typing import Dict, List, Sequence, Tuple

CATEGORIES = ("文本总结", "编程", "常识", "逻辑推理")
LENGTH_BUCKETS = ("short", "medium", "long")


@dataclasses.dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    category: str
    length_bucket: str
    question: str


def _pick(values: Sequence[str], index: int) -> str:
    return values[index % len(values)]


def _case_id(prefix: str, bucket: str, index: int) -> str:
    return f"{prefix}_{bucket}_{index + 1:03d}"


def _summary_cases() -> List[BenchmarkCase]:
    domains = [
        "社区养老", "智能制造", "高校教学", "城市交通", "县域电商", "医院门诊", "物流仓储", "园区能源",
        "银行风控", "博物馆导览", "城市更新", "企业知识库", "餐饮排班", "政务服务", "车企售后", "垃圾分类",
        "新闻编辑", "药物研发", "机场安检", "夜间公交", "设备运维", "出版发行", "连锁药店", "城市水务",
        "教育作业", "港口调度", "保险理赔", "家庭医生", "景区预约", "零售补货", "实验室共享", "智慧停车",
        "协作机械臂", "远程柜员", "农业灌溉", "应急指挥", "视频转码", "招聘初筛", "地方文献", "数字孪生",
    ]
    systems = [
        "把多个入口整合到统一平台", "用本地模型做初筛再由人工复核", "引入预约制和动态分配", "基于历史数据预测高峰",
        "打通线上申请和线下服务", "通过传感器实时监测状态", "按风险等级安排处置", "把低频资料开放全文检索",
    ]
    benefits = [
        "等待时间明显缩短", "人工重复工作减少", "资源利用率提高", "异常发现更及时", "用户触达范围扩大",
        "高峰期体验改善", "运营成本下降", "决策依据更透明",
    ]
    caveats = [
        "旧数据质量仍需治理", "特殊场景仍要人工兜底", "设备维护成本需要分摊", "跨部门标准还不统一",
        "隐私授权需要更清晰", "高峰期容量仍需扩展", "现场引导还要优化", "模型结果必须持续校验",
    ]
    actions = [
        "增加分级告警", "补充人工复核流程", "完善权限管理", "优化移动端入口", "定期回收用户反馈",
        "制定应急预案", "统一数据口径", "把试点经验复制到更多区域",
    ]

    cases: List[BenchmarkCase] = []
    for index in range(40):
        text = (
            f"{_pick(domains, index)}项目{_pick(systems, index)}。试运行后，{_pick(benefits, index)}，"
            f"但{_pick(caveats, index)}。请用三句话总结核心内容。"
        )
        cases.append(BenchmarkCase(_case_id("summary", "short", index), "文本总结", "short", text))

    for index in range(40):
        text = (
            f"某地推进{_pick(domains, index + 5)}数字化改造，第一阶段重点是{_pick(systems, index + 1)}，"
            f"第二阶段计划{_pick(actions, index + 2)}。上线三个月后，{_pick(benefits, index + 3)}，"
            f"一线人员反馈流程更顺，但{_pick(caveats, index + 4)}。管理方希望在扩大范围前先评估投入产出、"
            "用户满意度和长期维护压力。请概括背景、成效和风险。"
        )
        cases.append(BenchmarkCase(_case_id("summary", "medium", index), "文本总结", "medium", text))

    for index in range(40):
        text = (
            f"请阅读材料并写一段结构化摘要。材料：在{_pick(domains, index + 11)}场景中，团队先梳理线下流程，"
            f"发现主要瓶颈来自信息分散、责任边界不清和高峰期处理能力不足。随后项目组{_pick(systems, index + 2)}，"
            f"并配套{_pick(actions, index + 3)}。试点期间，{_pick(benefits, index + 4)}，相关人员也认为数据看板让问题暴露更早。"
            f"不过，{_pick(caveats, index + 5)}，部分老用户仍偏好原有渠道，预算部门也关注后续扩容和维护成本。"
            "下一步，项目组准备扩大样本、补齐培训材料，并把关键指标按月复盘。请总结主要做法、收益、限制和下一步。"
        )
        cases.append(BenchmarkCase(_case_id("summary", "long", index), "文本总结", "long", text))
    return cases


def _programming_cases() -> List[BenchmarkCase]:
    short_tasks = [
        "判断字符串是否为回文，忽略大小写和非字母数字字符",
        "合并两个有序列表并返回新的有序列表",
        "统计列表中出现次数最多的元素",
        "把秒数格式化为 HH:MM:SS",
        "删除列表中重复元素并保持原顺序",
        "检查括号字符串是否合法",
        "计算斐波那契数列第 n 项，要求迭代实现",
        "把嵌套列表拍平成一层列表",
        "解析形如 key=value 的配置行",
        "找出数组中和为目标值的两个下标",
    ]
    data_types = ["list[int]", "dict[str, int]", "str", "list[str]", "tuple[int, int]", "set[str]"]
    constraints = [
        "输入可能为空", "需要保持稳定顺序", "时间复杂度尽量为 O(n)", "不要修改原始输入", "需要处理非法值",
        "结果要便于单元测试", "不要使用全局变量", "请补充两个示例",
    ]
    services = [
        "订单去重", "日志聚合", "库存预警", "用户分群", "任务调度", "缓存淘汰", "接口限流", "配置合并",
        "成绩统计", "路径匹配", "告警压缩", "报表分页", "文件索引", "实验分桶", "消息重试", "权限校验",
    ]
    cases: List[BenchmarkCase] = []
    for index in range(40):
        question = (
            f"请用 Python 写一个函数，功能是{_pick(short_tasks, index)}。"
            f"输入类型可以假设为 {_pick(data_types, index)}，并且{_pick(constraints, index)}。"
        )
        cases.append(BenchmarkCase(_case_id("coding", "short", index), "编程", "short", question))

    for index in range(40):
        question = (
            f"请实现一个 Python 函数用于{_pick(services, index)}。输入包含若干记录，每条记录至少有 id、时间戳和状态字段；"
            f"需要先过滤无效记录，再按业务键分组，最后输出排序后的统计结果。要求：{_pick(constraints, index + 2)}，"
            f"{_pick(constraints, index + 5)}，并解释核心思路和复杂度。"
        )
        cases.append(BenchmarkCase(_case_id("coding", "medium", index), "编程", "medium", question))

    for index in range(40):
        question = (
            f"你正在为一个{_pick(services, index + 3)}模块写核心逻辑。输入是一批 JSON 风格的字典记录，字段包括 user_id、"
            "event_type、timestamp、payload 和 retry_count，其中部分字段可能缺失或类型错误。请设计并实现 Python 代码："
            "1. 校验并清洗输入；2. 按用户和事件类型聚合最近 24 小时的数据；3. 输出可序列化的摘要结构；"
            f"4. 对异常记录给出可追踪的错误列表。要求{_pick(constraints, index + 1)}，{_pick(constraints, index + 6)}，"
            "同时给出一个小型示例输入和输出。"
        )
        cases.append(BenchmarkCase(_case_id("coding", "long", index), "编程", "long", question))
    return cases


def _common_sense_cases() -> List[BenchmarkCase]:
    phenomena = [
        "下雨后空气让人感觉更清新", "冰箱频繁开门会更耗电", "高原地区水更容易沸腾", "远光灯夜间让人刺眼",
        "运动后立刻喝大量冰水容易不适", "低温下手机电池掉电更快", "金属摸起来比木头更凉", "太阳落山时颜色偏红",
        "电脑风扇积灰会影响性能", "热身能降低运动受伤概率", "宽口杯里的热水凉得更快", "浴室镜子洗澡后起雾",
        "肥皂能洗掉手上的油污", "轮胎气压过低会增加油耗", "蔬菜放久会失水变蔫", "飞机起降时耳朵发闷",
    ]
    scenarios = [
        "家庭厨房", "办公室", "学校宿舍", "高速列车", "医院候诊区", "户外运动", "地下车库", "商场电梯",
        "城市道路", "山区旅行", "实验室", "工厂车间", "图书馆", "机场", "社区服务站", "快递网点",
    ]
    factors = [
        "温度变化", "空气流动", "材料导热", "压强差异", "水分蒸发", "微生物繁殖", "能量损耗", "光线散射",
        "摩擦力", "电池化学反应", "人体感知", "设备维护", "安全冗余", "信息标识", "负载分布", "通风条件",
    ]
    cases: List[BenchmarkCase] = []
    for index in range(40):
        question = f"为什么{_pick(phenomena, index)}？请用通俗语言解释。"
        cases.append(BenchmarkCase(_case_id("common", "short", index), "常识", "short", question))

    for index in range(40):
        phenomenon = _pick(phenomena, index + 3)
        question = (
            f"在{_pick(scenarios, index)}里，人们经常会遇到“{phenomenon}”这种现象。"
            f"请结合{_pick(factors, index)}和{_pick(factors, index + 5)}解释原因，并给出一个生活中的应对建议。"
        )
        cases.append(BenchmarkCase(_case_id("common", "medium", index), "常识", "medium", question))

    for index in range(40):
        question = (
            f"某人在{_pick(scenarios, index + 4)}观察到几个现象：一是{_pick(phenomena, index + 6)}，"
            f"二是{_pick(phenomena, index + 9)}，三是环境中的{_pick(factors, index + 2)}发生了变化。"
            "请分别解释这些现象背后的常识原理，指出哪些因素可能相互影响，并给出两个安全或节能方面的建议。"
        )
        cases.append(BenchmarkCase(_case_id("common", "long", index), "常识", "long", question))
    return cases


def _logic_cases() -> List[BenchmarkCase]:
    names = ["甲", "乙", "丙", "丁", "小王", "小李", "小张", "小赵"]
    objects = ["红球", "蓝球", "黄球", "白球", "书", "票", "硬币", "钥匙"]
    relations = ["比", "少于", "多于", "等于", "早于", "晚于", "不在", "必须在"]
    cases: List[BenchmarkCase] = []
    for index in range(40):
        a = 2 + index % 7
        b = 3 + (index * 2) % 9
        c = a * 3 + b
        question = f"一个数乘以 3 再加 {b} 等于 {c}，这个数是多少？请写出一步推理。"
        cases.append(BenchmarkCase(_case_id("logic", "short", index), "逻辑推理", "short", question))

    for index in range(40):
        n1 = _pick(names, index)
        n2 = _pick(names, index + 1)
        n3 = _pick(names, index + 2)
        total = 45 + index % 12
        diff1 = 2 + index % 5
        diff2 = 3 + index % 4
        question = (
            f"{n1}比{n2}大{diff1}岁，{n2}比{n3}大{diff2}岁，三人年龄和为{total}岁。"
            "请列式求三人的年龄；如果出现非整数年龄，也请说明原因。"
        )
        cases.append(BenchmarkCase(_case_id("logic", "medium", index), "逻辑推理", "medium", question))

    for index in range(40):
        n1 = _pick(names, index)
        n2 = _pick(names, index + 2)
        n3 = _pick(names, index + 4)
        obj1 = _pick(objects, index)
        obj2 = _pick(objects, index + 3)
        obj3 = _pick(objects, index + 5)
        rel = _pick(relations, index)
        question = (
            f"{n1}、{n2}、{n3}三人各拿一种物品，物品可能是{obj1}、{obj2}和{obj3}。"
            f"已知：{n1}拿的不是{obj1}；{n2}说{n3}拿的是{obj2}；{n3}说{n1}和{n2}都没有拿{obj2}；"
            + f"三句话中恰好只有一句为真，并且有一个额外条件是顺序关系“{rel}”。请给出一种自洽分配，"
            "并说明如果条件不足，哪些信息还需要补充。"
        )
        cases.append(BenchmarkCase(_case_id("logic", "long", index), "逻辑推理", "long", question))
    return cases


def build_cases() -> List[BenchmarkCase]:
    cases = _summary_cases() + _programming_cases() + _common_sense_cases() + _logic_cases()
    counts: Dict[Tuple[str, str], int] = {}
    for case in cases:
        counts[(case.category, case.length_bucket)] = counts.get((case.category, case.length_bucket), 0) + 1
    expected = {(cat, bucket): 40 for cat in CATEGORIES for bucket in LENGTH_BUCKETS}
    if counts != expected:
        raise RuntimeError(f"unexpected case distribution: {counts}")
    return cases
