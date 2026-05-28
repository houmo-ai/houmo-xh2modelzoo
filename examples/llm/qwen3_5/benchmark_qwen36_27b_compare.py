#!/usr/bin/python3

import argparse
import contextlib
import csv
import dataclasses
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from loguru import logger
from tcim.test_utils.utils import DeviceLock


SCRIPT_DIR = Path(__file__).resolve().parent
TMP_ROOT = SCRIPT_DIR.parents[1]
DFLASH_MODEL_DIR = TMP_ROOT / "dflash" / "qwen36_27b"
DFLASH_DEMO_DIR = TMP_ROOT / "dflash" / "qwen36_35b"

CATEGORIES = ("文本总结", "编程", "常识", "逻辑推理")
LENGTH_BUCKETS = ("short", "medium", "long")
METHODS = ("mtp", "dflash")


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
        question = (
            f"在{_pick(scenarios, index)}里，人们经常会遇到“{_pick(phenomena, index + 3)}”这种现象。"
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
        rel = _pick(relations, index)
        question = (
            f"{n1}、{n2}、{n3}三人各拿一种物品，物品可能是{obj1}、{obj2}和{_pick(objects, index + 5)}。"
            f"已知：{n1}拿的不是{obj1}；{n2}说{n3}拿的是{obj2}；{n3}说{n1}和{n2}都没有拿{obj2}；"
            f"三句话中恰好只有一句为真，并且有一个额外条件是顺序关系“{rel}”。请给出一种自洽分配，"
            "并说明如果条件不足，哪些信息还需要补充。"
        )
        cases.append(BenchmarkCase(_case_id("logic", "long", index), "逻辑推理", "long", question))
    return cases


def build_cases() -> List[BenchmarkCase]:
    cases = _summary_cases() + _programming_cases() + _common_sense_cases() + _logic_cases()
    counts: Dict[Tuple[str, str], int] = {}
    for case in cases:
        counts[(case.category, case.length_bucket)] = counts.get((case.category, case.length_bucket), 0) + 1
    expected = {(category, bucket): 40 for category in CATEGORIES for bucket in LENGTH_BUCKETS}
    if counts != expected:
        raise RuntimeError(f"unexpected case distribution: {counts}")
    return cases


def _set_env(device_ids: Sequence[int]) -> None:
    os.environ.setdefault("HDPL_PLATFORM", "ASIC")
    os.environ["HM_PLATFORM"] = "2"
    os.environ["TCIM_BACKEND"] = "Xh2HalBackend"
    os.environ["TCIM_XH2_USE_SYNC_MODE"] = "1"
    if len(device_ids) == 1:
        os.environ.setdefault("TCIM_LOG_LEVEL", "5")
    os.environ["HDPL_API_TIMEOUT"] = "12345678"


def _load_mtp_demo_class():
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    sys.modules.pop("xh2_qwen_demo", None)
    from xh2_qwen_mtp_demo import Xh2QwenMtpDemo  # pylint: disable=import-outside-toplevel
    from xh2_qwen_demo import reset_aicore  # pylint: disable=import-outside-toplevel

    return Xh2QwenMtpDemo, reset_aicore


def _load_dflash_module():
    if str(DFLASH_DEMO_DIR) in sys.path:
        sys.path.remove(str(DFLASH_DEMO_DIR))
    sys.path.insert(0, str(DFLASH_DEMO_DIR))
    sys.modules.pop("xh2_qwen_demo", None)
    import dflash_demo  # pylint: disable=import-outside-toplevel

    return dflash_demo


def _select_cases(args: argparse.Namespace) -> List[BenchmarkCase]:
    cases = build_cases()
    if args.category:
        selected = set(args.category)
        cases = [case for case in cases if case.category in selected]
    if args.length_bucket:
        selected = set(args.length_bucket)
        cases = [case for case in cases if case.length_bucket in selected]
    if args.case_id:
        selected = set(args.case_id)
        cases = [case for case in cases if case.case_id in selected]
    if args.case_limit > 0:
        cases = cases[: args.case_limit]
    return cases


def _load_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, rows: List[Dict[str, object]]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _status_by_key(rows: Iterable[Dict[str, object]]) -> Dict[Tuple[str, str], str]:
    return {(str(row["method"]), str(row["case_id"])): str(row.get("status", "")) for row in rows}


def _replace_row(rows: List[Dict[str, object]], row: Dict[str, object]) -> None:
    key = (row["method"], row["case_id"])
    for index, old_row in enumerate(rows):
        if (old_row.get("method"), old_row.get("case_id")) == key:
            rows[index] = row
            return
    rows.append(row)


def _fmt_float(value: Optional[float], digits: int = 2) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value:.{digits}f}"


def _fmt_pct(value: Optional[float], digits: int = 2) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value * 100:.{digits}f}%"


def _ok_rows(rows: Iterable[Dict[str, object]], method: Optional[str] = None, category: Optional[str] = None) -> List[Dict[str, object]]:
    selected = [row for row in rows if row.get("status") == "ok"]
    if method is not None:
        selected = [row for row in selected if row.get("method") == method]
    if category is not None:
        selected = [row for row in selected if row.get("category") == category]
    return selected


def _metric_rows(rows: Iterable[Dict[str, object]], metric: str) -> List[Dict[str, object]]:
    out = []
    for row in rows:
        value = row.get(metric)
        if value is not None:
            out.append(row)
    return out


def _summary(rows: List[Dict[str, object]]) -> Dict[str, object]:
    speed_rows = _metric_rows(rows, "decode_tokens_per_s")
    accept_rows = _metric_rows(rows, "acceptance_rate")
    speeds = [float(row["decode_tokens_per_s"]) for row in speed_rows]
    accepts = [float(row["acceptance_rate"]) for row in accept_rows]
    fastest = max(speed_rows, key=lambda row: float(row["decode_tokens_per_s"])) if speed_rows else None
    slowest = min(speed_rows, key=lambda row: float(row["decode_tokens_per_s"])) if speed_rows else None
    best_accept = max(accept_rows, key=lambda row: float(row["acceptance_rate"])) if accept_rows else None
    worst_accept = min(accept_rows, key=lambda row: float(row["acceptance_rate"])) if accept_rows else None
    return {
        "ok": len(rows),
        "median_speed": statistics.median(speeds) if speeds else None,
        "mean_speed": statistics.mean(speeds) if speeds else None,
        "fastest_speed": float(fastest["decode_tokens_per_s"]) if fastest else None,
        "fastest_case_id": fastest.get("case_id") if fastest else None,
        "slowest_speed": float(slowest["decode_tokens_per_s"]) if slowest else None,
        "slowest_case_id": slowest.get("case_id") if slowest else None,
        "median_acceptance": statistics.median(accepts) if accepts else None,
        "mean_acceptance": statistics.mean(accepts) if accepts else None,
        "best_acceptance": float(best_accept["acceptance_rate"]) if best_accept else None,
        "best_acceptance_case_id": best_accept.get("case_id") if best_accept else None,
        "worst_acceptance": float(worst_accept["acceptance_rate"]) if worst_accept else None,
        "worst_acceptance_case_id": worst_accept.get("case_id") if worst_accept else None,
    }


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = [
        "method", "case_id", "category", "length_bucket", "status", "device_ids", "decode_tokens_per_s", "acceptance_rate",
        "decode_tokens", "rounds", "draft_tokens", "accepted_tokens", "prompt_tokens", "question_chars", "elapsed_s",
        "error", "question",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _bar_svg(path: Path, rows: List[Dict[str, object]]) -> None:
    width = 1100
    height = 640
    margin_left = 90
    panel_top = 70
    panel_height = 210
    gap = 80
    colors = {"mtp": "#2563eb", "dflash": "#059669"}
    labels = {"mtp": "MTP", "dflash": "DFlash"}

    def med(method: str, category: str, metric: str) -> Optional[float]:
        selected = _ok_rows(rows, method, category)
        values = [float(row[metric]) for row in selected if row.get(metric) is not None]
        return statistics.median(values) if values else None

    speed_data = {(method, category): med(method, category, "decode_tokens_per_s") for method in METHODS for category in CATEGORIES}
    accept_data = {(method, category): med(method, category, "acceptance_rate") for method in METHODS for category in CATEGORIES}
    max_speed = max([value for value in speed_data.values() if value is not None] or [1.0])
    max_accept = 1.0

    def panel(y0: int, title: str, data: Dict[Tuple[str, str], Optional[float]], max_value: float, unit: str) -> List[str]:
        chart_width = width - margin_left - 60
        base = y0 + panel_height
        group_width = chart_width / len(CATEGORIES)
        bar_width = 42
        out = [
            f'<text x="{margin_left}" y="{y0 - 25}" font-size="22" font-weight="600">{title}</text>',
            f'<line x1="{margin_left}" y1="{base}" x2="{width - 40}" y2="{base}" stroke="#334155" stroke-width="1"/>',
            f'<line x1="{margin_left}" y1="{y0}" x2="{margin_left}" y2="{base}" stroke="#334155" stroke-width="1"/>',
        ]
        for tick in range(5):
            value = max_value * tick / 4
            y = base - panel_height * tick / 4
            out.append(f'<line x1="{margin_left - 5}" y1="{y:.1f}" x2="{width - 40}" y2="{y:.1f}" stroke="#e2e8f0"/>')
            label = f"{value * 100:.0f}%" if unit == "%" else f"{value:.0f}"
            out.append(f'<text x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="#475569">{label}</text>')
        for cidx, category in enumerate(CATEGORIES):
            center = margin_left + group_width * cidx + group_width / 2
            out.append(f'<text x="{center:.1f}" y="{base + 28}" text-anchor="middle" font-size="14">{category}</text>')
            for midx, method in enumerate(METHODS):
                value = data.get((method, category))
                if value is None:
                    continue
                x = center - bar_width - 6 + midx * (bar_width + 12)
                bar_h = 0 if max_value <= 0 else panel_height * value / max_value
                y = base - bar_h
                out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_h:.1f}" fill="{colors[method]}"/>')
                text = f"{value * 100:.1f}%" if unit == "%" else f"{value:.1f}"
                out.append(f'<text x="{x + bar_width / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-size="12" fill="#0f172a">{text}</text>')
        return out

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="640" viewBox="0 0 1100 640">',
        '<rect width="1100" height="640" fill="#ffffff"/>',
        '<text x="40" y="36" font-size="26" font-weight="700">Qwen3.6 27B DFlash vs MTP Benchmark</text>',
        f'<rect x="760" y="18" width="18" height="18" fill="{colors["mtp"]}"/><text x="786" y="33" font-size="15">{labels["mtp"]}</text>',
        f'<rect x="850" y="18" width="18" height="18" fill="{colors["dflash"]}"/><text x="876" y="33" font-size="15">{labels["dflash"]}</text>',
    ]
    lines.extend(panel(panel_top, "Median Decode Speed (tokens/s)", speed_data, max_speed * 1.15, "tok/s"))
    lines.extend(panel(panel_top + panel_height + gap, "Median Acceptance Rate", accept_data, max_accept, "%"))
    lines.append('</svg>')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(path: Path, rows: List[Dict[str, object]], args: argparse.Namespace, chart_path: Path, csv_path: Path) -> None:
    total_cases = len(build_cases())
    lines = [
        "# Qwen3.6 27B DFlash vs MTP Benchmark",
        "",
        f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Target cases: {total_cases} prompts = 4 categories x 120 prompts; each category has 40 short, 40 medium, 40 long prompts.",
        f"- max_new_tokens: {args.max_new_tokens}",
        f"- device_ids: {args.device_id}",
        f"- Results JSON: {args.json}",
        f"- Results CSV: {csv_path.name}",
        f"- Chart: {chart_path.name}",
        "- Speed uses generated decode tokens per second; acceptance rate is accepted draft tokens / draft candidates.",
        "",
        f"![Qwen3.6 27B benchmark chart]({chart_path.name})",
        "",
        "## Completion Matrix",
        "",
        "| Method | Category | OK | Failed | Pending |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        for category in CATEGORIES:
            method_rows = [row for row in rows if row.get("method") == method and row.get("category") == category]
            ok_count = sum(row.get("status") == "ok" for row in method_rows)
            failed_count = sum(row.get("status") == "failed" for row in method_rows)
            pending_count = 120 - ok_count - failed_count
            lines.append(f"| {method} | {category} | {ok_count} | {failed_count} | {pending_count} |")

    lines.extend([
        "",
        "## Overall Summary",
        "",
        "| Method | OK | Median speed | Fastest speed | Slowest speed | Median acceptance | Best acceptance | Worst acceptance |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for method in METHODS:
        stats = _summary(_ok_rows(rows, method))
        lines.append(
            f"| {method} | {stats['ok']} | {_fmt_float(stats['median_speed'])} | "
            f"{_fmt_float(stats['fastest_speed'])} ({stats['fastest_case_id'] or '-'}) | "
            f"{_fmt_float(stats['slowest_speed'])} ({stats['slowest_case_id'] or '-'}) | "
            f"{_fmt_pct(stats['median_acceptance'])} | "
            f"{_fmt_pct(stats['best_acceptance'])} ({stats['best_acceptance_case_id'] or '-'}) | "
            f"{_fmt_pct(stats['worst_acceptance'])} ({stats['worst_acceptance_case_id'] or '-'}) |"
        )

    lines.extend([
        "",
        "## Category Summary",
        "",
        "| Category | Method | OK | Median speed | Mean speed | Fastest speed | Slowest speed | Median acceptance | Mean acceptance | Best acceptance | Worst acceptance |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for category in CATEGORIES:
        for method in METHODS:
            stats = _summary(_ok_rows(rows, method, category))
            lines.append(
                f"| {category} | {method} | {stats['ok']} | {_fmt_float(stats['median_speed'])} | "
                f"{_fmt_float(stats['mean_speed'])} | {_fmt_float(stats['fastest_speed'])} ({stats['fastest_case_id'] or '-'}) | "
                f"{_fmt_float(stats['slowest_speed'])} ({stats['slowest_case_id'] or '-'}) | "
                f"{_fmt_pct(stats['median_acceptance'])} | {_fmt_pct(stats['mean_acceptance'])} | "
                f"{_fmt_pct(stats['best_acceptance'])} ({stats['best_acceptance_case_id'] or '-'}) | "
                f"{_fmt_pct(stats['worst_acceptance'])} ({stats['worst_acceptance_case_id'] or '-'}) |"
            )

    failed = [row for row in rows if row.get("status") == "failed"]
    if failed:
        lines.extend(["", "## Failed Cases", ""])
        for row in failed[:50]:
            error = str(row.get("error", "")).replace("\n", " ")
            lines.append(f"- {row.get('method')} {row.get('case_id')}: {error}")
        if len(failed) > 50:
            lines.append(f"- ... {len(failed) - 50} more")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_artifacts(rows: List[Dict[str, object]], args: argparse.Namespace) -> None:
    json_path = Path(args.json).resolve()
    csv_path = Path(args.csv).resolve()
    report_path = Path(args.report).resolve()
    chart_path = Path(args.chart).resolve()
    _write_json(json_path, rows)
    _write_csv(csv_path, rows)
    _bar_svg(chart_path, rows)
    _write_report(report_path, rows, args, chart_path, csv_path)


@contextlib.contextmanager
def _maybe_suppress_stdout(enabled: bool):
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def _run_mtp(cases: List[BenchmarkCase], rows: List[Dict[str, object]], args: argparse.Namespace) -> None:
    Xh2QwenMtpDemo, reset_aicore = _load_mtp_demo_class()
    if args.reset:
        reset_aicore()
    demo = Xh2QwenMtpDemo(
        tokenizer_path=args.mtp_tokenizer,
        embedding_path=args.mtp_embedding,
        prefill_model=args.mtp_prefill_model,
        mtp_prefill_model=args.mtp_prefill_draft,
        mtp_decode_model=args.mtp_decode_draft,
        verify_model=args.mtp_verify,
        device_ids=args.device_id,
        no_think=True,
        verbose=False,
        debug=False,
        mtp_num_tokens=args.mtp_num_tokens,
        context_length=args.context,
        window_size=args.window_size,
        quiet_perf=True,
    )
    completed = _status_by_key(rows)
    for index, case in enumerate(cases, 1):
        key = ("mtp", case.case_id)
        if args.resume and completed.get(key) == "ok":
            continue
        logger.info(f"BENCH_START method=mtp case={index}/{len(cases)} case_id={case.case_id} category={case.category} bucket={case.length_bucket}")
        start = time.time()
        row: Dict[str, object] = {
            "method": "mtp",
            "case_id": case.case_id,
            "category": case.category,
            "length_bucket": case.length_bucket,
            "device_ids": list(args.device_id),
            "question": case.question,
            "question_chars": len(case.question),
            "status": "ok",
        }
        try:
            demo.session = demo._new_session()
            demo.streaming_decoder.pending_ids = []
            demo.last_perf_summary = {}
            with _maybe_suppress_stdout(args.suppress_output):
                output = demo.chat(case.question, args.max_new_tokens, args.stop_on_repeat_ngram)
            summary = dict(demo.last_perf_summary)
            if output is None or not summary:
                raise RuntimeError("MTP did not produce performance summary")
            row.update(
                {
                    "decode_tokens_per_s": summary.get("decode_tps"),
                    "acceptance_rate": summary.get("acceptance_rate"),
                    "decode_tokens": summary.get("decode_tokens"),
                    "rounds": summary.get("mtp_rounds"),
                    "draft_tokens": summary.get("draft_tokens"),
                    "accepted_tokens": summary.get("accepted_drafts"),
                    "prompt_tokens": summary.get("prefill_tokens"),
                    "elapsed_s": time.time() - start,
                    "raw_stats": summary,
                }
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc()
            row["elapsed_s"] = time.time() - start
            logger.exception(f"BENCH_FAILED method=mtp case_id={case.case_id}: {exc}")
        _replace_row(rows, row)
        completed[key] = str(row["status"])
        _write_artifacts(rows, args)
        logger.info(
            f"BENCH_DONE method=mtp case_id={case.case_id} status={row['status']} "
            f"speed={row.get('decode_tokens_per_s')} acceptance={row.get('acceptance_rate')}"
        )


def _run_dflash(cases: List[BenchmarkCase], rows: List[Dict[str, object]], args: argparse.Namespace) -> None:
    dflash_demo = _load_dflash_module()
    if args.reset:
        dflash_demo.reset_aicore()
    demo = dflash_demo.DFlashDemo(
        tokenizer_path=args.dflash_tokenizer,
        embedding_path=args.dflash_embedding,
        prefill_model=args.dflash_prefill,
        prefill_context_model=args.dflash_prefill_context,
        decode_context_model=args.dflash_decode_context,
        draft_model=args.dflash_draft,
        verify_model=args.dflash_decode_verify,
        device_ids=args.device_id,
        no_think=True,
        verbose=False,
        debug=False,
        context_length=args.context,
        max_new_tokens=args.max_new_tokens,
        stop_on_repeat_ngram=args.stop_on_repeat_ngram,
    )
    completed = _status_by_key(rows)
    for index, case in enumerate(cases, 1):
        key = ("dflash", case.case_id)
        if args.resume and completed.get(key) == "ok":
            continue
        logger.info(f"BENCH_START method=dflash case={index}/{len(cases)} case_id={case.case_id} category={case.category} bucket={case.length_bucket}")
        start = time.time()
        row: Dict[str, object] = {
            "method": "dflash",
            "case_id": case.case_id,
            "category": case.category,
            "length_bucket": case.length_bucket,
            "device_ids": list(args.device_id),
            "question": case.question,
            "question_chars": len(case.question),
            "status": "ok",
        }
        try:
            demo.session = dflash_demo.Session(messages=[])
            demo.streaming_decoder.pending_ids = []
            demo.last_stats = {}
            with _maybe_suppress_stdout(args.suppress_output):
                output = demo.chat(case.question)
            summary = dict(demo.last_stats)
            if output is None or not summary:
                raise RuntimeError("DFlash did not produce performance summary")
            row.update(
                {
                    "decode_tokens_per_s": summary.get("decode_tokens_per_s"),
                    "acceptance_rate": summary.get("acceptance_rate"),
                    "decode_tokens": summary.get("decode_tokens"),
                    "rounds": summary.get("rounds"),
                    "draft_tokens": summary.get("draft_total"),
                    "accepted_tokens": summary.get("accepted_total"),
                    "prompt_tokens": summary.get("prompt_tokens"),
                    "elapsed_s": time.time() - start,
                    "raw_stats": summary,
                }
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc()
            row["elapsed_s"] = time.time() - start
            logger.exception(f"BENCH_FAILED method=dflash case_id={case.case_id}: {exc}")
        _replace_row(rows, row)
        completed[key] = str(row["status"])
        _write_artifacts(rows, args)
        logger.info(
            f"BENCH_DONE method=dflash case_id={case.case_id} status={row['status']} "
            f"speed={row.get('decode_tokens_per_s')} acceptance={row.get('acceptance_rate')}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Qwen3.6 27B MTP and DFlash on the same prompt set.")
    parser.add_argument("--method", choices=("mtp", "dflash", "both", "report"), default="both")
    parser.add_argument("--device-id", type=int, nargs="+", default=[0])
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--mtp-num-tokens", type=int, default=4)
    parser.add_argument("--context", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=0)
    parser.add_argument("--stop-on-repeat-ngram", type=int, default=8)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--show-output", action="store_false", dest="suppress_output")
    parser.set_defaults(suppress_output=True)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--category", action="append", choices=CATEGORIES, default=[])
    parser.add_argument("--length-bucket", action="append", choices=LENGTH_BUCKETS, default=[])
    parser.add_argument("--json", default="qwen36_27b_compare_results.json")
    parser.add_argument("--csv", default="qwen36_27b_compare_results.csv")
    parser.add_argument("--report", default="qwen36_27b_compare_report.md")
    parser.add_argument("--chart", default="qwen36_27b_compare_chart.svg")

    parser.add_argument("--mtp-tokenizer", default=str(SCRIPT_DIR / "qwen36_27b_tokenizer"))
    parser.add_argument("--mtp-embedding", default=str(SCRIPT_DIR / "qwen36_27b_embedding" / "quant_embedding.pt"))
    parser.add_argument("--mtp-prefill-model", default=str(SCRIPT_DIR / "qwen3.6_prefill.hmm"))
    parser.add_argument("--mtp-decode-draft", default=str(SCRIPT_DIR / "qwen3.6_decode_mtp.hmm"))
    parser.add_argument("--mtp-prefill-draft", default=str(SCRIPT_DIR / "qwen3.6_prefill_mtp.hmm"))
    parser.add_argument("--mtp-verify", default=str(SCRIPT_DIR / "qwen3.6_decode.hmm"))

    parser.add_argument("--dflash-tokenizer", default=str(DFLASH_MODEL_DIR / "hf_config"))
    parser.add_argument("--dflash-embedding", default=str(DFLASH_MODEL_DIR / "token_embedding.pt"))
    parser.add_argument("--dflash-prefill", default=str(DFLASH_MODEL_DIR / "prefill.hmm"))
    parser.add_argument("--dflash-prefill-context", default=str(DFLASH_MODEL_DIR / "prefill_context.hmm"))
    parser.add_argument("--dflash-decode-context", default=str(DFLASH_MODEL_DIR / "decode_context.hmm"))
    parser.add_argument("--dflash-draft", default=str(DFLASH_MODEL_DIR / "dflash_draft.hmm"))
    parser.add_argument("--dflash-decode-verify", default=str(DFLASH_MODEL_DIR / "decode_verify.hmm"))
    return parser.parse_args()


def _artifact_path(path: str, method: str) -> Path:
    original = Path(path).resolve()
    return original.with_name(f"{original.stem}.{method}{original.suffix}")


def _strip_replaced_child_args(args: Sequence[str]) -> List[str]:
    one_value_options = {"--method", "--json", "--csv", "--report", "--chart"}
    many_value_options = {"--device-id"}
    out: List[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if any(arg.startswith(f"{option}=") for option in one_value_options | many_value_options):
            index += 1
            continue
        if arg in one_value_options:
            index += 2
            continue
        if arg in many_value_options:
            index += 1
            while index < len(args) and not args[index].startswith("--"):
                index += 1
            continue
        out.append(arg)
        index += 1
    return out


def _child_command(method: str, device_ids: Sequence[int], args: argparse.Namespace) -> List[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        *_strip_replaced_child_args(sys.argv[1:]),
        "--method",
        method,
        "--device-id",
        *(str(device_id) for device_id in device_ids),
        "--json",
        str(_artifact_path(args.json, method)),
        "--csv",
        str(_artifact_path(args.csv, method)),
        "--report",
        str(_artifact_path(args.report, method)),
        "--chart",
        str(_artifact_path(args.chart, method)),
    ]


def _seed_method_json(args: argparse.Namespace, method: str, main_rows: List[Dict[str, object]]) -> None:
    method_json = _artifact_path(args.json, method)
    if not args.resume:
        _write_json(method_json, [])
        return
    if method_json.exists():
        return
    _write_json(method_json, [row for row in main_rows if row.get("method") == method])


def _merge_method_rows(args: argparse.Namespace, main_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    merged = [row for row in main_rows if row.get("method") not in METHODS]
    for method in METHODS:
        method_json = _artifact_path(args.json, method)
        if method_json.exists():
            method_rows = _load_rows(method_json)
        else:
            method_rows = [row for row in main_rows if row.get("method") == method]
        for row in method_rows:
            if row.get("method") == method:
                _replace_row(merged, row)
    merged.sort(key=lambda row: (str(row.get("case_id", "")), str(row.get("method", ""))))
    return merged


def _run_both_in_child_processes(args: argparse.Namespace) -> None:
    main_rows = _load_rows(Path(args.json).resolve())
    for method in METHODS:
        _seed_method_json(args, method, main_rows)

    if len(args.device_id) >= 2:
        assignments = {"mtp": [args.device_id[0]], "dflash": [args.device_id[1]]}
        logger.info(f"BENCH_BOTH_MODE parallel assignments={assignments}")
        processes = []
        for method, device_ids in assignments.items():
            command = _child_command(method, device_ids, args)
            logger.info(f"BENCH_CHILD_START method={method} device_ids={device_ids} command={' '.join(command)}")
            processes.append((method, subprocess.Popen(command)))
        failures = []
        for method, process in processes:
            return_code = process.wait()
            if return_code:
                failures.append((method, return_code))
            logger.info(f"BENCH_CHILD_DONE method={method} return_code={return_code}")
        rows = _merge_method_rows(args, main_rows)
        _write_artifacts(rows, args)
        if failures:
            raise RuntimeError(f"both-mode child process failures: {failures}")
        return

    logger.warning("BENCH_BOTH_MODE sequential_single_device; pass at least two --device-id values to run both methods concurrently")
    for method in METHODS:
        command = _child_command(method, args.device_id, args)
        logger.info(f"BENCH_CHILD_START method={method} device_ids={args.device_id} command={' '.join(command)}")
        subprocess.run(command, check=True)
        logger.info(f"BENCH_CHILD_DONE method={method}")
    rows = _merge_method_rows(args, main_rows)
    _write_artifacts(rows, args)


def main() -> None:
    args = parse_args()
    _set_env(args.device_id)
    rows = _load_rows(Path(args.json).resolve())
    cases = _select_cases(args)
    logger.info(
        f"BENCH_PLAN method={args.method} selected_cases={len(cases)} max_new_tokens={args.max_new_tokens} "
        f"device_ids={args.device_id} resume={args.resume}"
    )
    if args.method == "report":
        _write_artifacts(rows, args)
        return
    if args.method == "both":
        _run_both_in_child_processes(args)
        return

    with contextlib.ExitStack() as stack:
        for device_id in sorted(args.device_id):
            stack.enter_context(DeviceLock("xh2", device_id, "qwen36_27b_compare"))
        if args.method == "mtp":
            _run_mtp(cases, rows, args)
        if args.method == "dflash":
            _run_dflash(cases, rows, args)

    _write_artifacts(rows, args)
    logger.info(f"BENCH_REPORT {Path(args.report).resolve()}")
    logger.info(f"BENCH_JSON {Path(args.json).resolve()}")
    logger.info(f"BENCH_CSV {Path(args.csv).resolve()}")
    logger.info(f"BENCH_CHART {Path(args.chart).resolve()}")


if __name__ == "__main__":
    main()