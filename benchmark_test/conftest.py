import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

import pytest
import yagmail
from allure_commons import plugin_manager
from allure_commons._hooks import AllureDeveloperHooks
from allure_commons.model2 import TestResult
from allure_commons.types import AttachmentType
from dotenv import load_dotenv
from jinja2 import Template
from pluggy import HookimplMarker
from tinydb import Query, TinyDB

# 将项目根目录添加到 sys.path
# root_dir = Path(__file__).parent.parent.parent.resolve()
# print(root_dir)
# sys.path.append(str("/data01/user/xuchen/work/release/hmodel"))

# 加载环境变量
load_dotenv(os.path.join((os.path.dirname(__file__)), ".env_benchmark"))
ABS_PATH = os.path.dirname(__file__)

# 全局变量用于存储插件实例
_allure_plugin = None


class TestResultManager:
    yag: yagmail.SMTP

    def __init__(self, db_path="TEST_DATABASE.json"):
        self.allure_results = dict()
        if os.path.dirname(db_path):
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db = TinyDB(db_path)
        self.yag = yagmail.SMTP(
            os.getenv("EMAIL_USER", "brotherhappy520@163.com"),
            os.getenv("EMAIL_PASSWORD", "ARnfueNiV7jFHZ3Y"),
            host=os.getenv("SMTP_SERVER", "smtp.163.com"),
        )
        self.change_id = "without provide Change-ID"

    def get_allure_timestamp(self, allure_results_dir="allure-results"):
        """从allure-results目录中获取最新的时间戳

        Args:
            allure_results_dir: allure-results目录的路径

        Returns:
            str: ISO格式的时间戳字符串
        """
        latest_timestamp = None

        # 遍历allure-results目录中的所有json文件
        for filename in os.listdir(allure_results_dir):
            if filename.endswith("-result.json"):
                file_path = os.path.join(allure_results_dir, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        # 获取stop时间戳（毫秒）
                        stop_timestamp = data.get("stop")
                        if stop_timestamp:
                            # 将毫秒时间戳转换为datetime对象
                            dt = datetime.fromtimestamp(stop_timestamp / 1000)
                            # 转换为ISO格式字符串
                            iso_timestamp = dt.isoformat()

                            # 更新最新的时间戳
                            if latest_timestamp is None or iso_timestamp > latest_timestamp:
                                latest_timestamp = iso_timestamp
                except Exception as e:
                    print(f"处理文件 {filename} 时出错: {str(e)}")
                    continue

        return latest_timestamp or datetime.now().isoformat()

    def add_test_result(self, title, result_data):
        """添加单个测试结果到数据库"""
        table = self.db.table(title)
        result_data["timestamp"] = datetime.now().isoformat()
        table.insert(result_data)

    def add_test_results(self, results: List[dict]):
        """批量添加测试结果到数据库，使用allure-results中的时间戳

        Args:
            results: 测试结果列表，每个结果必须包含 'title' 字段
        """
        if not results:
            return

        # 按title分组处理结果
        for result in results:
            title = result.get("title")
            if not title:
                continue

            # 添加时间戳
            timestamp = result.get("timestamp", datetime.now().isoformat())
            result["timestamp"] = timestamp

            # 获取对应的数据表
            table = self.db.table(title)

            # 检查该时间戳是否已存在
            Test = Query()
            existing_results = table.search(Test.timestamp == timestamp)

            # 如果该时间戳的数据不存在，则插入
            if not existing_results:
                table.insert(result)
            else:
                print(f"时间戳 {timestamp} 的数据已存在，跳过插入")

    def get_all_results(self):
        """获取所有测试结果"""
        results = {}
        for table_name in self.db.tables():
            table = self.db.table(table_name)
            results[table_name] = table.all()
        return results

    def generate_markdown_report(self, results):
        """生成Markdown格式报告"""
        markdown_content = []
        markdown_content.append(f"Use XHQuantool with Change-ID ## {self.change_id}")
        for title, title_results in results.items():
            columns = dict()
            for rst in title_results:
                columns.update(rst)
            columns = list(columns.keys())
            columns.remove("title")

            # 添加标题
            markdown_content.append(f"## {title}")

            # 添加表头
            header = "| " + " | ".join(columns) + " |"
            # separator = "| " + " | ".join(["---"] * len(columns)) + " |"
            markdown_content.extend([header])

            # 按时间戳倒序排序
            sorted_results = sorted(title_results, key=lambda x: x.get("timestamp", ""), reverse=True)

            # 添加数据行
            for i, result in enumerate(sorted_results):
                # 第一行使用加粗
                if i == 0:
                    row = "| " + " | ".join(f"**{str(result.get(col, ''))}**" for col in columns) + " |"
                else:
                    row = "| " + " | ".join(str(result.get(col, "")) for col in columns) + " |"
                markdown_content.append(row)

            # 添加空行分隔
            markdown_content.append("\n")

        markdown_content = "".join(markdown_content)
        markdown_path = "/tmp/test_report.md"
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        return markdown_path

    def generate_email_content(self, results):
        """生成邮件正文HTML内容，显示所有测试结果

        Args:
            results: 测试结果字典，key为表名，value为该表的所有记录列表
        """
        html_parts = []

        # 添加工具和变更ID信息
        html_parts.append(f'<div style="font-family: Arial, sans-serif; line-height: 1.4; margin-bottom: 20px;">')
        html_parts.append(f'<div style="color: #555; font-size: 14px; margin-bottom: 10px;">')
        html_parts.append(f"<b>Use HMQuantool with Change-ID ## {self.change_id}</b>")
        html_parts.append(f"</div>")

        # 添加测试说明
        html_parts.append(
            f'<div style="color: #333; font-size: 13px; background-color: #f8f8f8; padding: 10px; border-left: 3px solid #4CAF50; margin-bottom: 20px;">'
        )
        html_parts.append(f'<p style="margin: 0 0 5px 0; font-weight: bold;">测试说明：</p>')
        html_parts.append(f'<ul style="margin: 0; padding-left: 20px; line-height: 1.6;">')
        html_parts.append(f"<li>此邮件每周五定期发送</li>")
        html_parts.append(f"<li>可主动发送标题或内容包含<code>test</code>邮件至该邮箱进行标准测试</li>")
        html_parts.append(
            f"<li>标题或内容单独包含 <code>int8</code>/<code>mix</code>/<code>int16</code> 可进行单一模式测试</li>"
        )
        html_parts.append(f"<li>包含 <code>fast</code> 使用10张子图数据测试模型 e.g.(test int8 fast)</li>")
        html_parts.append(f"</ul>")
        html_parts.append(f"</div>")

        # 添加表格标题
        html_parts.append('<h3 style="color: #333; margin: 15px 0;">测试结果</h3>')
        html_parts.append("</div>")

        # 遍历每个表
        for title, records in results.items():
            if not records:
                continue

            # 为当前表创建列标题
            columns = []
            for record in records:
                for key in record.keys():
                    if key not in columns:
                        columns.append(key)

            # 添加表名作为二级标题
            html_parts.append(f'<h4 style="color: #666; margin: 10px 0;">{title}</h4>')

            # 添加表格
            html_parts.append('<table style="border-collapse: collapse; width: 100%; margin: 0 0 20px 0;">')

            # 添加表头
            html_parts.append('<tr style="background-color: #f2f2f2;">')
            for col in columns:
                html_parts.append(
                    f'<th style="border: 1px solid #ddd; padding: 6px; text-align: left; font-size: 14px;">{col}</th>'
                )
            html_parts.append("</tr>")

            # 按时间戳倒序排序记录
            sorted_records = sorted(records, key=lambda x: x.get("timestamp", ""), reverse=True)

            # 添加数据行
            for i, record in enumerate(sorted_records):
                bg_color = "#f9f9f9" if i % 2 == 0 else "#ffffff"
                html_parts.append(f'<tr style="background-color: {bg_color};">')
                for col in columns:
                    html_parts.append(
                        f'<td style="border: 1px solid #ddd; padding: 6px; text-align: left; font-size: 14px;">{record.get(col, "")}</td>'
                    )
                html_parts.append("</tr>")

            html_parts.append("</table>")

        return "".join(html_parts)

    def generate_report(self, current_results: List = None):
        """生成报告"""
        # 获取数据库中的历史结果
        db_results = self.get_all_results()

        # 合并当前结果和历史结果
        merged_results = {}
        if current_results:
            # 处理当前结果
            for result in current_results:
                if "stderr" in result.keys():
                    result.pop("stderr")
                title = result.get("title", "未命名测试")
                if title not in merged_results:
                    merged_results[title] = []
                merged_results[title].append(result)

            # 合并历史结果
            for title, results in db_results.items():
                if title not in merged_results:
                    merged_results[title] = []
                merged_results[title].extend(results)

            # 按时间戳排序并删除重复项
            for title in merged_results:
                # 按时间戳排序

                merged_results[title].sort(key=lambda x: x.get("timestamp", ""), reverse=True)

                # 使用字典去重，保留最新的记录
                unique_results = {}
                for result in merged_results[title]:
                    timestamp = result.get("timestamp", "")
                    if timestamp not in unique_results:
                        unique_results[timestamp] = result

                # 更新结果列表
                merged_results[title] = list(unique_results.values())
        else:
            merged_results = db_results

        # 生成Markdown报告
        markdown_path = self.generate_markdown_report(merged_results)
        html_path = self.generate_html_report(merged_results)

        # 生成邮件正文HTML内容（只使用当前结果）
        current_results_dict = {}
        if current_results:
            for result in current_results:
                if "stderr" in result.keys():
                    result.pop("stderr")
                
                print_fields = ["stdout", "stderr", "console", "print", "日志", "输出"]
                for field in print_fields:
                    if field in result.keys():
                        result.pop(field)
                
                title = result.get("title", "未命名测试")
                if title not in current_results_dict:
                    current_results_dict[title] = []
                current_results_dict[title].append(result)
        email_content = self.generate_email_content(current_results_dict)

        return html_path, markdown_path, email_content

    def generate_html_report(self, data: Dict[str, List[Dict]]) -> str:
        """生成HTML格式的测试报告

        Args:
            data: 测试结果数据，格式为 {title: [result1, result2, ...]}

        Returns:
            str: HTML报告的文件路径
        """
        # 为每个测试标题生成列名
        title_columns = {}
        for title, results in data.items():
            if results:
                # 获取所有可能的列名
                columns = set()
                for result in results:
                    columns.update(result.keys())
                title_columns[title] = sorted(list(columns))

        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                tr:first-child { background-color: #e6e6e6; font-weight: bold; }
                h2 { color: #333; margin-top: 30px; }
            </style>
        </head>
        <body>
            {% for title, results in data.items() %}
            <h2>{{ title }}</h2>
            <table>
                <tr>
                    {% for col in title_columns[title] %}
                    {% if col != 'title' %}
                    <th>{{ col }}</th>
                    {% endif %}
                    {% endfor %}
                </tr>
                {% for result in results|sort(attribute='timestamp', reverse=true) %}
                <tr {% if loop.first %}style="background-color: #e6e6e6; font-weight: bold;"{% endif %}>
                    {% for col in title_columns[title] %}
                    {% if col != 'title' %}
                    <td>{{ result.get(col, '') }}</td>
                    {% endif %}
                    {% endfor %}
                </tr>
                {% endfor %}
            </table>
            {% endfor %}
        </body>
        </html>
        """

        template = Template(template_str)
        html_content = template.render(data=data, title_columns=title_columns)

        # 保存HTML报告
        html_path = "/tmp/test_report.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return html_path

    def send_email_report(
        self,
        recipient_email: Union[str, List[str]],
        current_results: List = None,
        Change_id=None,
    ):
        """发送邮件报告

        Args:
            recipient_email: 收件人邮箱，可以是单个邮箱地址或邮箱地址列表
        """
        self.change_id = Change_id
        html_path, markdown_path, email_content = self.generate_report(current_results)

        # 确保recipient_email是列表格式
        if isinstance(recipient_email, str):
            recipient_email = [recipient_email]

        # 使用yagmail发送邮件，将HTML和Markdown报告作为附件
        self.yag.send(
            to=recipient_email,  # yagmail支持直接传入列表
            subject="测试结果报告",
            contents=email_content,  # 确保内容使用UTF-8编码
            attachments=[html_path],  # 两个报告都作为附件 # test int8 fast
        )

        # 删除临时文件
        os.remove(html_path)
        os.remove(markdown_path)


# 定义 hook 标记
hookimpl = HookimplMarker("allure")


class AllureResultPlugin(AllureDeveloperHooks):
    """Allure 测试结果插件，同时处理结果收集和报告生成"""

    def __init__(self, config):
        global _allure_plugin
        self.config = config
        self.allure_results = {}
        self.result_manager = TestResultManager(
            db_path=os.path.join(ABS_PATH, os.getenv("RESULTS_DATABASE_PATH", "save/results_database.json"))
        )
        _allure_plugin = self  # 保存实例到全局变量

    @hookimpl
    def report_result(self, result: TestResult):
        """在每个测试用例结果写入 JSON 之前被调用"""
        if (result.description is None) and (len(result.attachments) == 0):  # 说明没有记录任何的信息,我就不去记录它了
            return

        uuid = result.uuid
        self.allure_results[uuid] = {
            "title": result.name,
            "description": result.description,
            "status": result.status,
            "timestamp": datetime.fromtimestamp(result.stop / 1000).strftime("%Y-%m-%d %H:%M:%S"),
        }
        if result.statusDetails is not None:
            self.allure_results[uuid]["statusDetails"] = str(result.statusDetails.message)

        # 收集附件
        # attachment.type in [
        #         AttachmentType.TEXT,
        #         "text/plain",
        #         "txt",
        #     ] and
        for attachment in result.attachments:
            if hasattr(attachment, "source"):
                try:
                    allure_results_dir = self.config.getoption("--alluredir", "allure-results")
                    with open(
                        os.path.join(allure_results_dir, attachment.source),
                        "r",
                        encoding="utf-8",
                    ) as f:
                        content = f.read().strip()
                        self.allure_results[uuid][attachment.name] = content
                except Exception as e:
                    print(f"无法读取附件 {attachment.name}: {e}")

    @pytest.hookimpl
    def pytest_sessionfinish(self, session, exitstatus):
        """测试会话结束时处理结果"""
        if not self.allure_results:
            print("\n没有收集到测试结果。")
            return

        # 转换当前结果为列表格式
        current_results = list(self.allure_results.values())

        # 存储结果
        if self.config.getoption("--store-results"):
            if current_results:
                self.result_manager.add_test_results(current_results)
                print(f"\n已存储 {len(current_results)} 条测试结果到数据库")
            else:
                print("\n没有测试结果可存储。")

        # 发送邮件
        if self.config.getoption("--send-email"):
            recipient_email = [email.strip() for email in os.getenv("RECIPIENT_EMAIL", "xing.hu@houmo.ai").split(",")]
            if recipient_email:
                # 只使用当前结果生成邮件内容
                self.result_manager.send_email_report(
                    recipient_email, current_results, self.config.getoption("--Change-ID")
                )
                print("\n邮件报告已发送")
            else:
                print("\n未设置收件人邮箱，无法发送邮件报告")


def pytest_addoption(parser):
    """添加命令行参数"""
    parser.addoption(
        "--store-results",
        action="store_true",
        default=False,
        help="是否存储测试结果到数据库",
    )
    parser.addoption("--send-email", action="store_true", default=False, help="是否发送邮件报告")
    parser.addoption("--Change-ID", default="without provide Change-ID", help="是否发送邮件报告")


def pytest_configure(config):
    """配置 pytest 插件和 Allure 拦截器"""
    global _allure_plugin

    # 检查是否需要启用 Allure
    if config.getoption("--store-results") or config.getoption("--send-email"):
        # 创建插件实例
        plugin = AllureResultPlugin(config)

        # 确保 Allure 插件管理器已初始化
        if not hasattr(plugin_manager, "_plugin_manager"):
            plugin_manager._plugin_manager = plugin_manager.get_plugin_manager()

        # 注册为 Allure 拦截器
        plugin_manager.register(plugin, name="allure_interceptor")

        # 注册为 pytest 插件
        config.pluginmanager.register(plugin, "allure_result_plugin")

        print("\n已注册 Allure 结果处理插件。")


def pytest_unconfigure(config):
    """清理插件"""
    global _allure_plugin
    if _allure_plugin:
        plugin_manager.unregister(_allure_plugin)
        _allure_plugin = None
