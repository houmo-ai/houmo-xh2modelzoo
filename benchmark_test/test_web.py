import imaplib
import smtplib
import email
from email.mime.text import MIMEText
import time
import psutil
import schedule
import re
import os
from dotenv import load_dotenv
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, redirect, url_for
import threading
import logging
from logging.handlers import RotatingFileHandler
import sys

# 加载环境变量
load_dotenv()

# ========== 全局配置 ==========
# 邮箱配置
EMAIL_ACCOUNT = os.getenv("EMAIL_ACCOUNT", "metroplex@houmo.ai")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "WTRW2pPpZyVTXGag")
IMAP_SERVER = os.getenv("IMAP_SERVER", "imap.feishu.cn")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.feishu.cn")
SMTP_PORT = int(os.getenv("SMTP_PORT", 465))

# 测试命令配置
TEST_COMMAND = "pytest benchmark_test_openpose.py --send-email"
TEST_ALL_COMMAND = "pytest --send-email"
TEST_KEYWORD = "test"
TEST_PRIVATE_KEYWORD = "private"

ABS_PATH = os.path.dirname(__file__)

# Flask 配置
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "test_service_2026")

# 全局状态变量
APP_STATE = {
    "is_running": True,
    "last_check_time": None,
    "test_running": False,
    "log_content": [],
    "current_test_command": TEST_COMMAND  # 新增：记录当前执行的测试命令
}

# 新增：记录当前测试进程PID
CURRENT_TEST_PID = None

# ========== 日志配置 ==========
def setup_logger():
    """配置日志系统，同时输出到文件和内存"""
    # 关闭Flask默认的werkzeug访问日志
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)  # 只保留ERROR级别以上的werkzeug日志
    
    # 创建日志处理器
    log_handler = RotatingFileHandler(
        'test_service.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_handler.setFormatter(log_formatter)
    
    # 配置根日志
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 先清空已有处理器，避免重复输出
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    logger.addHandler(log_handler)
    
    # 自定义内存日志处理器
    class MemoryLogHandler(logging.Handler):
        def emit(self, record):
            log_entry = self.format(record)
            APP_STATE["log_content"].append(log_entry)
            # 只保留最新的500条日志
            if len(APP_STATE["log_content"]) > 500:
                APP_STATE["log_content"] = APP_STATE["log_content"][-500:]
    
    memory_handler = MemoryLogHandler()
    memory_handler.setFormatter(log_formatter)
    logger.addHandler(memory_handler)
    
    return logger

logger = setup_logger()

# ========== 原有核心功能 ==========
def is_test_running():
    """检查pytest是否正在运行（修复：支持自定义命令检测）"""
    global CURRENT_TEST_PID
    try:
        # 优先检查记录的PID
        if CURRENT_TEST_PID:
            try:
                proc = psutil.Process(CURRENT_TEST_PID)
                if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                    # 检查进程是否是pytest进程（放宽检测条件）
                    cmdline = ' '.join(proc.cmdline()).lower()
                    if 'pytest' in cmdline:
                        APP_STATE["test_running"] = True
                        return True
                else:
                    # PID已终止，重置
                    CURRENT_TEST_PID = None
            except psutil.NoSuchProcess:
                CURRENT_TEST_PID = None

        # 遍历所有进程检测pytest
        for proc in psutil.process_iter(['pid', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and isinstance(cmdline, list):
                    full_cmd = ' '.join(cmdline).lower()
                    # 检测是否包含pytest（兼容自定义命令）
                    if 'pytest' in full_cmd and '--send-email' in full_cmd:
                        APP_STATE["test_running"] = True
                        CURRENT_TEST_PID = proc.info['pid']  # 更新PID
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # 无运行中的测试进程
        APP_STATE["test_running"] = False
        CURRENT_TEST_PID = None
        return False
    except Exception as e:
        logger.error(f"检查测试运行状态失败: {str(e)}")
        APP_STATE["test_running"] = False
        CURRENT_TEST_PID = None
        return False

def send_reply_email(recipient, subject, content):
    """发送自动回复邮件"""
    try:
        msg = MIMEText(content, 'plain', 'utf-8')
        msg['From'] = EMAIL_ACCOUNT
        msg['To'] = recipient
        msg['Subject'] = subject

        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(EMAIL_ACCOUNT, EMAIL_PASSWORD)
            server.send_message(msg)
        
        logger.info(f"自动回复已发送至 {recipient}")
        return True
    except Exception as e:
        logger.error(f"发送邮件失败: {str(e)}")
        return False

def run_test_command(custom_command=None):
    """执行测试命令（支持自定义命令）"""
    command = custom_command or TEST_COMMAND
    logger.info(f"开始执行测试命令: {command}")
    
    # 执行前主动标记为运行中
    APP_STATE["test_running"] = True
    APP_STATE["current_test_command"] = command  # 记录当前命令

    try:
        import subprocess
        result = subprocess.run(
            command.split(),
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=3600
        )
        
        if result.returncode == 0:
            logger.info(f"测试命令执行成功: {result.stdout}")
        else:
            logger.error(f"测试命令执行失败: {result.stderr}")
            
    except Exception as e:
        logger.error(f"执行测试命令出错: {str(e)}")
    
    finally:
        # 无论成功/失败/异常，执行结束后都重置状态
        APP_STATE["test_running"] = False
        logger.info(f"测试命令执行完成: {command}")

def check_new_emails():
    """检查新邮件并处理"""
    if not APP_STATE["is_running"]:
        return
    
    APP_STATE["last_check_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_ACCOUNT, EMAIL_PASSWORD)
        mail.select('INBOX')

        status, data = mail.search(None, 'UNSEEN')
        if status != 'OK':
            mail.close()
            mail.logout()
            return

        email_ids = data[0].split()
        if not email_ids:
            mail.close()
            mail.logout()
            return

        for email_id in email_ids:
            status, data = mail.fetch(email_id, '(RFC822)')
            if status != 'OK':
                continue

            msg = email.message_from_bytes(data[0][1])
            sender = msg.get('From', '')
            sender_email = re.search(r'<([^>]+)>', sender)
            sender_email = sender_email.group(1) if sender_email else sender

            # 读取邮件正文
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type == 'text/plain' and not part.get_filename():
                        body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        break
            else:
                body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')

            # 检查邮件内容是否包含test关键词
            if TEST_KEYWORD.lower() in body.lower():
                if is_test_running():
                    send_reply_email(
                        sender_email,
                        "测试状态通知",
                        "当前已在执行测试任务，请稍后再试。"
                    )
                else:
                    # 私密性测试
                    if TEST_PRIVATE_KEYWORD.lower() in body.lower():
                        global TEST_COMMAND
                        TEST_COMMAND = TEST_COMMAND + f" --send-email-dress {sender_email}"
                        logger.info(f"更新测试命令为: {TEST_COMMAND}")

                    send_reply_email(
                        sender_email,
                        "测试启动通知",
                        f"已收到测试请求，即将开始执行测试任务{TEST_COMMAND}。"
                    )

                    # 异步执行测试命令
                    threading.Thread(target=run_test_command).start()

            # 将邮件标记为已读
            mail.store(email_id, '+FLAGS', '\\Seen')

        mail.close()
        mail.logout()
        logger.info(f"邮件检查完成，处理了 {len(email_ids)} 封新邮件")
        
    except Exception as e:
        logger.error(f"检查邮件出错: {str(e)}")

def friday_night_test():
    """每周五凌晨12点执行的测试任务"""
    logger.info("触发周五定时测试任务")
    if not is_test_running():
        global TEST_ALL_COMMAND
        run_test_command(TEST_ALL_COMMAND)
    else:
        logger.info("周五定时任务：测试已在运行中")

def background_tasks():
    """后台任务线程：邮件监听 + 定时任务"""
    # 配置定时任务
    schedule.every().friday.at("23:00").do(friday_night_test)
    logger.info("周五定时测试任务已配置（每周五23:00执行）")
    
    # 主循环
    while APP_STATE["is_running"]:
        try:
            check_new_emails()
            schedule.run_pending()
            time.sleep(5)
        except Exception as e:
            logger.error(f"后台任务出错：{str(e)}")
            time.sleep(60)
    
    logger.info("后台任务已停止")

# ========== Flask Web 路由 ==========
# 直接使用 render_template_string 替代模板文件，避免目录问题
@app.route('/')
def index():
    """首页：展示状态和操作界面（修复HTML结构）"""
    is_test_running()  # 更新测试运行状态
    
    # 获取当前目录下以benchmark开头的文件列表
    benchmark_files = []
    try:
        # 遍历当前目录
        for file in os.listdir('.'):
            if file.startswith('benchmark') and os.path.isfile(file):
                benchmark_files.append(file)
        benchmark_files.sort()  # 按名称排序
    except Exception as e:
        logger.error(f"获取benchmark文件列表失败: {str(e)}")

    # HTML模板内容（修复form ID和按钮选择器）
    html_content = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>测试任务管理系统</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Arial', sans-serif;
        }
        body {
            background-color: #f5f7fa;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 20px;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }
        .status-card {
            background: #e8f4f8;
            border-radius: 6px;
            padding: 15px;
            margin-bottom: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .status-item {
            display: flex;
            flex-direction: column;
        }
        .status-label {
            font-size: 14px;
            color: #666;
            margin-bottom: 5px;
        }
        .status-value {
            font-size: 16px;
            font-weight: bold;
            color: #333;
        }
        .status-value.running {
            color: #28a745;
        }
        .status-value.stopped {
            color: #dc3545;
        }
        .section {
            margin-bottom: 30px;
        }
        .section-title {
            font-size: 18px;
            margin-bottom: 15px;
            color: #2c3e50;
            font-weight: 600;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 500;
            color: #444;
        }
        input, textarea, button {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }
        textarea {
            min-height: 80px;
            resize: vertical;
        }
        .btn {
            background: #007bff;
            color: white;
            border: none;
            cursor: pointer;
            font-weight: 500;
            transition: background 0.3s;
        }
        .btn:hover {
            background: #0056b3;
        }
        .btn-danger {
            background: #dc3545;
        }
        .btn-danger:hover {
            background: #c82333;
        }
        .btn-success {
            background: #28a745;
        }
        .btn-success:hover {
            background: #218838;
        }
        .log-container {
            background: #f8f9fa;
            border: 1px solid #eee;
            border-radius: 6px;
            padding: 15px;
            height: 300px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.5;
        }
        .log-line {
            margin-bottom: 2px;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        @media (max-width: 768px) {
            .grid {
                grid-template-columns: 1fr;
            }
        }
        .alert {
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 15px;
            display: none;
        }
        .alert-success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
            display: block;
        }
        .alert-error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
            display: block;
        }
        /* 拖拽相关样式 */
        #file-list .file-item:hover {
            background: #007bff;
            color: white;
            transition: background 0.2s;
        }
        #file-list .file-item:active {
            background: #0056b3;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>测试任务管理系统</h1>
            <span>当前邮箱：{{ email_account }}</span>
        </div>

        <!-- 状态卡片 -->
        <div class="status-card">
            <div class="status-item">
                <span class="status-label">服务状态</span>
                <span class="status-value {{ 'running' if app_state.is_running else 'stopped' }}">
                    {{ '运行中' if app_state.is_running else '已停止' }}
                </span>
            </div>
            <div class="status-item">
                <span class="status-label">测试任务状态</span>
                <span class="status-value {{ 'running' if app_state.test_running else 'stopped' }}">
                    {{ '运行中' if app_state.test_running else '未运行' }}
                </span>
            </div>
            <div class="status-item">
                <span class="status-label">最后邮件检查时间</span>
                <span class="status-value">{{ app_state.last_check_time or '未检查' }}</span>
            </div>
        </div>

        <!-- 手动触发测试（修复：添加form ID） -->
        <div class="section">
            <div class="section-title">手动触发测试</div>
            <div id="alert-container"></div>
            <form id="trigger-form">
                <div style="display: flex; gap: 15px; margin-bottom: 15px;">
                    <!-- 测试命令输入框 -->
                    <div style="flex: 1;">
                        <div class="form-group">
                            <label for="custom_command">测试命令（可选自定义）</label>
                            <textarea id="custom_command" name="custom_command">{{ test_command }}</textarea>
                        </div>
                    </div>
                    <!-- 可拖动的文件列表 -->
                    <div style="width: 300px; border: 1px solid #ddd; border-radius: 4px; margin-top: -40px;">
                        <div style="padding: 10px; background: #f8f9fa; border-bottom: 1px solid #ddd; font-weight: 500;">
                            可选测试文件（拖动选择）
                        </div>
                        <div id="file-list" style="height: 120px; overflow-y: auto; padding: 10px;">
                            {% if benchmark_files %}
                                {% for file in benchmark_files %}
                                <!-- 新增onclick事件，点击直接替换命令 -->
                                <div class="file-item" data-file="{{ file }}" 
                                    style="padding: 5px; margin-bottom: 5px; border-radius: 3px; background: #e9ecef; cursor: pointer;"
                                    onclick="replaceCommandWithFile('{{ file }}')">
                                    {{ file }}
                                </div>
                                {% endfor %}
                            {% else %}
                                <div style="color: #666; text-align: center; padding: 10px;">暂无benchmark开头的文件</div>
                            {% endif %}
                        </div>
                    </div>
                </div>
                
                <button type="submit" class="btn btn-success" id="test-trigger-btn" {{ 'disabled' if app_state.test_running else '' }}>
                    {{ '测试中...' if app_state.test_running else '立即执行测试' }}
                </button>
            </form>
        </div>

        <!-- 配置管理 -->
        <div class="section">
            <div class="section-title">系统配置</div>
            <form id="config-form" method="POST" action="/update-config">
                <div class="grid">
                    <div>
                        <div class="form-group">
                            <label for="test_command">默认测试命令</label>
                            <input type="text" id="test_command" name="test_command" value="{{ test_command }}">
                        </div>
                        <div class="form-group">
                            <label for="test_keyword">邮件触发关键词</label>
                            <input type="text" id="test_keyword" name="test_keyword" value="{{ test_keyword }}">
                        </div>
                    </div>
                    <div>
                        <div class="form-group">
                            <label for="private_keyword">私密测试关键词</label>
                            <input type="text" id="private_keyword" name="private_keyword" value="{{ private_keyword }}">
                        </div>
                        <div class="form-group">
                            <label for="email_account">监听邮箱</label>
                            <input type="text" id="email_account" value="{{ email_account }}" disabled>
                        </div>
                    </div>
                </div>
                <button type="submit" class="btn">保存配置</button>
            </form>
        </div>

        <!-- 运行日志 -->
        <div class="section">
            <div class="section-title">运行日志</div>
            <div class="log-container" id="log-container">
                {% for log in app_state.log_content %}
                <div class="log-line">{{ log }}</div>
                {% endfor %}
            </div>
            <button class="btn" style="margin-top: 10px;" onclick="refreshLogs()">刷新日志</button>
        </div>

        <!-- 系统操作 -->
        <div class="section">
            <div class="section-title">系统操作</div>
            <button class="btn btn-danger" id="stop-btn">停止服务</button>
        </div>

        <!-- 新增：邮件报告预览 -->
        <div class="section">
            <div class="section-title">邮件报告预览 (email_report.html)</div>
            <div style="display: flex; gap: 10px; margin-bottom: 10px;">
                <button class="btn" style="width: auto;" onclick="refreshEmailReport()">刷新报告</button>
                <button class="btn" style="width: auto; background: #6c757d;" onclick="clearEmailReport()">清空内容</button>
            </div>
            <div class="log-container" id="email-report-container" style="height: 400px;">
                <div class="log-line" style="text-align: center; color: #666;">点击"刷新报告"加载 email_report.html 内容</div>
            </div>
        </div>
    </div>

<script>
    // 显示提示框
    function showAlert(message, isError = false) {
        const alertContainer = document.getElementById('alert-container');
        const alertClass = isError ? 'alert-error' : 'alert-success';
        alertContainer.innerHTML = `<div class="alert ${alertClass}">${message}</div>`;
        
        // 3秒后隐藏
        setTimeout(() => {
            alertContainer.innerHTML = '';
        }, 3000);
    }

    // 刷新页面状态（修复：按钮选择器）
    async function refreshPageState() {
        try {
            const response = await fetch('/get-state');
            const state = await response.json();
            
            // 更新测试状态显示
            const testStatusEl = document.querySelector('.status-card .status-item:nth-child(2) .status-value');
            testStatusEl.className = 'status-value ' + (state.test_running ? 'running' : 'stopped');
            testStatusEl.textContent = state.test_running ? '运行中' : '未运行';
            
            // 更新服务状态显示
            const serviceStatusEl = document.querySelector('.status-card .status-item:nth-child(1) .status-value');
            serviceStatusEl.className = 'status-value ' + (state.is_running ? 'running' : 'stopped');
            serviceStatusEl.textContent = state.is_running ? '运行中' : '已停止';
            
            // 更新最后检查时间
            const checkTimeEl = document.querySelector('.status-card .status-item:nth-child(3) .status-value');
            checkTimeEl.textContent = state.last_check_time || '未检查';
            
            // 更新按钮状态（修复：使用ID选择器）
            const triggerBtn = document.getElementById('test-trigger-btn');
            if (state.test_running) {
                triggerBtn.disabled = true;
                triggerBtn.textContent = '测试中...';
            } else {
                triggerBtn.disabled = false;
                triggerBtn.textContent = '立即执行测试';
            }
            
        } catch (error) {
            console.error('刷新状态失败:', error);
        }
    }

    // 手动触发测试（修复：事件绑定）
    document.getElementById('trigger-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = document.getElementById('test-trigger-btn');
        const customCommand = document.getElementById('custom_command').value;
        
        // 禁用按钮并提示
        btn.disabled = true;
        btn.textContent = '执行中...';
        
        try {
            const response = await fetch('/trigger-test', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `custom_command=${encodeURIComponent(customCommand)}`
            });
            
            const result = await response.json();
            showAlert(result.message, result.status === 'error');
            
        } catch (error) {
            showAlert('请求失败：' + error.message, true);
        } finally {
            // 立即刷新状态
            setTimeout(() => {
                refreshPageState();
            }, 500);
        }
    });

    // 停止服务
    document.getElementById('stop-btn').addEventListener('click', async () => {
        if (!confirm('确定要停止服务吗？停止后将不再监听邮件和执行定时任务！')) {
            return;
        }
        
        try {
            const response = await fetch('/stop-service', {
                method: 'POST'
            });
            
            const result = await response.json();
            showAlert(result.message);
            
            // 5秒后刷新
            setTimeout(() => window.location.reload(), 5000);
            
        } catch (error) {
            showAlert('停止服务失败：' + error.message, true);
        }
    });

    // 刷新日志
    async function refreshLogs() {
        try {
            const response = await fetch('/get-logs');
            const result = await response.json();
            
            const logContainer = document.getElementById('log-container');
            logContainer.innerHTML = '';
            
            result.logs.forEach(log => {
                const logLine = document.createElement('div');
                logLine.className = 'log-line';
                logLine.textContent = log;
                logContainer.appendChild(logLine);
            });
            
            // 滚动到底部
            logContainer.scrollTop = logContainer.scrollHeight;
            
        } catch (error) {
            showAlert('刷新日志失败：' + error.message, true);
        }
    }

    // 自动刷新日志和状态（每2秒）
    setInterval(() => {
        refreshLogs();
        refreshPageState();
        // 可选：自动刷新报告（注释掉则仅手动刷新）
        // refreshEmailReport();
    }, 2000);
    
    // 点击文件项直接替换测试命令
    function replaceCommandWithFile(fileName) {
        const commandInput = document.getElementById('custom_command');
        const currentValue = commandInput.value;
        
        // 核心逻辑：替换命令中的benchmark开头的.py文件
        const regex = /benchmark[^ ]+\.py/;
        let newCommand = '';
        
        if (regex.test(currentValue)) {
            // 替换已有的benchmark文件名
            newCommand = currentValue.replace(regex, fileName);
        } else {
            // 无匹配则追加到命令末尾
            newCommand = currentValue.trim() + ' ' + fileName;
        }
        
        // 更新输入框内容
        commandInput.value = newCommand;
        // 显示提示
        showAlert(`已选择文件：${fileName}`, false);
    }

    // 拖拽逻辑（保留）
    let draggedFile = null;
    function handleDragStart(e) {
        if (e.target.classList.contains('file-item')) {
            draggedFile = e.target.dataset.file;
            e.dataTransfer.setData('text/plain', draggedFile);
            e.target.style.opacity = '0.5';
        } else {
            e.preventDefault();
        }
    }
    document.addEventListener('dragend', function(e) {
        if (e.target.classList.contains('file-item')) {
            e.target.style.opacity = '1';
        }
    });
    document.getElementById('custom_command').addEventListener('dragover', function(e) {
        e.preventDefault();
        this.style.border = '1px solid #007bff';
    });
    document.getElementById('custom_command').addEventListener('dragleave', function(e) {
        this.style.border = '1px solid #ddd';
    });
    document.getElementById('custom_command').addEventListener('drop', function(e) {
        e.preventDefault();
        this.style.border = '1px solid #ddd';
        const fileName = e.dataTransfer.getData('text/plain');
        if (fileName) {
            replaceCommandWithFile(fileName);
        }
    });

    // 新增：刷新邮件报告
    async function refreshEmailReport() {
        try {
            const response = await fetch('/get-email-report');
            const result = await response.json();
            
            const reportContainer = document.getElementById('email-report-container');
            if (result.status === 'success') {
                // 直接渲染HTML内容
                reportContainer.innerHTML = result.content;
                // 修复样式冲突（可选）
                reportContainer.style.fontFamily = 'Arial, sans-serif';
                reportContainer.style.padding = '15px';
            } else {
                reportContainer.innerHTML = `<div class="log-line" style="color: #dc3545;">${result.content}</div>`;
            }
            
            // 滚动到底部
            reportContainer.scrollTop = reportContainer.scrollHeight;
            
        } catch (error) {
            showAlert('刷新邮件报告失败：' + error.message, true);
            document.getElementById('email-report-container').innerHTML = 
                `<div class="log-line" style="color: #dc3545;">刷新失败：${error.message}</div>`;
        }
    }

    // 新增：清空邮件报告
    function clearEmailReport() {
        const reportContainer = document.getElementById('email-report-container');
        reportContainer.innerHTML = '<div class="log-line" style="text-align: center; color: #666;">内容已清空</div>';
    }
    

    // 页面加载完成后初始化
    window.onload = () => {
        refreshLogs();
        refreshPageState();
        // 新增：页面加载时自动刷新报告
        refreshEmailReport();
    };
</script>

</body>
</html>
    '''
    
    return render_template_string(
        html_content,
        app_state=APP_STATE,
        test_command=TEST_COMMAND,
        test_keyword=TEST_KEYWORD,
        private_keyword=TEST_PRIVATE_KEYWORD,
        email_account=EMAIL_ACCOUNT,
        benchmark_files=benchmark_files,
    )

@app.route('/get-state')
def get_state():
    """获取当前系统状态（供前端实时刷新）"""
    is_test_running()  # 确保状态是最新的
    return jsonify({
        "is_running": APP_STATE["is_running"],
        "test_running": APP_STATE["test_running"],
        "last_check_time": APP_STATE["last_check_time"]
    })

@app.route('/trigger-test', methods=['POST'])
def trigger_test():
    """手动触发测试（优化：更健壮的进程管理）"""
    if is_test_running():
        return jsonify({"status": "error", "message": "测试任务正在运行中！"})
    
    # 获取自定义命令（如果有）
    custom_command = request.form.get('custom_command', TEST_COMMAND).strip()
    if not custom_command:
        custom_command = TEST_COMMAND
    
    # 异步执行测试命令，并捕获进程PID
    def run_and_record_pid():
        global CURRENT_TEST_PID
        APP_STATE["test_running"] = True  # 立即标记为运行中
        try:
            import subprocess
            # 改用Popen启动，记录PID
            proc = subprocess.Popen(
                custom_command.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            CURRENT_TEST_PID = proc.pid  # 记录PID
            logger.info(f"测试进程已启动，PID: {proc.pid}")
            
            # 等待进程结束
            stdout, stderr = proc.communicate(timeout=3600)
            
            if proc.returncode == 0:
                logger.info(f"测试命令执行成功: {stdout}")
                # 可选：发送成功通知
            else:
                logger.error(f"测试命令执行失败: {stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error(f"测试命令执行超时（3600秒）")
            # 终止超时进程
            if CURRENT_TEST_PID:
                try:
                    proc = psutil.Process(CURRENT_TEST_PID)
                    proc.terminate()
                except:
                    pass
        except Exception as e:
            logger.error(f"执行测试命令出错: {str(e)}")
        finally:
            APP_STATE["test_running"] = False
            CURRENT_TEST_PID = None
            logger.info(f"测试命令执行完成: {custom_command}")

    # 启动线程执行
    threading.Thread(target=run_and_record_pid).start()
    logger.info(f"手动触发测试，命令：{custom_command}")
    
    return jsonify({"status": "success", "message": "测试任务已启动！"})

@app.route('/update-config', methods=['POST'])
def update_config():
    """更新配置参数"""
    global TEST_COMMAND, TEST_KEYWORD, TEST_PRIVATE_KEYWORD
    
    # 获取表单数据
    TEST_COMMAND = request.form.get('test_command', TEST_COMMAND).strip()
    TEST_KEYWORD = request.form.get('test_keyword', TEST_KEYWORD).strip()
    TEST_PRIVATE_KEYWORD = request.form.get('private_keyword', TEST_PRIVATE_KEYWORD).strip()
    
    logger.info(f"配置已更新 - 测试命令：{TEST_COMMAND} | 关键词：{TEST_KEYWORD} | 私密关键词：{TEST_PRIVATE_KEYWORD}")
    
    return redirect(url_for('index'))

@app.route('/get-logs')
def get_logs():
    """获取最新日志"""
    return jsonify({"logs": APP_STATE["log_content"]})

@app.route('/get-email-report')
def get_email_report():
    """读取 email_report.html 内容并返回"""
    try:
        # 这里要和你之前定义的 EMAIL_HTML_REPORT_PATH 保持一致
        report_path = os.path.join(ABS_PATH, "email_report.html") # if 'ABS_PATH' in globals() else "email_report.html"
        
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return jsonify({"status": "success", "content": content})
        else:
            return jsonify({"status": "error", "content": "email_report.html 文件不存在，请先执行测试生成报告"})
    except Exception as e:
        logger.error(f"读取邮件报告失败: {str(e)}")
        return jsonify({"status": "error", "content": f"读取报告失败：{str(e)}"})

@app.route('/stop-service', methods=['POST'])
def stop_service():
    """停止服务"""
    APP_STATE["is_running"] = False
    logger.info("服务已停止（通过Web界面）")
    return jsonify({"status": "success", "message": "服务将在5秒内停止！"})

# ========== 程序入口 ==========
def start_web_server():
    """启动Flask Web服务器（非阻塞）"""
    from werkzeug.serving import make_server
    
    # 再次确认关闭werkzeug日志
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    
    # 创建服务器实例
    server = make_server(
        host='0.0.0.0',
        port=5000,
        app=app,
        threaded=True
    )
    
    logger.info(f"Web服务已启动，访问地址：http://0.0.0.0:5000")
    
    # 启动服务器（非阻塞）
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    
    return server, server_thread

def main():
    """主函数：启动后台任务和Web服务"""
    logger.info(f"测试服务启动，监听邮箱：{EMAIL_ACCOUNT}")
    logger.info("="*50)
    
    # 启动后台任务线程（邮件监听+定时任务）
    bg_thread = threading.Thread(target=background_tasks, daemon=False)
    bg_thread.start()
    
    # 启动Web服务器（非阻塞）
    server, server_thread = start_web_server()
    
    # 保持主线程运行
    try:
        while APP_STATE["is_running"]:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("接收到停止信号，正在关闭服务...")
        APP_STATE["is_running"] = False
        
        # 停止Web服务器
        server.shutdown()
        server_thread.join()
        
        # 等待后台线程结束
        bg_thread.join(timeout=10)
        
        logger.info("服务已完全停止")
        sys.exit(0)

if __name__ == "__main__":
    main()