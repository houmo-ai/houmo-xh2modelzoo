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

# 加载环境变量（建议将敏感信息放在.env文件中）
load_dotenv()

# 邮箱配置（请替换为你的实际配置，或在.env文件中配置）
EMAIL_ACCOUNT = os.getenv("EMAIL_ACCOUNT", "metroplex@houmo.ai")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "WTRW2pPpZyVTXGag")
IMAP_SERVER = os.getenv("IMAP_SERVER", "imap.feishu.cn")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.feishu.cn")
SMTP_PORT = int(os.getenv("SMTP_PORT", 465))  # SSL端口

# 测试命令配置
TEST_COMMAND = "pytest benchmark_test_openpose.py --send-email"
TEST_KEYWORD = "test"
TEST_PRIVATE_KEYWORD = "private"

ABS_PATH = os.path.dirname(__file__)

def is_test_running():
    """检查pytest --send-email是否正在运行"""
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and isinstance(cmdline, list):
                # 拼接命令行参数，检查是否包含目标命令
                full_cmd = ' '.join(cmdline).lower()
                if TEST_COMMAND.lower() in full_cmd:
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False

def send_reply_email(recipient, subject, content):
    """发送自动回复邮件"""
    msg = MIMEText(content, 'plain', 'utf-8')
    msg['From'] = EMAIL_ACCOUNT
    msg['To'] = recipient
    msg['Subject'] = subject

    try:
        # 连接SMTP服务器并发送邮件
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(EMAIL_ACCOUNT, EMAIL_PASSWORD)
            server.send_message(msg)
        print(f"[{datetime.now()}] 自动回复已发送至 {recipient}")
        return True
    except Exception as e:
        print(f"[{datetime.now()}] 发送邮件失败: {str(e)}")
        return False

def run_test_command():
    """执行测试命令"""
    try:
        import subprocess
        # 执行测试命令，捕获输出
        print(TEST_COMMAND)
        result = subprocess.run(
            TEST_COMMAND.split(),
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=3600  # 设置超时时间1小时
        )
        if result.returncode == 0:
            print(f"[{datetime.now()}] 测试命令执行成功: {result.stdout}")
        else:
            print(f"[{datetime.now()}] 测试命令执行失败: {result.stderr}")
    except Exception as e:
        print(f"[{datetime.now()}] 执行测试命令出错: {str(e)}")

def check_new_emails():
    """检查新邮件并处理"""
    global TEST_COMMAND

    # try:
    # 连接IMAP服务器
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL_ACCOUNT, EMAIL_PASSWORD)
    mail.select('INBOX')  # 选择收件箱

    # 搜索未读邮件
    status, data = mail.search(None, 'UNSEEN')
    print(status, data)
    if status != 'OK':
        mail.close()
        mail.logout()
        return

    email_ids = data[0].split()
    print(email_ids)
    if not email_ids:
        mail.close()
        mail.logout()
        return

    for email_id in email_ids:
        # 获取邮件内容
        status, data = mail.fetch(email_id, '(RFC822)')
        if status != 'OK':
            continue

        # 解析邮件
        msg = email.message_from_bytes(data[0][1])
        sender = msg.get('From', '')
        # 提取发件人邮箱地址
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
            # 检查测试命令是否正在运行
            if is_test_running():
                # 回复已在测试中
                send_reply_email(
                    sender_email,
                    "测试状态通知",
                    "当前已在执行测试任务，请稍后再试。"
                )
            else:
                # 私密性测试
                if TEST_PRIVATE_KEYWORD.lower() in body.lower():
                    TEST_COMMAND = TEST_COMMAND + f" --send-email-dress {sender_email}"

                # 回复开始测试并执行命令
                send_reply_email(
                    sender_email,
                    "测试启动通知",
                    f"已收到测试请求，即将开始执行测试任务{TEST_COMMAND}。"
                )

                # 异步执行测试命令（避免阻塞邮件监听）
                import threading
                threading.Thread(target=run_test_command).start()

        # 将邮件标记为已读（可选）
        mail.store(email_id, '+FLAGS', '\\Seen')

    mail.close()
    mail.logout()

    # except Exception as e:
    #     print(f"[{datetime.now()}] 检查邮件出错: {str(e)}")

def friday_night_test():
    """每周五凌晨12点执行的测试任务"""
    print(f"[{datetime.now()}] 触发周五定时测试任务")
    if not is_test_running():
        run_test_command()
    else:
        print(f"[{datetime.now()}] 周五定时任务：测试已在运行中")

def main():
    """主函数：启动监听和定时任务"""
    # 配置定时任务：每周五（friday）凌晨0点0分
    schedule.every().friday.at("00:00").do(friday_night_test)

    print(f"[{datetime.now()}] 邮件监听服务已启动，监听邮箱：{EMAIL_ACCOUNT}")
    print(f"[{datetime.now()}] 周五定时测试任务已配置（每周五00:00执行）")

    # 主循环
    while True:
        try:
            # 检查新邮件（每30秒检查一次，可根据需要调整）
            check_new_emails()
            # 运行定时任务
            schedule.run_pending()
            # # 休眠30秒
            time.sleep(5)
        except KeyboardInterrupt:
            print(f"\n[{datetime.now()}] 服务已手动停止")
            break
        except Exception as e:
            print(f"[{datetime.now()}] 主循环出错：{str(e)}")
            time.sleep(60)  # 出错后休眠1分钟再重试

if __name__ == "__main__":
    main()