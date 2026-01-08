import subprocess
import sys
import pkg_resources

def check_and_set_transformers_version(target_version="4.40.0"):
    """
    检测transformers版本，不满足则临时安装指定版本，返回原始版本号（用于后续回滚）
    """
    # ========== 1. 获取原始版本号 ==========
    try:
        # 获取已安装的transformers版本
        original_version = pkg_resources.get_distribution("transformers").version
        print(f"当前transformers版本：{original_version}")
    except pkg_resources.DistributionNotFound:
        # 未安装transformers，标记为"未安装"
        original_version = "not_installed"
        print("未检测到transformers，将临时安装4.40.0版本")

    # ========== 2. 检测版本是否满足要求 ==========
    need_rollback = False  # 是否需要回滚的标记
    if original_version != target_version:
        need_rollback = True
        # 安装指定版本
        print(f"开始安装transformers=={target_version}...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", f"transformers=={target_version}", "--force-reinstall", "-i https://pypi.tuna.tsinghua.edu.cn/simple"],
            check=True,
            stdout=None,
            stderr=None,
            encoding="utf-8"
        )
        print(f"成功安装transformers=={target_version}")
    else:
        print(f"transformers版本已满足要求（{target_version}），无需安装")

    return original_version, need_rollback

def rollback_transformers_version(original_version):
    """
    回滚transformers到原始版本
    """
    if original_version == "not_installed":
        # 原始状态是未安装，直接卸载
        print("开始卸载transformers...")
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "transformers", "-y"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8"
        )
        print("成功卸载transformers")
    else:
        # 回滚到原始版本
        print(f"开始回滚transformers到原始版本：{original_version}...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", f"transformers=={original_version}", "--force-reinstall", "-i https://pypi.tuna.tsinghua.edu.cn/simple"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8"
        )
        print(f"成功回滚到transformers=={original_version}")

# ========== 核心业务逻辑（你的代码） ==========
def your_business_code():
    """
    这里放你需要运行的核心代码（依赖transformers==4.40.0）
    """
    try:
        from transformers.models.qwen2_vl.modeling_qwen2_vl import VisionSdpaAttention
        print("✅ 成功导入VisionSdpaAttention，开始执行核心逻辑...")
        # 你的核心代码写在这里
        # --------------------------
        # 示例：验证版本
        import transformers
        print(f"当前运行版本：{transformers.__version__}")
        # --------------------------
        print("✅ 核心逻辑执行完成")
    except Exception as e:
        print(f"❌ 核心逻辑执行失败：{e}")
        raise  # 抛出异常，确保回滚仍会执行

# ========== 主流程 ==========
if __name__ == "__main__":
    original_version = None
    need_rollback = False
    try:
        # 1. 检测并安装指定版本
        original_version, need_rollback = check_and_set_transformers_version(target_version="4.40.0")
        
        # 2. 执行核心业务代码
        your_business_code()

    except Exception as e:
        print(f"程序执行出错：{e}")
    finally:
        # 3. 无论是否出错，都回滚版本（关键：finally确保必执行）
        if need_rollback:
            try:
                rollback_transformers_version(original_version)
            except Exception as e:
                print(f"⚠️ 版本回滚失败：{e}")
        else:
            print("无需回滚版本")
    
    print("✅ 整个流程执行完毕")