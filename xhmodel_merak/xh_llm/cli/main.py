import os
import sys


def cli_main() -> None:
    # ... 保持你的 import 不变 ...
    from xhmodel_merak.xh_llm.cli.llm_export import LLMExportCommand
    from xhquant.api import get_xhquant_logger, xhquant_init

    route_mapping: dict[str, type] = {
        "export": LLMExportCommand,
    }

    debug = bool(os.environ.get("DEBUG", 0))
    xhquant_init(debug=debug)
    logger = get_xhquant_logger()

    # 辅助函数：打印帮助信息
    def print_help():
        commands = []
        for k, v in route_mapping.items():
            # 获取类定义的 doc_string，如果没有则留空
            doc = getattr(v, "doc_string", "No description available")
            commands.append(f"\t{k:12}: {doc}")

        help_msg = (
            "\nUsage: hmquant2 <command> [arguments]\n\n"
            "Available Commands:\n"
            + "\n".join(commands)
            + "\n\nUse 'hmquant2 <command> --help' for more information on a specific command."
        )
        logger.info(help_msg)

    if len(sys.argv) == 1:
        print_help()
        return

    argv = sys.argv[1:]
    first_arg = argv[0].lower()

    # 1. 处理全局帮助命令：支持 hmquant2 --help, -h, help
    if first_arg in ["--help", "-h", "help", "command", "commands"]:
        print_help()
        return

    method_name = first_arg.replace("-", "_")

    # 2. 检查子命令是否存在
    if method_name not in route_mapping:
        logger.error(f"Unknown command: '{first_arg}'")
        print_help()
        return

    # 3. 实例化并运行子命令
    # 注意：大部分 Command 类（如基于 argparse 的）会自动处理 argv[1:] 中的 --help
    try:
        command = route_mapping[method_name](argv[1:])
        command.run()
    except Exception as e:
        logger.error(f"Error running command '{method_name}': {e}")


if __name__ == "__main__":
    cli_main()
