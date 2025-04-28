import flet as ft
from gui import main as gui_main
from logger_config import get_logger


# 旧的 async def main() 函数及其所有内容都已删除或移至 gui.py


if __name__ == "__main__":
    # 1. 配置日志 (在启动 Flet 之前完成)
    # Logging setup is handled within gui.py now (inside its main and upon import)
    # from logger_config import setup_logging # Removed

    # 2. (Optional) Early checks can go here if needed.

    # 3. 启动 Flet 应用程序
    # gui.py 中的 main 函数现在是 Flet 的目标
    # Flet 会处理自己的事件循环和窗口管理
    ft.app(target=gui_main)

    # Flet 应用关闭后，程序将在这里退出
    # 可以在这里添加最后的清理代码，但通常清理工作在 GUI 的关闭事件中处理
    final_logger = get_logger(__name__)  # 获取 logger 以记录最终消息
    final_logger.info("--- VRCMeow 已退出 ---")
