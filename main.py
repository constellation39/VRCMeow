import flet as ft
from gui import main as gui_main
from logger_config import get_logger


# 旧的 async def main() 函数及其所有内容都已删除或移至 gui.py


if __name__ == "__main__":
    # 1. 配置日志 (在启动 Flet 之前完成)
    # setup_logging() 在 gui.py 中调用，这里不再需要
    # logger = get_logger(__name__) # 获取 logger 的操作也移至 gui.py 或其他需要的地方

    # 2. (可选) 可以在这里执行一些非常早期的、与 UI 无关的检查，
    #    例如检查 Python 版本或关键文件是否存在，但大多数初始化应在 GUI 内部完成。

    # 3. 启动 Flet 应用程序
    # gui.py 中的 main 函数现在是 Flet 的目标
    # Flet 会处理自己的事件循环和窗口管理
    ft.app(target=gui_main)

    # Flet 应用关闭后，程序将在这里退出
    # 可以在这里添加最后的清理代码，但通常清理工作在 GUI 的关闭事件中处理
    final_logger = get_logger(__name__)  # 获取 logger 以记录最终消息
    final_logger.info("--- VRCMeow 已退出 ---")
