import flet as ft
from gui import main as gui_main
from logger_config import get_logger


# 旧的 async def main() 函数及其所有内容都已删除或移至 gui.py


if __name__ == "__main__":
    # Logging setup is handled within gui.py
    # Start the Flet application, targeting the main function in gui.py
    ft.app(target=gui_main)

    # Code here runs after the Flet app closes
    final_logger = get_logger(__name__)
    final_logger.info("--- VRCMeow 已退出 ---")
