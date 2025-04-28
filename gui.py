import asyncio
import threading
import flet as ft
from typing import Optional

# 从项目中导入所需模块
from config import config, Config  # 导入 Config 类用于类型提示
from logger_config import setup_logging, get_logger
from audio_recorder import AudioManager
from output_dispatcher import OutputDispatcher
from llm_client import LLMClient
from osc_client import VRCClient


# 初始化日志记录 (在 Flet 应用启动前)
setup_logging()
logger = get_logger("VRCMeowGUI")


class AppState:
    """简单类，用于在回调函数之间共享状态"""
    def __init__(self):
        self.is_running = False
        self.audio_manager: Optional[AudioManager] = None
        self.output_dispatcher: Optional[OutputDispatcher] = None
        self.llm_client: Optional[LLMClient] = None
        self.vrc_client: Optional[VRCClient] = None


def main(page: ft.Page):
    """Flet GUI 主函数"""
    page.title = "VRCMeow Dashboard"
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    # 设置初始窗口大小 (可选)
    page.window_width = 600
    page.window_height = 450
    page.window_resizable = True # 允许调整大小

    app_state = AppState() # 创建状态实例

    # --- UI 元素 ---
    status_text = ft.Text("状态: 未启动", selectable=True)
    output_text = ft.TextField(
        label="最终输出",
        multiline=True,
        read_only=True,
        expand=True, # 允许文本区域扩展填充空间
        min_lines=5,
    )
    start_button = ft.ElevatedButton("启动", on_click=None, disabled=False)
    stop_button = ft.ElevatedButton("停止", on_click=None, disabled=True)

    # --- 回调函数 (用于更新 UI) ---
    def update_status_display(message: str):
        """线程安全地更新状态文本"""
        if page: # 确保页面仍然存在
            page.run_thread_safe(
                lambda: setattr(status_text, 'value', f"状态: {message}") or page.update() # type: ignore
            )

    def update_output_display(text: str):
        """线程安全地将文本附加到输出区域"""
        if page: # 确保页面仍然存在
            page.run_thread_safe(
                lambda: setattr(output_text, 'value', (output_text.value or "") + text + "\n") or page.update() # type: ignore
            )

    # --- 启动/停止逻辑 ---
    async def start_recording(e: ft.ControlEvent):
        """启动按钮点击事件处理程序"""
        if app_state.is_running:
            return

        update_status_display("正在启动...")
        start_button.disabled = True
        stop_button.disabled = False
        page.update() # 更新按钮状态

        try:
            # --- 初始化组件 ---
            logger.info("GUI 请求启动，正在初始化组件...")

            # 检查关键配置 (可以在这里再次检查或依赖 main.py 中的早期检查)
            dashscope_api_key = config.get("dashscope_api_key")
            if not dashscope_api_key:
                 error_msg = "错误：Dashscope API Key 未设置。"
                 logger.error(error_msg)
                 update_status_display(error_msg)
                 # 重置按钮状态
                 start_button.disabled = False
                 stop_button.disabled = True
                 page.update()
                 return

            # 1. VRCClient (如果启用)
            vrc_osc_enabled = config.get("outputs.vrc_osc.enabled", False)
            if vrc_osc_enabled:
                osc_address = config.get("outputs.vrc_osc.address", "127.0.0.1")
                osc_port = config.get("outputs.vrc_osc.port", 9000)
                osc_interval = config.get("outputs.vrc_osc.message_interval", 1.333)
                app_state.vrc_client = VRCClient(
                    address=osc_address, port=osc_port, interval=osc_interval
                )
                # 在 AudioManager 启动前启动 VRCClient
                await app_state.vrc_client.start() # VRCClient 现在需要显式启动
                logger.info("VRCClient 已初始化并启动。")
            else:
                logger.info("VRC OSC 输出已禁用，跳过 VRCClient 初始化。")
                app_state.vrc_client = None

            # 2. LLMClient (如果启用)
            llm_enabled = config.get("llm.enabled", False)
            if llm_enabled:
                app_state.llm_client = LLMClient()
                if not app_state.llm_client.enabled:
                    logger.warning("LLMClient 初始化失败或 API Key 缺失，LLM 处理将被禁用。")
                    app_state.llm_client = None
                else:
                    logger.info("LLMClient 已初始化。")
            else:
                app_state.llm_client = None

            # 3. OutputDispatcher (传递 VRC 客户端和 GUI 输出回调)
            app_state.output_dispatcher = OutputDispatcher(
                vrc_client_instance=app_state.vrc_client,
                gui_output_callback=update_output_display # 传递新的回调
            )
            logger.info("OutputDispatcher 已初始化。")

            # 4. AudioManager (传递 LLM 客户端、调度器和状态回调)
            app_state.audio_manager = AudioManager(
                llm_client=app_state.llm_client,
                output_dispatcher=app_state.output_dispatcher,
                status_callback=update_status_display # 传递状态更新回调
            )
            logger.info("AudioManager 已初始化。")

            # --- 启动 AudioManager ---
            # AudioManager.start() 会启动后台线程
            app_state.audio_manager.start()
            app_state.is_running = True
            logger.info("AudioManager 已启动。系统运行中。")
            # 状态将在 AudioManager 内部通过回调更新为 "Running" 等

        except Exception as ex:
            error_msg = f"启动过程中出错: {ex}"
            logger.critical(error_msg, exc_info=True)
            update_status_display(error_msg)
            # 尝试清理可能已部分启动的资源
            await stop_recording(e, is_error=True) # 调用停止逻辑进行清理

    async def stop_recording(e: Optional[ft.ControlEvent], is_error: bool = False):
        """停止按钮点击事件处理程序或错误处理调用"""
        if not app_state.is_running and not is_error: # 如果因错误调用，即使未标记为运行也要尝试停止
            return

        update_status_display("正在停止...")
        start_button.disabled = True # 在停止完成前禁用两个按钮
        stop_button.disabled = True
        page.update()

        logger.info("GUI 请求停止...")
        tasks_to_await = []

        # 停止 AudioManager (这会触发 STT 和音频流的停止)
        if app_state.audio_manager:
            logger.info("正在停止 AudioManager...")
            # AudioManager.stop() 是同步的，但在内部等待线程
            # 在 Flet 事件处理程序中直接调用可能导致 UI 冻结，最好在线程中运行
            # 或者，AudioManager.stop() 本身应该是非阻塞信号，然后我们异步等待线程结束
            # 当前 AudioManager.stop() 是阻塞的，这在 Flet 事件循环中可能不是最佳实践
            # 为了简单起见，暂时直接调用，但请注意这可能导致 UI 短暂无响应
            try:
                # 将阻塞操作放入线程以避免冻结 UI
                await asyncio.to_thread(app_state.audio_manager.stop)
                logger.info("AudioManager 已停止。")
            except Exception as am_stop_err:
                 logger.error(f"停止 AudioManager 时出错: {am_stop_err}", exc_info=True)
            app_state.audio_manager = None

        # 停止 VRCClient (如果存在)
        if app_state.vrc_client:
            logger.info("正在停止 VRCClient...")
            try:
                await app_state.vrc_client.stop()
                logger.info("VRCClient 已停止。")
            except Exception as vrc_stop_err:
                 logger.error(f"停止 VRCClient 时出错: {vrc_stop_err}", exc_info=True)
            app_state.vrc_client = None

        # 清理其他资源
        app_state.llm_client = None
        app_state.output_dispatcher = None

        app_state.is_running = False
        logger.info("所有组件已停止。")
        update_status_display("已停止")
        start_button.disabled = False # 重新启用启动按钮
        stop_button.disabled = True
        page.update()

    # --- 绑定事件 ---
    start_button.on_click = start_recording
    stop_button.on_click = stop_recording

    # --- 页面关闭处理 ---
    async def on_window_event(e: ft.ControlEvent):
        if e.data == "close":
            logger.info("检测到窗口关闭事件。")
            page.window_destroy() # 允许窗口关闭
            # 确保在关闭前停止运行中的进程
            if app_state.is_running:
                 logger.info("窗口关闭时正在停止后台进程...")
                 await stop_recording(None) # 调用停止逻辑

    page.on_window_event = on_window_event

    # --- 布局 ---
    page.add(
        ft.Column(
            [
                ft.Row([start_button, stop_button], alignment=ft.MainAxisAlignment.CENTER),
                status_text,
                ft.Container(output_text, expand=True), # 使文本区域填充剩余空间
            ],
            expand=True # 使列扩展以填充页面
        )
    )

    # 初始页面更新
    page.update()

# 注意：此文件不再包含 if __name__ == "__main__": ft.app(...)
# 这将移至 main.py
