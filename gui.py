import asyncio
import flet as ft
from typing import Optional, Dict, Any
import os
import pathlib
import sys
import yaml # Needed for saving config
import copy # Needed for handling config data
import logging # Add missing import for logging constants

# --- 设置正确的工作目录 ---
# 确定脚本文件所在的目录
script_dir = pathlib.Path(__file__).parent.resolve()
# 假设项目根目录与脚本目录相同，并且 config.yaml 在那里
project_root = script_dir
config_file_path = project_root / "config.yaml"

# 检查当前工作目录是否正确 (包含 config.yaml)
# 如果不正确，并且我们能找到正确的 config.yaml 路径，则更改 CWD
# 这主要用于修复 Flet 打包后 CWD 不正确的问题
current_cwd = pathlib.Path.cwd()
if not (current_cwd / "config.yaml").exists() and config_file_path.exists():
    print(f"[INFO] Initial CWD '{current_cwd}' seems incorrect (config.yaml not found).")
    print(f"[INFO] Changing CWD to detected project root: '{project_root}'")
    os.chdir(project_root)
    print(f"[INFO] Current CWD after change: '{os.getcwd()}'")
    # 如果需要，将项目根目录添加到 sys.path，以确保模块发现
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"[INFO] Added '{project_root}' to sys.path")
else:
    print(f"[INFO] Initial CWD '{current_cwd}' seems correct or config.yaml not found at expected root.")
# --- 工作目录设置结束 ---


# --- 现在导入依赖于 CWD 或 sys.path 的项目模块 ---
from config import config  # 导入 Config 类用于类型提示 <- 移到 CWD 设置之后
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
    page.window_resizable = True  # 允许调整大小
    page.padding = 10 # Add some padding around the page content

    app_state = AppState() # 创建状态实例

    # --- UI 元素 ---

    # == Dashboard Tab Elements ==
    status_text = ft.Text("状态: 未启动", selectable=True)
    output_text = ft.TextField(
        label="最终输出",
        multiline=True,
        read_only=True,
        expand=True,  # 允许文本区域扩展填充空间
        min_lines=5, # Keep min_lines for the output text
    )
    start_button = ft.ElevatedButton("启动", on_click=None, disabled=False, tooltip="启动音频处理和连接")
    stop_button = ft.ElevatedButton("停止", on_click=None, disabled=True, tooltip="停止所有后台进程")

    # == Configuration Tab Elements ==
    # We will store references to input controls to easily read their values later
    config_controls: Dict[str, ft.Control] = {}

    def create_config_section(title: str, controls: list[ft.Control]) -> ft.Card:
        """Helper to create a bordered section for config options."""
        return ft.Card(
            ft.Container(
                ft.Column([
                    ft.Text(title, style=ft.TextThemeStyle.TITLE_MEDIUM),
                    ft.Divider(height=1),
                    *controls,
                ]),
                padding=10,
            ),
            # elevation=2, # Optional: Add shadow
        )

    # -- Dashscope Settings --
    config_controls["dashscope.api_key"] = ft.TextField(
        label="API Key", # Label can be simpler now it's under Dashscope section
        value=config.get('dashscope.api_key', ''),
        password=True,
        can_reveal_password=True,
        hint_text="从环境变量 DASHSCOPE_API_KEY 覆盖",
        tooltip="阿里云 Dashscope 服务所需的 API Key"
    )
    # Add STT settings under Dashscope
    config_controls["dashscope.stt.model"] = ft.Dropdown(
        label="STT 模型",
        value=config.get('dashscope.stt.model', 'gummy-realtime-v1'), # New key
        options=[
            ft.dropdown.Option("gummy-realtime-v1", "Gummy (支持翻译)"),
            ft.dropdown.Option("paraformer-realtime-v2", "Paraformer V2 (仅识别)"),
            ft.dropdown.Option("paraformer-realtime-v1", "Paraformer V1 (仅识别)"),
        ],
        tooltip="选择 Dashscope 提供的语音识别模型"
    )
    config_controls["dashscope.stt.translation_target_language"] = ft.TextField(
        label="翻译目标语言 (Gummy)",
        value=config.get('dashscope.stt.translation_target_language') or "", # New key, Use empty string if None
        hint_text="留空则禁用翻译 (例如: en, ja, ko)",
        tooltip="如果使用 Gummy 并希望翻译，在此处输入目标语言代码"
    )
    config_controls["dashscope.stt.intermediate_result_behavior"] = ft.Dropdown(
        label="中间结果处理 (VRC OSC)",
        value=config.get('dashscope.stt.intermediate_result_behavior', 'ignore'), # New key
        options=[
            ft.dropdown.Option("ignore", "忽略"),
            ft.dropdown.Option("show_typing", "显示 'Typing...'"),
            ft.dropdown.Option("show_partial", "显示部分文本"),
        ],
        tooltip="如何处理非最终的语音识别结果 (仅影响 VRChat 输出)"
    )
    dashscope_section = create_config_section("Dashscope 设置 (含 STT)", [
        config_controls["dashscope.api_key"],
        ft.Divider(height=5), # Add a small divider
        ft.Text("语音识别 (STT)", style=ft.TextThemeStyle.TITLE_SMALL), # Sub-header
        config_controls["dashscope.stt.model"],
        config_controls["dashscope.stt.translation_target_language"],
        config_controls["dashscope.stt.intermediate_result_behavior"],
        ft.Divider(height=5), # Add another divider
        ft.Text("音频输入", style=ft.TextThemeStyle.TITLE_SMALL), # Sub-header for Audio
        config_controls["audio.sample_rate"], # Added Audio controls
        config_controls["audio.channels"],
        config_controls["audio.dtype"],
        config_controls["audio.debug_echo_mode"],
    ])


    # -- LLM API Key definition (moved here temporarily before adding to LLM section) --
    config_controls["llm.api_key"] = ft.TextField(
        label="API Key", # Label can be simpler now
        value=config.get('llm.api_key', ''),
        password=True,
        can_reveal_password=True,
        hint_text="从环境变量 OPENAI_API_KEY 覆盖",
        tooltip="用于 LLM 处理的 OpenAI 兼容 API Key (如果启用)"
    )
    # api_keys_section = create_config_section("API Keys", [...]) # REMOVED

    # -- STT Settings --  # REMOVED - These controls are now part of dashscope_section
    # config_controls["stt.model"] = ...
    # config_controls["stt.translation_target_language"] = ...
    # config_controls["stt.intermediate_result_behavior"] = ...
    # stt_section = create_config_section("语音识别 (STT)", [...]) # REMOVED


    # -- Audio Settings Definitions (Moved before Dashscope section creation) --
    config_controls["audio.sample_rate"] = ft.TextField(
        label="采样率 (Hz)",
        value=str(config.get('audio.sample_rate') or ""), # Use empty string if None
        hint_text="留空则使用设备默认值 (例如 16000)",
        keyboard_type=ft.KeyboardType.NUMBER,
        tooltip="音频输入采样率。需要与所选 STT 模型兼容"
    )
    # Note: Channels and dtype are often fixed by the STT model, maybe don't make them configurable?
    # Let's keep them for now but add a note.
    config_controls["audio.channels"] = ft.TextField(
        label="声道数",
        value=str(config.get('audio.channels', 1)),
        keyboard_type=ft.KeyboardType.NUMBER,
        tooltip="音频输入声道数 (通常为 1)"
    )
    config_controls["audio.dtype"] = ft.TextField(
        label="数据类型",
        value=config.get('audio.dtype', 'int16'),
        tooltip="音频数据类型 (例如 int16)"
    )
    config_controls["audio.debug_echo_mode"] = ft.Switch(
        label="调试回声模式",
        value=config.get('audio.debug_echo_mode', False),
        tooltip="将输入音频直接路由到输出以进行测试"
    )
    audio_section = create_config_section("音频输入", [
        config_controls["audio.sample_rate"],
        config_controls["audio.channels"],
        config_controls["audio.dtype"],
        config_controls["audio.debug_echo_mode"],
    ]) # REMOVED - Controls moved to Dashscope section


    # -- LLM Settings --
    config_controls["llm.enabled"] = ft.Switch(
        label="启用 LLM 处理",
        value=config.get('llm.enabled', False),
        tooltip="是否将识别/翻译后的文本发送给 LLM 进行处理"
    )
    config_controls["llm.base_url"] = ft.TextField(
        label="LLM API Base URL",
        value=config.get('llm.base_url') or "",
        hint_text="留空则使用 OpenAI 默认 URL",
        tooltip="用于本地 LLM 或代理 (例如 http://localhost:11434/v1)"
    )
    config_controls["llm.model"] = ft.TextField(
        label="LLM 模型",
        value=config.get('llm.model', 'gpt-3.5-turbo'),
        tooltip="要使用的 OpenAI 兼容模型名称"
    )
    config_controls["llm.system_prompt"] = ft.TextField(
        label="LLM 系统提示",
        value=config.get('llm.system_prompt', 'You are a helpful assistant.'),
        multiline=True,
        min_lines=3,
        max_lines=5,
        tooltip="指导 LLM 行为的系统消息"
    )
    config_controls["llm.temperature"] = ft.TextField(
        label="LLM Temperature",
        value=str(config.get('llm.temperature', 0.7)),
        keyboard_type=ft.KeyboardType.NUMBER,
        tooltip="控制 LLM 输出的随机性 (0.0-2.0)"
    )
    config_controls["llm.max_tokens"] = ft.TextField(
        label="LLM Max Tokens",
        value=str(config.get('llm.max_tokens', 150)),
        keyboard_type=ft.KeyboardType.NUMBER,
        tooltip="LLM 响应的最大长度"
    )
    # Few-shot examples are complex for a simple GUI, skip for now.
    llm_section = create_config_section("语言模型 (LLM)", [
        config_controls["llm.enabled"],
        config_controls["llm.api_key"], # ADDED: API Key moved into the LLM section
        config_controls["llm.base_url"],
        config_controls["llm.model"],
        config_controls["llm.system_prompt"],
        config_controls["llm.temperature"],
        config_controls["llm.max_tokens"],
    ])

    # -- Output Settings --
    # VRC OSC
    config_controls["outputs.vrc_osc.enabled"] = ft.Switch(
        label="启用 VRChat OSC 输出",
        value=config.get('outputs.vrc_osc.enabled', True),
    )
    config_controls["outputs.vrc_osc.address"] = ft.TextField(
        label="VRC OSC 地址",
        value=config.get('outputs.vrc_osc.address', '127.0.0.1'),
    )
    config_controls["outputs.vrc_osc.port"] = ft.TextField(
        label="VRC OSC 端口",
        value=str(config.get('outputs.vrc_osc.port', 9000)),
        keyboard_type=ft.KeyboardType.NUMBER,
    )
    config_controls["outputs.vrc_osc.message_interval"] = ft.TextField(
        label="VRC OSC 消息间隔 (秒)",
        value=str(config.get('outputs.vrc_osc.message_interval', 1.333)),
        keyboard_type=ft.KeyboardType.NUMBER,
        tooltip="发送到 VRChat 的最小时间间隔"
    )
    vrc_osc_output_section = create_config_section("输出: VRChat OSC", [
         config_controls["outputs.vrc_osc.enabled"],
         config_controls["outputs.vrc_osc.address"],
         config_controls["outputs.vrc_osc.port"],
         config_controls["outputs.vrc_osc.message_interval"],
    ])
    # Console
    config_controls["outputs.console.enabled"] = ft.Switch(
        label="启用控制台输出",
        value=config.get('outputs.console.enabled', True),
    )
    config_controls["outputs.console.prefix"] = ft.TextField(
        label="控制台输出前缀",
        value=config.get('outputs.console.prefix', '[Final Text]'),
    )
    console_output_section = create_config_section("输出: 控制台", [
        config_controls["outputs.console.enabled"],
        config_controls["outputs.console.prefix"],
    ])
    # File
    config_controls["outputs.file.enabled"] = ft.Switch(
        label="启用文件输出",
        value=config.get('outputs.file.enabled', False),
    )
    config_controls["outputs.file.path"] = ft.TextField(
        label="文件输出路径",
        value=config.get('outputs.file.path', 'output_log.txt'),
    )
    config_controls["outputs.file.format"] = ft.TextField(
        label="文件输出格式",
        value=config.get('outputs.file.format', '{timestamp} - {text}'),
        tooltip="可用占位符: {timestamp}, {text}"
    )
    file_output_section = create_config_section("输出: 文件", [
        config_controls["outputs.file.enabled"],
        config_controls["outputs.file.path"],
        config_controls["outputs.file.format"],
    ])

    # -- Logging Settings --
    config_controls["logging.level"] = ft.Dropdown(
        label="日志级别",
        value=config.get('logging.level', 'INFO'),
        options=[
            ft.dropdown.Option("DEBUG"),
            ft.dropdown.Option("INFO"),
            ft.dropdown.Option("WARNING"),
            ft.dropdown.Option("ERROR"),
            ft.dropdown.Option("CRITICAL"),
        ],
        tooltip="控制应用程序记录信息的详细程度"
    )
    logging_section = create_config_section("日志记录", [
        config_controls["logging.level"],
    ])

    save_config_button = ft.ElevatedButton("保存配置", on_click=None, icon=ft.icons.SAVE, tooltip="将当前设置写入 config.yaml")
    reload_config_button = ft.ElevatedButton("从文件重载", on_click=None, icon=ft.icons.REFRESH, tooltip="放弃当前更改并从 config.yaml 重新加载")

    # --- 回调函数 (用于更新 UI 和处理事件) ---
    def update_status_display(message: str):
        """线程安全地更新状态文本"""
        if page: # 确保页面仍然存在
            # 使用 run_thread 执行一个简单的 lambda 来更新 UI
            page.run_thread(
                lambda: setattr(status_text, 'value', f"状态: {message}") or page.update() # type: ignore
            )

    def update_output_display(text: str):
        """线程安全地将文本附加到输出区域"""
        if page: # 确保页面仍然存在
            # 使用 run_thread 执行一个简单的 lambda 来更新 UI
            current_value = output_text.value if output_text.value is not None else ""
            page.run_thread(
                lambda: setattr(output_text, 'value', current_value + text + "\n") or page.update()  # type: ignore
            )

    def show_snackbar(message: str, error: bool = False):
        """显示一个 SnackBar 通知"""
        if page:
            page.show_snack_bar(
                ft.SnackBar(
                    ft.Text(message),
                    bgcolor=ft.colors.ERROR if error else ft.colors.GREEN_700,
                    open=True
                )
            )

    # --- 配置保存/重载逻辑 ---
    async def save_config_handler(e: ft.ControlEvent):
        """保存按钮点击事件处理程序 (配置选项卡)"""
        logger.info("Save configuration button clicked.")
        new_config_data = copy.deepcopy(config.data) # Start with current config as base

        try:
            # Helper to safely get and convert values from controls
            def get_control_value(key: str, control_type: type = str, default: Any = None):
                control = config_controls.get(key)
                if control is None:
                    logger.warning(f"Control for config key '{key}' not found in GUI.")
                    return default

                value = getattr(control, 'value', default)

                # Handle specific control types
                if isinstance(control, ft.Switch):
                    return bool(value)
                if isinstance(control, ft.Dropdown):
                    return value # Dropdown value is usually correct type

                # Handle text fields (TextField)
                if value is None or value == "":
                    if key in ["stt.translation_target_language", "audio.sample_rate", "llm.base_url"]:
                         # These can be None in the config
                        return None
                    elif default is not None:
                        return default
                    else: # Should not happen if default config has values
                        return ""

                # Type conversion for text fields
                try:
                    if control_type == int:
                        return int(value)
                    if control_type == float:
                        return float(value)
                    return str(value) # Default to string
                except ValueError:
                    logger.error(f"Invalid value '{value}' for {key}. Expected type {control_type}. Keeping original value.")
                    # Retrieve original value to avoid saving invalid data
                    original_value = config.get(key, default)
                    return original_value


            # Recursively update the new_config_data dictionary
            def update_nested_dict(data_dict: Dict, key: str, value: Any):
                keys = key.split('.')
                d = data_dict
                for k in keys[:-1]:
                    d = d.setdefault(k, {}) # Create nested dicts if they don't exist
                d[keys[-1]] = value

            # Update dictionary from controls
            update_nested_dict(new_config_data, "dashscope.api_key", get_control_value("dashscope.api_key"))
            update_nested_dict(new_config_data, "dashscope.stt.model", get_control_value("dashscope.stt.model")) # New key
            update_nested_dict(new_config_data, "dashscope.stt.translation_target_language", get_control_value("dashscope.stt.translation_target_language")) # New key
            update_nested_dict(new_config_data, "dashscope.stt.intermediate_result_behavior", get_control_value("dashscope.stt.intermediate_result_behavior")) # New key
            update_nested_dict(new_config_data, "audio.sample_rate", get_control_value("audio.sample_rate", int, None))
            update_nested_dict(new_config_data, "audio.channels", get_control_value("audio.channels", int, 1))
            update_nested_dict(new_config_data, "audio.dtype", get_control_value("audio.dtype", str, "int16"))
            update_nested_dict(new_config_data, "audio.debug_echo_mode", get_control_value("audio.debug_echo_mode", bool, False))
            update_nested_dict(new_config_data, "llm.enabled", get_control_value("llm.enabled", bool, False))
            update_nested_dict(new_config_data, "llm.api_key", get_control_value("llm.api_key"))
            update_nested_dict(new_config_data, "llm.base_url", get_control_value("llm.base_url", str, None))
            update_nested_dict(new_config_data, "llm.model", get_control_value("llm.model"))
            update_nested_dict(new_config_data, "llm.system_prompt", get_control_value("llm.system_prompt"))
            update_nested_dict(new_config_data, "llm.temperature", get_control_value("llm.temperature", float, 0.7))
            update_nested_dict(new_config_data, "llm.max_tokens", get_control_value("llm.max_tokens", int, 150))
            update_nested_dict(new_config_data, "outputs.vrc_osc.enabled", get_control_value("outputs.vrc_osc.enabled", bool, True))
            update_nested_dict(new_config_data, "outputs.vrc_osc.address", get_control_value("outputs.vrc_osc.address"))
            update_nested_dict(new_config_data, "outputs.vrc_osc.port", get_control_value("outputs.vrc_osc.port", int, 9000))
            update_nested_dict(new_config_data, "outputs.vrc_osc.message_interval", get_control_value("outputs.vrc_osc.message_interval", float, 1.333))
            update_nested_dict(new_config_data, "outputs.console.enabled", get_control_value("outputs.console.enabled", bool, True))
            update_nested_dict(new_config_data, "outputs.console.prefix", get_control_value("outputs.console.prefix"))
            update_nested_dict(new_config_data, "outputs.file.enabled", get_control_value("outputs.file.enabled", bool, False))
            update_nested_dict(new_config_data, "outputs.file.path", get_control_value("outputs.file.path"))
            update_nested_dict(new_config_data, "outputs.file.format", get_control_value("outputs.file.format"))
            update_nested_dict(new_config_data, "logging.level", get_control_value("logging.level"))

            # Directly update the singleton's internal data BEFORE saving
            config._config_data = new_config_data
            # Recalculate derived values like logging level int after update
            log_level_str = new_config_data.get("logging", {}).get("level", "INFO").upper()
            log_level = getattr(logging, log_level_str, logging.INFO)
            config._config_data["logging"]["level_int"] = log_level

            # Call the save method on the config instance
            await asyncio.to_thread(config.save) # Run synchronous save in thread
            show_snackbar("配置已成功保存到 config.yaml")

        except Exception as ex:
            error_msg = f"保存配置时出错: {ex}"
            logger.critical(error_msg, exc_info=True)
            show_snackbar(error_msg, error=True)

    def reload_config_controls():
        """Updates the GUI controls with values from the reloaded config."""
        logger.info("Reloading config values into GUI controls.")
        for key, control in config_controls.items():
            value = config.get(key)
            try:
                if isinstance(control, ft.Switch):
                    control.value = bool(value) if value is not None else False
                elif isinstance(control, ft.Dropdown):
                    # Ensure the value exists in options, otherwise might clear selection
                    if any(opt.key == value for opt in control.options):
                         control.value = value
                    else:
                         logger.warning(f"Value '{value}' for dropdown '{key}' not in options. Keeping previous selection.")
                         # Optionally set to a default or leave as is. Let's leave it.
                elif isinstance(control, ft.TextField):
                    if key in ["stt.translation_target_language", "audio.sample_rate", "llm.base_url"] and value is None:
                         control.value = "" # Use empty string for None in text fields
                    else:
                         control.value = str(value) if value is not None else ""
                # Add other control types if necessary
            except Exception as ex:
                 logger.error(f"Error reloading control for key '{key}' with value '{value}': {ex}")
        page.update()


    async def reload_config_handler(e: ft.ControlEvent):
        """Reloads configuration from file and updates the GUI."""
        logger.info("Reload configuration button clicked.")
        try:
            await asyncio.to_thread(config.reload) # Run synchronous reload in thread
            reload_config_controls() # Update GUI fields with new values
            show_snackbar("配置已从 config.yaml 重新加载")
        except Exception as ex:
             error_msg = f"重新加载配置时出错: {ex}"
             logger.error(error_msg, exc_info=True)
             show_snackbar(error_msg, error=True)


    # --- 启动/停止逻辑 (Dashboard Tab) ---
    async def start_recording(e: ft.ControlEvent):
        """启动按钮点击事件处理程序 (Dashboard Tab)"""
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
            # Use the new nested key 'dashscope.api_key'
            dashscope_api_key = config.get("dashscope.api_key")
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

    # --- Bind event handlers ---
    start_button.on_click = start_recording
    stop_button.on_click = stop_recording
    save_config_button.on_click = save_config_handler
    reload_config_button.on_click = reload_config_handler
    page.on_window_event = on_window_event

    # --- Layout using Tabs ---
    dashboard_tab_content = ft.Column(
        [
            ft.Row([start_button, stop_button], alignment=ft.MainAxisAlignment.CENTER),
            status_text,
            ft.Container(output_text, expand=True), # Make output text area fill available space
        ],
        expand=True, # Make column fill tab height
        alignment=ft.MainAxisAlignment.START,
        spacing=10, # Add spacing between elements
    )

    config_tab_content = ft.Column(
        [
            ft.Row( # Buttons Row
                [save_config_button, reload_config_button],
                alignment=ft.MainAxisAlignment.END # Align buttons to the right
            ),
            ft.Divider(height=10),
            # Sections - Removed stt_section and audio_section
            dashscope_section, # Now contains STT and Audio settings
            # stt_section, # REMOVED
            # audio_section, # REMOVED
            llm_section,
            vrc_osc_output_section,
            console_output_section,
            file_output_section,
            logging_section,
        ],
        expand=True,
        scroll=ft.ScrollMode.ADAPTIVE, # Make config tab scrollable if needed
        spacing=15, # Space between config sections/elements
    )


    page.add(
        ft.Tabs(
            [
                ft.Tab(
                    text="仪表盘",
                    icon=ft.icons.DASHBOARD,
                    content=dashboard_tab_content
                ),
                ft.Tab(
                    text="配置",
                    icon=ft.icons.SETTINGS,
                    content=config_tab_content
                ),
            ],
            expand=True, # Make tabs fill the page width
        )
    )

    # Initial page update
    page.update()

# 注意：此文件不再包含 if __name__ == "__main__": ft.app(...)
# 这将移至 main.py
