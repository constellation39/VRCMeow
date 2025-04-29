import flet as ft
from typing import Dict, Any, Optional
import logging

# Use standard logging; setup happens elsewhere
logger = logging.getLogger(__name__)

# --- Configuration UI Element Definitions & Helpers ---

# The central dictionary `all_config_controls` is defined and managed in gui.py.

def create_config_section(title: str, controls: list[ft.Control]) -> ft.Card:
    """Helper to create a bordered section for config options."""
    return ft.Card(
        ft.Container(
            ft.Column(
                [
                    ft.Text(title, style=ft.TextThemeStyle.TITLE_MEDIUM),
                    ft.Divider(height=1),
                    *controls,
                ]
            ),
            padding=10,
        ),
    )

# --- Control Creation Functions (called from gui.py) ---

def create_dashscope_controls(initial_config: Dict[str, Any]) -> Dict[str, ft.Control]:
    """Creates controls for the Dashscope section."""
    controls = {}
    dashscope_conf = initial_config.get("dashscope", {})
    stt_conf = dashscope_conf.get("stt", {})

    controls["dashscope.api_key"] = ft.TextField(
        label="API Key",
        value=dashscope_conf.get("api_key", ""),
        password=True,
        can_reveal_password=True,
        hint_text="从环境变量 DASHSCOPE_API_KEY 覆盖",
        tooltip="阿里云 Dashscope 服务所需的 API Key",
    )
    controls["dashscope.stt.model"] = ft.Dropdown(
        label="STT 模型",
        value=stt_conf.get("model", "gummy-realtime-v1"),
        options=[
            ft.dropdown.Option("gummy-realtime-v1", "Gummy (支持翻译)"),
            ft.dropdown.Option("paraformer-realtime-v2", "Paraformer V2 (仅识别)"),
            ft.dropdown.Option("paraformer-realtime-v1", "Paraformer V1 (仅识别)"),
        ],
        tooltip="选择 Dashscope 提供的语音识别模型",
    )
    controls["dashscope.stt.translation_target_language"] = ft.TextField(
        label="翻译目标语言 (Gummy)",
        value=stt_conf.get("translation_target_language") or "", # Handle None
        hint_text="留空则禁用翻译 (例如: en, ja, ko)",
        tooltip="如果使用 Gummy 并希望翻译，在此处输入目标语言代码",
    )
    controls["dashscope.stt.intermediate_result_behavior"] = ft.Dropdown(
        label="中间结果处理 (VRC OSC)",
        value=stt_conf.get("intermediate_result_behavior", "ignore"),
        options=[
            ft.dropdown.Option("ignore", "忽略"),
            ft.dropdown.Option("show_typing", "显示 'Typing...'"),
            ft.dropdown.Option("show_partial", "显示部分文本"),
        ],
        tooltip="如何处理非最终的语音识别结果 (仅影响 VRChat 输出)",
    )
    return controls

def create_audio_controls(initial_config: Dict[str, Any]) -> Dict[str, ft.Control]:
    """Creates controls for the Audio Input section."""
    controls = {}
    audio_conf = initial_config.get("audio", {})
    controls["audio.sample_rate"] = ft.TextField(
        label="采样率 (Hz)",
        value=str(audio_conf.get("sample_rate") or ""),
        hint_text="留空则使用设备默认值 (例如 16000)",
        keyboard_type=ft.KeyboardType.NUMBER,
        tooltip="音频输入采样率。需要与所选 STT 模型兼容",
    )
    controls["audio.channels"] = ft.TextField(
        label="声道数",
        value=str(audio_conf.get("channels", 1)),
        keyboard_type=ft.KeyboardType.NUMBER,
        tooltip="音频输入声道数 (通常为 1)",
    )
    controls["audio.dtype"] = ft.TextField(
        label="数据类型",
        value=audio_conf.get("dtype", "int16"),
        tooltip="音频数据类型 (例如 int16)",
    )
    controls["audio.debug_echo_mode"] = ft.Switch(
        label="调试回声模式",
        value=audio_conf.get("debug_echo_mode", False),
        tooltip="将输入音频直接路由到输出以进行测试",
    )
    return controls

def create_llm_controls(initial_config: Dict[str, Any]) -> Dict[str, ft.Control]:
    """Creates controls for the LLM section."""
    controls = {}
    llm_conf = initial_config.get("llm", {})
    controls["llm.enabled"] = ft.Switch(
        label="启用 LLM 处理",
        value=llm_conf.get("enabled", False),
        tooltip="是否将识别/翻译后的文本发送给 LLM 进行处理",
    )
    controls["llm.api_key"] = ft.TextField(
        label="API Key",
        value=llm_conf.get("api_key", ""),
        password=True,
        can_reveal_password=True,
        hint_text="从环境变量 OPENAI_API_KEY 覆盖",
        tooltip="用于 LLM 处理的 OpenAI 兼容 API Key (如果启用)",
    )
    controls["llm.base_url"] = ft.TextField(
        label="LLM API Base URL",
        value=llm_conf.get("base_url") or "",
        hint_text="留空则使用 OpenAI 默认 URL",
        tooltip="用于本地 LLM 或代理 (例如 http://localhost:11434/v1)",
    )
    controls["llm.model"] = ft.TextField(
        label="LLM 模型",
        value=llm_conf.get("model", "gpt-3.5-turbo"),
        tooltip="要使用的 OpenAI 兼容模型名称",
    )
    controls["llm.system_prompt"] = ft.TextField(
        label="LLM 系统提示",
        value=llm_conf.get("system_prompt", "You are a helpful assistant."),
        multiline=True,
        min_lines=3,
        max_lines=5,
        tooltip="指导 LLM 行为的系统消息",
    )
    controls["llm.temperature"] = ft.TextField(
        label="LLM Temperature",
        value=str(llm_conf.get("temperature", 0.7)),
        keyboard_type=ft.KeyboardType.NUMBER,
        tooltip="控制 LLM 输出的随机性 (0.0-2.0)",
    )
    controls["llm.max_tokens"] = ft.TextField(
        label="LLM Max Tokens",
        value=str(llm_conf.get("max_tokens", 150)),
        keyboard_type=ft.KeyboardType.NUMBER,
        tooltip="LLM 响应的最大长度",
    )
    # Few-shot examples UI elements (created in main, passed to layout function)
    # Define placeholders here; actual controls created in gui.py
    controls["llm.few_shot_examples_column"] = ft.Column(controls=[], spacing=5)
    controls["llm.add_example_button"] = ft.TextButton("添加 Few-Shot 示例", icon=ft.icons.ADD)
    return controls

def create_vrc_osc_controls(initial_config: Dict[str, Any]) -> Dict[str, ft.Control]:
    """Creates controls for the VRC OSC Output section."""
    controls = {}
    output_conf = initial_config.get("outputs", {})
    vrc_conf = output_conf.get("vrc_osc", {})
    controls["outputs.vrc_osc.enabled"] = ft.Switch(
        label="启用 VRChat OSC 输出",
        value=vrc_conf.get("enabled", True),
    )
    controls["outputs.vrc_osc.address"] = ft.TextField(
        label="VRC OSC 地址",
        value=vrc_conf.get("address", "127.0.0.1"),
    )
    controls["outputs.vrc_osc.port"] = ft.TextField(
        label="VRC OSC 端口",
        value=str(vrc_conf.get("port", 9000)),
        keyboard_type=ft.KeyboardType.NUMBER,
    )
    controls["outputs.vrc_osc.message_interval"] = ft.TextField(
        label="VRC OSC 消息间隔 (秒)",
        value=str(vrc_conf.get("message_interval", 1.333)),
        keyboard_type=ft.KeyboardType.NUMBER,
        tooltip="发送到 VRChat 的最小时间间隔",
    )
    return controls

def create_console_output_controls(initial_config: Dict[str, Any]) -> Dict[str, ft.Control]:
    """Creates controls for the Console Output section."""
    controls = {}
    output_conf = initial_config.get("outputs", {})
    console_conf = output_conf.get("console", {})
    controls["outputs.console.enabled"] = ft.Switch(
        label="启用控制台输出",
        value=console_conf.get("enabled", True),
    )
    controls["outputs.console.prefix"] = ft.TextField(
        label="控制台输出前缀",
        value=console_conf.get("prefix", "[Final Text]"),
    )
    return controls

def create_file_output_controls(initial_config: Dict[str, Any]) -> Dict[str, ft.Control]:
    """Creates controls for the File Output section."""
    controls = {}
    output_conf = initial_config.get("outputs", {})
    file_conf = output_conf.get("file", {})
    controls["outputs.file.enabled"] = ft.Switch(
        label="启用文件输出",
        value=file_conf.get("enabled", False),
    )
    controls["outputs.file.path"] = ft.TextField(
        label="文件输出路径",
        value=file_conf.get("path", "output_log.txt"),
    )
    controls["outputs.file.format"] = ft.TextField(
        label="文件输出格式",
        value=file_conf.get("format", "{timestamp} - {text}"),
        tooltip="可用占位符: {timestamp}, {text}",
    )
    return controls

def create_logging_controls(initial_config: Dict[str, Any]) -> Dict[str, ft.Control]:
    """Creates controls for the Logging section."""
    controls = {}
    log_conf = initial_config.get("logging", {})
    controls["logging.level"] = ft.Dropdown(
        label="日志级别",
        value=log_conf.get("level", "INFO"),
        options=[
            ft.dropdown.Option("DEBUG"),
            ft.dropdown.Option("INFO"),
            ft.dropdown.Option("WARNING"),
            ft.dropdown.Option("ERROR"),
            ft.dropdown.Option("CRITICAL"),
        ],
        tooltip="控制应用程序记录信息的详细程度",
    )
    return controls


# --- Configuration Layout Function ---
# Takes created controls and buttons as arguments

def create_config_tab_content(
    save_button: ft.ElevatedButton,
    reload_button: ft.ElevatedButton,
    all_controls: Dict[str, ft.Control] # Pass the complete dictionary
) -> ft.Column:
    """Creates the layout Column for the Configuration tab."""

    # Helper to get control by key, avoids KeyError
    def get_ctrl(key: str) -> Optional[ft.Control]:
        ctrl = all_controls.get(key)
        if ctrl is None:
             logger.warning(f"Control '{key}' not found for config layout.")
        return ctrl

    # Extract few-shot elements (created in gui.py, passed via all_controls)
    few_shot_column = get_ctrl("llm.few_shot_examples_column") or ft.Column() # Fallback
    add_example_btn = get_ctrl("llm.add_example_button") or ft.TextButton() # Fallback

    dashscope_section = create_config_section(
        "Dashscope 设置",
        [
            get_ctrl("dashscope.api_key"),
            ft.Divider(height=5),
            get_ctrl("dashscope.stt.model"),
            get_ctrl("dashscope.stt.translation_target_language"),
            get_ctrl("dashscope.stt.intermediate_result_behavior"),
            ft.Divider(height=5),
            ft.Text("音频输入", style=ft.TextThemeStyle.TITLE_SMALL),
            get_ctrl("audio.sample_rate"),
            get_ctrl("audio.channels"),
            get_ctrl("audio.dtype"),
            get_ctrl("audio.debug_echo_mode"),
        ],
    )

    llm_section = create_config_section(
        "语言模型 (LLM)",
        [
            get_ctrl("llm.enabled"),
            get_ctrl("llm.api_key"),
            get_ctrl("llm.base_url"),
            get_ctrl("llm.model"),
            get_ctrl("llm.system_prompt"),
            get_ctrl("llm.temperature"),
            get_ctrl("llm.max_tokens"),
            ft.Divider(height=5),
            ft.Text("Few-Shot 示例", style=ft.TextThemeStyle.TITLE_SMALL),
            ft.Text("这些示例指导 LLM 如何响应特定输入。", size=11, italic=True),
            few_shot_column,
            add_example_btn,
        ],
    )

    vrc_osc_output_section = create_config_section(
        "输出: VRChat OSC",
        [
            get_ctrl("outputs.vrc_osc.enabled"),
            get_ctrl("outputs.vrc_osc.address"),
            get_ctrl("outputs.vrc_osc.port"),
            get_ctrl("outputs.vrc_osc.message_interval"),
        ],
    )

    console_output_section = create_config_section(
        "输出: 控制台",
        [
            get_ctrl("outputs.console.enabled"),
            get_ctrl("outputs.console.prefix"),
        ],
    )

    file_output_section = create_config_section(
        "输出: 文件",
        [
            get_ctrl("outputs.file.enabled"),
            get_ctrl("outputs.file.path"),
            get_ctrl("outputs.file.format"),
        ],
    )

    logging_section = create_config_section(
        "日志记录",
        [
            get_ctrl("logging.level"),
        ],
    )

    # Filter out None controls before adding to layout
    layout_controls = [
        ctrl for ctrl in [
            ft.Row(
                [save_button, reload_button],
                alignment=ft.MainAxisAlignment.END,
            ),
            ft.Divider(height=10),
            dashscope_section,
            llm_section,
            vrc_osc_output_section,
            console_output_section,
            file_output_section,
            logging_section,
        ] if ctrl is not None
    ]


    return ft.Column(
        controls=layout_controls,
        expand=True,
        scroll=ft.ScrollMode.ADAPTIVE,
        spacing=15,
    )
