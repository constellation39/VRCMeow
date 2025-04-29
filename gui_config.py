import flet as ft
from typing import Dict, Any, Optional, Callable, TYPE_CHECKING, List
import logging
import asyncio
import sounddevice as sd # Import sounddevice
import copy
import gui_utils  # Import for close_banner
from audio_recorder import get_input_devices # Import device getter


# Use standard logging; setup happens elsewhere
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from config import Config  # Type hint only
    # Avoid circular import for Config singleton instance, pass it as arg

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
        value=stt_conf.get("translation_target_language") or "",  # Handle None
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

    # --- Microphone Selection Dropdown ---
    device_options = []
    selected_device_value = audio_conf.get("device", "Default") # Get configured value
    try:
        available_devices = get_input_devices()
        # Add "Default" option first
        device_options.append(ft.dropdown.Option(key="Default", text="Default Input Device"))
        # Add other devices
        for device in available_devices:
            if device.get("name") == "Error querying devices": # Handle error case from get_input_devices
                 device_options.append(ft.dropdown.Option(key="error", text="Error querying devices", disabled=True))
                 continue
            # Use the user-friendly name for display and also as the key to store in config
            device_name = device.get("name", "Unknown Device")
            option_text = f"{device_name}"
            if device.get("is_default"):
                option_text += " (System Default)" # Indicate which one is the system default

            device_options.append(ft.dropdown.Option(key=device_name, text=option_text))

        # Ensure the configured value is actually in the options list
        if selected_device_value != "Default" and not any(opt.key == selected_device_value for opt in device_options):
            logger.warning(f"Configured audio device '{selected_device_value}' not found in available devices. Falling back to Default.")
            # Add a temporary option for the missing device? Or just fallback? Fallback is safer.
            # device_options.append(ft.dropdown.Option(key=selected_device_value, text=f"{selected_device_value} (Not Found)", disabled=True))
            selected_device_value = "Default" # Reset to default if not found

    except Exception as e:
        logger.error(f"Failed to populate microphone list: {e}", exc_info=True)
        device_options.append(ft.dropdown.Option(key="error", text="Error loading devices", disabled=True))
        selected_device_value = "Default" # Fallback

    controls["audio.device"] = ft.Dropdown(
        label="麦克风 (输入设备)",
        value=selected_device_value,
        options=device_options,
        tooltip="选择要使用的音频输入设备",
        # Add on_change handler? Maybe later for dynamic sample rate update.
    )
    # --- End Microphone Selection ---

    controls["audio.sample_rate"] = ft.TextField(
        label="采样率 (Hz)",
        value=str(audio_conf.get("sample_rate") or ""), # Keep existing logic
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
    controls["llm.add_example_button"] = ft.TextButton(
        "添加 Few-Shot 示例", icon=ft.icons.ADD
    )
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


def create_console_output_controls(
    initial_config: Dict[str, Any],
) -> Dict[str, ft.Control]:
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


def create_file_output_controls(
    initial_config: Dict[str, Any],
) -> Dict[str, ft.Control]:
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
    all_controls: Dict[str, ft.Control],  # Pass the complete dictionary
) -> ft.Column:
    """Creates the layout Column for the Configuration tab."""

    # Helper to get control by key, avoids KeyError
    def get_ctrl(key: str) -> Optional[ft.Control]:
        ctrl = all_controls.get(key)
        if ctrl is None:
            logger.warning(f"Control '{key}' not found for config layout.")
        return ctrl

    # Extract few-shot elements (created in gui.py, passed via all_controls)
    few_shot_column = (
        get_ctrl("llm.few_shot_examples_column") or ft.Column()
    )  # Fallback
    add_example_btn = get_ctrl("llm.add_example_button") or ft.TextButton()  # Fallback

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
            get_ctrl("audio.device"), # Add device dropdown here
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
        ctrl
        for ctrl in [
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
        ]
        if ctrl is not None
    ]

    # Ensure controls are valid before adding
    valid_layout_controls = [c for c in layout_controls if c is not None]
    if len(valid_layout_controls) != len(layout_controls):
        logger.warning("Some config layout controls were None and excluded.")

    return ft.Column(
        controls=valid_layout_controls,
        expand=True,
        scroll=ft.ScrollMode.ADAPTIVE,
        spacing=15,
    )


# --- Configuration Helper Functions & Handlers ---


# --- Helper function to get value from a control ---
def get_control_value(
    all_config_controls: Dict[str, ft.Control],  # Pass controls dict
    key: str,
    control_type: type = str,
    default: Any = None,
) -> Any:
    """Safely retrieves and converts the value from a GUI control."""
    control = all_config_controls.get(key)
    if control is None:
        # Don't log warning here, let caller decide if missing is an error
        # logger.warning(f"Control for config key '{key}' not found in GUI. Returning default: {default}")
        return default

    value = getattr(control, "value", default)

    # Handle specific control types first
    if isinstance(control, ft.Switch):
        # Ensure bool conversion, default to False if default is None
        return (
            bool(value)
            if value is not None
            else (bool(default) if default is not None else False)
        )
    if isinstance(control, ft.Dropdown):
        # Return the selected value directly, handle None case
        return value if value is not None else default

    # Handle text fields (TextField)
    # Treat None or empty string carefully
    if value is None or value == "":
        # Check specific keys that explicitly allow None or empty string maps to None
        if key in [
            "dashscope.stt.translation_target_language",  # Empty string means None
            "audio.sample_rate",  # Empty string means None (auto-detect)
            "llm.base_url",  # Empty string means None
            "llm.api_key",  # Empty string is valid (but might cause issues later)
            "dashscope.api_key",  # Empty string is valid
        ]:
            # Return None if empty string, otherwise return default
            return None if value == "" else default

        # If a default value is provided, return it for other empty fields
        if default is not None:
            return default
        # Otherwise, return None for numeric types or empty string for strings
        elif control_type in [int, float]:
            logger.debug(f"Empty value for numeric field '{key}', returning None.")
            return None  # Cannot convert empty string to number
        else:
            return ""  # Default empty string for text

    # Attempt type conversion for non-empty values from text-based inputs
    try:
        if control_type == int:
            return int(value)
        if control_type == float:
            # Handle potential locale issues if needed, assuming standard decimal format
            return float(value)
        if control_type == bool:  # Should be handled by Switch, but as fallback
            return str(value).lower() in ["true", "1", "yes", "on"]
        # Default to string if no other type matches (or if control_type is str)
        return str(value)
    except (ValueError, TypeError) as convert_err:
        logger.error(
            f"Invalid value '{value}' for '{key}'. Expected type {control_type}. Error: {convert_err}. Returning default value: {default}"
        )
        # Show banner? Requires page ref. For now, just return default.
        return default  # Return default on conversion error


# --- Configuration Save/Reload Logic ---
async def save_config_handler(
    page: ft.Page,  # Need page for banner
    all_config_controls: Dict[str, ft.Control],  # Need controls dict
    config_instance: "Config",  # Need config instance
):
    """保存按钮点击事件处理程序 (配置选项卡)"""
    logger.info("Save configuration button clicked.")
    if not config_instance:
        logger.error("Cannot save config, config object not available.")
        # Show error banner
        gui_utils.show_error_banner(page, "无法保存配置：配置对象不可用。")
        return
    # Start with a deep copy of the *current live* config data
    new_config_data = copy.deepcopy(config_instance.data)

    try:
        # Helper to update nested dictionary safely
        def update_nested_dict(data_dict: Dict, key: str, value: Any):
            keys = key.split(".")
            temp_dict = data_dict
            for i, k in enumerate(keys[:-1]):
                # Ensure intermediate level is a dictionary, create if not
                if not isinstance(temp_dict.get(k), dict):
                    logger.debug(
                        f"Creating intermediate dict for key '{k}' during save."
                    )
                    temp_dict[k] = {}
                temp_dict = temp_dict[k]
            # Set the final value
            temp_dict[keys[-1]] = value

        # --- Update dictionary from controls using get_control_value ---
        # Define expected type and default for each control value retrieval
        update_nested_dict(
            new_config_data,
            "dashscope.api_key",
            get_control_value(all_config_controls, "dashscope.api_key", str, ""),
        )
        update_nested_dict(
            new_config_data,
            "dashscope.stt.model",
            get_control_value(
                all_config_controls, "dashscope.stt.model", str, "gummy-realtime-v1"
            ),
        )
        update_nested_dict(
            new_config_data,
            "dashscope.stt.translation_target_language",
            get_control_value(
                all_config_controls,
                "dashscope.stt.translation_target_language",
                str,
                None,
            ),
        )
        update_nested_dict(
            new_config_data,
            "dashscope.stt.intermediate_result_behavior",
            get_control_value(
                all_config_controls,
                "dashscope.stt.intermediate_result_behavior",
                str,
                "ignore",
            ),
        )

        update_nested_dict(
            new_config_data,
            "audio.device", # Save selected device
            get_control_value(all_config_controls, "audio.device", str, "Default"),
        )
        update_nested_dict(
            new_config_data,
            "audio.sample_rate",
            get_control_value(all_config_controls, "audio.sample_rate", int, None),
        )
        update_nested_dict(
            new_config_data,
            "audio.channels",
            get_control_value(all_config_controls, "audio.channels", int, 1),
        )
        update_nested_dict(
            new_config_data,
            "audio.dtype",
            get_control_value(all_config_controls, "audio.dtype", str, "int16"),
        )
        update_nested_dict(
            new_config_data,
            "audio.debug_echo_mode",
            get_control_value(
                all_config_controls, "audio.debug_echo_mode", bool, False
            ),
        )

        update_nested_dict(
            new_config_data,
            "llm.enabled",
            get_control_value(all_config_controls, "llm.enabled", bool, False),
        )
        update_nested_dict(
            new_config_data,
            "llm.api_key",
            get_control_value(all_config_controls, "llm.api_key", str, ""),
        )
        update_nested_dict(
            new_config_data,
            "llm.base_url",
            get_control_value(all_config_controls, "llm.base_url", str, None),
        )
        update_nested_dict(
            new_config_data,
            "llm.model",
            get_control_value(all_config_controls, "llm.model", str, "gpt-3.5-turbo"),
        )  # Provide default
        update_nested_dict(
            new_config_data,
            "llm.system_prompt",
            get_control_value(
                all_config_controls,
                "llm.system_prompt",
                str,
                "You are a helpful assistant.",
            ),
        )  # Provide default
        update_nested_dict(
            new_config_data,
            "llm.temperature",
            get_control_value(all_config_controls, "llm.temperature", float, 0.7),
        )
        update_nested_dict(
            new_config_data,
            "llm.max_tokens",
            get_control_value(all_config_controls, "llm.max_tokens", int, 150),
        )

        update_nested_dict(
            new_config_data,
            "outputs.vrc_osc.enabled",
            get_control_value(
                all_config_controls, "outputs.vrc_osc.enabled", bool, True
            ),
        )
        update_nested_dict(
            new_config_data,
            "outputs.vrc_osc.address",
            get_control_value(
                all_config_controls, "outputs.vrc_osc.address", str, "127.0.0.1"
            ),
        )
        update_nested_dict(
            new_config_data,
            "outputs.vrc_osc.port",
            get_control_value(all_config_controls, "outputs.vrc_osc.port", int, 9000),
        )
        update_nested_dict(
            new_config_data,
            "outputs.vrc_osc.message_interval",
            get_control_value(
                all_config_controls, "outputs.vrc_osc.message_interval", float, 1.333
            ),
        )

        update_nested_dict(
            new_config_data,
            "outputs.console.enabled",
            get_control_value(
                all_config_controls, "outputs.console.enabled", bool, True
            ),
        )
        update_nested_dict(
            new_config_data,
            "outputs.console.prefix",
            get_control_value(
                all_config_controls, "outputs.console.prefix", str, "[Final Text]"
            ),
        )  # Match default

        update_nested_dict(
            new_config_data,
            "outputs.file.enabled",
            get_control_value(all_config_controls, "outputs.file.enabled", bool, False),
        )
        update_nested_dict(
            new_config_data,
            "outputs.file.path",
            get_control_value(
                all_config_controls, "outputs.file.path", str, "output_log.txt"
            ),
        )  # Match default
        update_nested_dict(
            new_config_data,
            "outputs.file.format",
            get_control_value(
                all_config_controls, "outputs.file.format", str, "{timestamp} - {text}"
            ),
        )  # Match default

        update_nested_dict(
            new_config_data,
            "logging.level",
            get_control_value(all_config_controls, "logging.level", str, "INFO"),
        )
        # --- End updating from controls ---

        # Update few-shot examples
        examples_list = []
        few_shot_column = all_config_controls.get("llm.few_shot_examples_column")
        if few_shot_column and isinstance(few_shot_column, ft.Column):
            for row in few_shot_column.controls:
                if (
                    isinstance(row, ft.Row) and len(row.controls) >= 3
                ):  # user, assistant, remove_button
                    user_tf = (
                        row.controls[0]
                        if isinstance(row.controls[0], ft.TextField)
                        else None
                    )
                    assistant_tf = (
                        row.controls[1]
                        if isinstance(row.controls[1], ft.TextField)
                        else None
                    )
                    if user_tf and assistant_tf:
                        user_text = user_tf.value or ""
                        assistant_text = assistant_tf.value or ""
                        # Only save if at least one field has text
                        if user_text or assistant_text:
                            examples_list.append(
                                {"user": user_text, "assistant": assistant_text}
                            )
                        else:
                            logger.debug(
                                "Skipping empty few-shot example row during save."
                            )
                    else:
                        logger.warning(
                            f"Unexpected control types in few-shot row during save: {row.controls}"
                        )
        else:
            logger.warning(
                "Few-shot examples column control not found or invalid during save."
            )

        logger.debug(f"Saving {len(examples_list)} few-shot examples.")
        # Ensure llm key exists before adding examples
        if "llm" not in new_config_data or not isinstance(
            new_config_data.get("llm"), dict
        ):
            new_config_data["llm"] = {}
        update_nested_dict(new_config_data, "llm.few_shot_examples", examples_list)

        # Directly update the singleton's internal data BEFORE saving
        # This makes the changes live immediately, even before file write
        config_instance._config_data = new_config_data
        # Recalculate derived values like logging level int after update
        log_level_str = new_config_data.get("logging", {}).get("level", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        # Ensure logging dict exists before setting level_int
        if "logging" not in config_instance._config_data:
            config_instance._config_data["logging"] = {}
        config_instance._config_data["logging"]["level_int"] = log_level
        logger.debug(
            f"Updated live config data in memory: {config_instance._config_data}"
        )

        # Call the save method on the config instance (runs in thread)
        await asyncio.to_thread(config_instance.save)

        # Show success banner
        gui_utils.show_success_banner(page, "配置已成功保存到 config.yaml")

    except Exception as ex:
        error_msg = f"保存配置时出错: {ex}"
        logger.critical(error_msg, exc_info=True)
        # Show error banner
        gui_utils.show_error_banner(page, error_msg)


def reload_config_controls(
    page: ft.Page,  # Need page for update
    all_config_controls: Dict[str, ft.Control],  # Need controls dict
    config_instance: "Config",  # Need config instance
    # Need specific function ref for creating rows, including its remove handler logic
    create_example_row_func: Callable[[str, str], ft.Row],
):
    """Updates the GUI controls with values from the reloaded config."""
    logger.info("Reloading config values into GUI controls.")
    if not config_instance:
        logger.error("Cannot reload controls, config instance not available.")
        gui_utils.show_error_banner(page, "无法重载控件：配置对象不可用。")
        return
    reloaded_config_data = config_instance.data  # Get reloaded data

    # Use the aggregated dictionary of controls
    for key, control in all_config_controls.items():
        # Skip special controls that don't map directly to config keys
        if key in ["llm.few_shot_examples_column", "llm.add_example_button"]:
            continue

        # Get value from the *reloaded* data using nested access if needed
        keys = key.split(".")
        current_value = reloaded_config_data
        valid_path = True
        for k in keys:
            try:
                # Check if current_value is a dictionary before indexing
                if isinstance(current_value, dict):
                    current_value = current_value[k]
                else:
                    logger.debug(
                        f"Config path for '{key}' invalid at '{k}': parent is not a dictionary. Skipping reload for this control."
                    )
                    current_value = None  # Indicate value not found
                    valid_path = False
                    break
            except (KeyError, TypeError, IndexError):
                logger.debug(
                    f"Key '{key}' path invalid or key missing at '{k}' in reloaded config data. Skipping reload for this control."
                )
                current_value = None  # Indicate value not found
                valid_path = False
                break

        if not valid_path:
            continue  # Skip update for this control if path was invalid

        # Assign the final retrieved value (or None if path was invalid/key missing)
        value = current_value

        try:
            if control is None:  # Should not happen if loop continues, but check anyway
                logger.debug(
                    f"Skipping reload for key '{key}' as control is None (already checked?)."
                )
                continue

            # --- Update control based on type ---
            if isinstance(control, ft.Switch):
                control.value = bool(value) if value is not None else False
            elif isinstance(control, ft.Dropdown):
                # Ensure the value exists in options before setting
                if (
                    value is not None
                    and hasattr(control, "options")
                    and isinstance(control.options, list)
                    and any(opt.key == value for opt in control.options)
                ):
                    control.value = value
                else:
                    # If value from config is invalid for dropdown, log and keep current value
                    if value is not None:  # Log only if there was a value expected
                        logger.warning(
                            f"Value '{value}' for dropdown '{key}' not in options or invalid. Keeping previous selection: {control.value}"
                        )
                    # Optionally set to None or a default if value is invalid? For now, keep existing.
                    # control.value = None # Or some default?
            elif isinstance(control, ft.Dropdown) and key == "audio.device":
                 # Special handling for device dropdown reload
                 # Refresh options in case devices changed? No, keep it simple for now.
                 # Just set the value if it exists in the current options.
                 current_options = control.options or []
                 if value is not None and any(opt.key == value for opt in current_options):
                     control.value = value
                 elif value == "Default": # Always allow Default
                     control.value = "Default"
                 else:
                     logger.warning(f"Reload: Configured audio device '{value}' not found in dropdown options. Setting to Default.")
                     control.value = "Default" # Fallback if saved device not in list
            elif isinstance(control, ft.TextField):
                # Handle keys where None should be represented as empty string
                if (
                    key
                    in [
                        "dashscope.stt.translation_target_language",
                        "audio.sample_rate",
                        "llm.base_url",
                    ]
                    and value is None
                ):
                    control.value = ""  # Use empty string for None
                else:
                    # Ensure value is converted to string for TextField
                    control.value = str(value) if value is not None else ""
            # Add other control types if necessary (e.g., Slider)
            else:
                logger.debug(
                    f"Control for key '{key}' has unhandled type '{type(control)}' during reload."
                )

        except Exception as ex:
            logger.error(
                f"Error reloading control for key '{key}' with value '{value}': {ex}",
                exc_info=True,
            )

    # --- Reload few-shot examples ---
    few_shot_column = all_config_controls.get("llm.few_shot_examples_column")
    if few_shot_column and isinstance(few_shot_column, ft.Column):
        few_shot_column.controls.clear()  # Remove existing rows first
        # Safely get examples from reloaded data
        loaded_examples = reloaded_config_data.get("llm", {}).get(
            "few_shot_examples", []
        )
        if isinstance(loaded_examples, list):
            logger.info(f"Reloading {len(loaded_examples)} few-shot examples into GUI.")
            for example in loaded_examples:
                if (
                    isinstance(example, dict)
                    and "user" in example
                    and "assistant" in example
                ):
                    # Call the passed-in function to create the row.
                    # This function must handle setting up the remove handler correctly.
                    try:
                        new_row = create_example_row_func(
                            example.get("user", ""), example.get("assistant", "")
                        )
                        few_shot_column.controls.append(new_row)
                    except Exception as row_ex:
                        logger.error(
                            f"Error creating few-shot row during reload for example {example}: {row_ex}",
                            exc_info=True,
                        )
                else:
                    logger.warning(
                        f"Skipping invalid few-shot example during reload: {example}"
                    )
        else:
            logger.warning(
                "'llm.few_shot_examples' in reloaded config is not a list. Cannot reload examples."
            )
    elif few_shot_column:
        logger.error("Few-shot column control is not a Column. Cannot reload examples.")
    else:
        logger.warning("Few-shot column control not found. Cannot reload examples.")

    # Update the page to show changes
    try:
        if page and page.controls:
            page.update()
        elif page:
            logger.warning(
                "Page has no controls, skipping final update in reload_config_controls."
            )
    except Exception as e:
        logger.error(
            f"Error during final page.update in reload_config_controls: {e}",
            exc_info=True,
        )


async def reload_config_handler(
    page: ft.Page,  # Need page for banner & update
    all_config_controls: Dict[str, ft.Control],  # Need controls dict
    config_instance: "Config",  # Need config instance
    create_example_row_func: Callable,  # Need row creation func
):
    """Reloads configuration from file and updates the GUI."""
    logger.info("Reload configuration button clicked.")
    if not config_instance:
        logger.error("Cannot reload, config instance not available.")
        gui_utils.show_error_banner(page, "无法重载配置：配置对象不可用。")
        return
    try:
        # Run synchronous reload in thread
        await asyncio.to_thread(config_instance.reload)
        # Update GUI fields with new values from the reloaded config_instance.data
        reload_config_controls(
            page, all_config_controls, config_instance, create_example_row_func
        )
        # Show success banner
        gui_utils.show_success_banner(page, "配置已从 config.yaml 重新加载")

    except Exception as ex:
        error_msg = f"重新加载配置时出错: {ex}"
        logger.error(error_msg, exc_info=True)
        # Show error banner
        gui_utils.show_error_banner(page, error_msg)


# --- Few-Shot Example Add/Remove Logic ---


# This function now needs page and the column reference passed in.
# It defines the remove handler internally, capturing the necessary scope.
def create_config_example_row(
    page: ft.Page,  # Need page for update
    few_shot_column: ft.Column,  # Need column ref to remove from
    user_text: str = "",
    assistant_text: str = "",
) -> ft.Row:
    """Creates a Flet Row for a single few-shot example with its remove handler."""
    # Create controls for the row
    user_input = ft.TextField(
        label="用户输入 (User)",
        value=user_text,
        multiline=True,
        max_lines=3,
        expand=True,
    )
    assistant_output = ft.TextField(
        label="助手响应 (Assistant)",
        value=assistant_text,
        multiline=True,
        max_lines=3,
        expand=True,
    )

    # Define remove handler within this scope to capture page and column
    async def remove_this_row(e_remove: ft.ControlEvent):
        row_to_remove = e_remove.control.data  # Get the Row associated with the button
        if few_shot_column and row_to_remove in few_shot_column.controls:
            few_shot_column.controls.remove(row_to_remove)
            logger.debug("Removed few-shot example row.")
            try:
                if page and page.controls:
                    page.update()
                elif page:
                    logger.warning(
                        "Page has no controls, skipping update after removing few-shot row."
                    )
            except Exception as e:
                logger.error(
                    f"Error updating page after removing few-shot row: {e}",
                    exc_info=True,
                )
        else:
            logger.warning(
                "Attempted to remove a row not found in the column or column is invalid."
            )

    remove_button = ft.IconButton(
        icon=ft.icons.DELETE_OUTLINE,
        tooltip="删除此示例",
        on_click=remove_this_row,  # Use the handler defined above
        icon_color=ft.colors.RED_ACCENT_400,
    )

    new_row = ft.Row(
        controls=[user_input, assistant_output, remove_button],
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        vertical_alignment=ft.CrossAxisAlignment.START,
    )
    remove_button.data = new_row  # Associate the row with the button for removal
    return new_row


async def add_example_handler(
    page: ft.Page,  # Need page for update and row creation
    all_config_controls: Dict[str, ft.Control],  # Need controls dict to find column
):
    """Adds a new, empty example row to the column."""
    few_shot_column = all_config_controls.get("llm.few_shot_examples_column")
    if few_shot_column and isinstance(few_shot_column, ft.Column):
        # Pass page and column ref to the internal row creation function
        try:
            new_row = create_config_example_row(page, few_shot_column) # Create row with handler
            few_shot_column.controls.append(new_row)
            logger.debug("Added new few-shot example row.")
            if page and page.controls:
                page.update()
            elif page:
                 logger.warning("Page has no controls, skipping update after adding few-shot row.")
        except Exception as e:
             logger.error(f"Error adding or updating page for new few-shot row: {e}", exc_info=True)
             gui_utils.show_error_banner(page, f"添加示例时出错: {e}")

    else:
        logger.error("Could not add few-shot example row: Column control not found or invalid.")
        gui_utils.show_error_banner(page, "无法添加示例：UI 元素丢失。")
