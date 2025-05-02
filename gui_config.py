import flet as ft
from typing import Dict, Any, Optional, Callable, TYPE_CHECKING
import logging
import asyncio
from openai import AsyncOpenAI, APIConnectionError, AuthenticationError, OpenAIError # Added OpenAI imports
import copy
import os # Import os for startfile
import subprocess # Import subprocess for cross-platform opening
import sys # Import sys for platform check
import gui_utils  # Import for close_banner
from audio_recorder import get_input_devices  # Import device getter
from typing import (
    Awaitable,
)  # Added Awaitable


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


def create_dashscope_controls(
    initial_config: Dict[str, Any],
    save_callback: Optional[Callable[[ft.ControlEvent], Awaitable[None]]] = None, # Add save callback param
) -> Dict[str, ft.Control]:
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
        on_change=save_callback, # Assign callback
    )

    # --- Dynamically create STT Model Dropdown ---
    stt_models_config = stt_conf.get("models", {})
    model_options = []
    selected_model_value = stt_conf.get("selected_model", None) # Get configured selected model

    if not stt_models_config:
        logger.warning("No STT models defined in config 'dashscope.stt.models'. Dropdown will be empty.")
        model_options.append(ft.dropdown.Option(key="error", text="配置中未定义模型", disabled=True))
        selected_model_value = "error" # Set value to error state
    else:
        for model_name, model_info in stt_models_config.items():
            # Create display text, optionally add translation support info
            display_text = model_name
            supports_translation = model_info.get("supports_translation", False)
            model_type = model_info.get("type", "unknown") # Get type for potential display
            # Example: "gummy-realtime-v1 (Gummy, 支持翻译)"
            display_text += f" ({model_type.capitalize()}"
            if supports_translation:
                display_text += ", 支持翻译"
            display_text += ")"

            model_options.append(ft.dropdown.Option(key=model_name, text=display_text))

        # Validate if the configured selected_model exists in the options
        if selected_model_value not in stt_models_config:
             logger.warning(f"Configured 'selected_model' ('{selected_model_value}') not found in 'models' list. Falling back.")
             # Fallback to the first available model key or None if empty
             selected_model_value = next(iter(stt_models_config.keys()), None)
             if selected_model_value is None:
                 logger.error("Cannot select a fallback model, models list is empty.")
                 model_options.append(ft.dropdown.Option(key="error", text="无可用模型", disabled=True))
                 selected_model_value = "error"


    # Use 'selected_model' as the key to match config structure
    controls["dashscope.stt.selected_model"] = ft.Dropdown(
        label="STT 模型",
        value=selected_model_value, # Use validated or fallback value
        options=model_options,
        tooltip="选择要使用的 Dashscope STT 模型 (来自 config.yaml)",
        on_change=save_callback, # Assign callback
    )
    # --- End STT Model Dropdown ---

    controls["dashscope.stt.translation_target_language"] = ft.TextField(
        label="翻译目标语言 (若模型支持)", # Clarify dependency
        value=stt_conf.get("translation_target_language") or "",  # Handle None
        hint_text="留空则禁用翻译 (例如: en, ja, ko)",
        tooltip="如果使用 Gummy 并希望翻译，在此处输入目标语言代码",
        on_change=save_callback, # Assign callback
    )
    controls["dashscope.stt.intermediate_result_behavior"] = ft.Dropdown(
        label="中间结果处理 (VRC OSC)",
        on_change=save_callback, # Assign callback
        value=stt_conf.get("intermediate_result_behavior", "ignore"),
        options=[
            ft.dropdown.Option("ignore", "忽略"),
            ft.dropdown.Option("show_typing", "显示 'Typing...'"),
            ft.dropdown.Option("show_partial", "显示部分文本"),
        ],
        tooltip="如何处理非最终的语音识别结果 (仅影响 VRChat 输出)",
    )
    return controls


def create_audio_controls(
    initial_config: Dict[str, Any],
    save_callback: Optional[Callable[[ft.ControlEvent], Awaitable[None]]] = None, # Add save callback param
) -> Dict[str, ft.Control]:
    """Creates controls for the Audio Input section."""
    controls = {}
    audio_conf = initial_config.get("audio", {})

    # --- Microphone Selection Dropdown ---
    device_options = []
    selected_device_value = audio_conf.get("device", "Default")  # Get configured value
    try:
        available_devices = get_input_devices()
        # Add "Default" option first
        device_options.append(
            ft.dropdown.Option(key="Default", text="Default Input Device")
        )
        # Add other devices
        for device in available_devices:
            if (
                device.get("name") == "Error querying devices"
            ):  # Handle error case from get_input_devices
                device_options.append(
                    ft.dropdown.Option(
                        key="error", text="Error querying devices", disabled=True
                    )
                )
                continue
            # Use the user-friendly name for display and also as the key to store in config
            device_name = device.get("name", "Unknown Device")
            option_text = f"{device_name}"
            if device.get("is_default"):
                option_text += (
                    " (System Default)"  # Indicate which one is the system default
                )

            device_options.append(ft.dropdown.Option(key=device_name, text=option_text))

        # Ensure the configured value is actually in the options list
        if selected_device_value != "Default" and not any(
            opt.key == selected_device_value for opt in device_options
        ):
            logger.warning(
                f"Configured audio device '{selected_device_value}' not found in available devices. Falling back to Default."
            )
            # Add a temporary option for the missing device? Or just fallback? Fallback is safer.
            # device_options.append(ft.dropdown.Option(key=selected_device_value, text=f"{selected_device_value} (Not Found)", disabled=True))
            selected_device_value = "Default"  # Reset to default if not found

    except Exception as e:
        logger.error(f"Failed to populate microphone list: {e}", exc_info=True)
        device_options.append(
            ft.dropdown.Option(key="error", text="Error loading devices", disabled=True)
        )
        selected_device_value = "Default"  # Fallback

    controls["audio.device"] = ft.Dropdown(
        label="麦克风 (输入设备)",
        value=selected_device_value,
        options=device_options,
        tooltip="选择要使用的音频输入设备",
        on_change=save_callback, # Assign callback
    )
    # --- End Microphone Selection ---

    # REMOVED: Sample rate is now determined by the selected STT model
    # controls["audio.sample_rate"] = ft.TextField(...)

    controls["audio.channels"] = ft.TextField(
        label="声道数 (固定)", # Indicate fixed value
        value=str(audio_conf.get("channels", 1)),
        keyboard_type=ft.KeyboardType.NUMBER,
        tooltip="音频输入声道数 (由代码固定为 1)",
        disabled=True, # Make read-only
    )
    controls["audio.dtype"] = ft.TextField(
        label="数据类型 (固定)", # Indicate fixed value
        value=audio_conf.get("dtype", "int16"),
        tooltip="音频数据类型 (由代码固定为 int16)",
        disabled=True, # Make read-only
    )
    controls["audio.debug_echo_mode"] = ft.Switch(
        label="调试回声模式",
        value=audio_conf.get("debug_echo_mode", False),
        tooltip="将输入音频直接路由到输出以进行测试",
        on_change=save_callback, # Assign callback
    )
    return controls


def create_llm_controls(
    initial_config: Dict[str, Any],
    save_callback: Optional[Callable[[ft.ControlEvent], Awaitable[None]]] = None, # Add save callback param
) -> Dict[str, ft.Control]:
    """Creates controls for the LLM section."""
    controls = {}
    llm_conf = initial_config.get("llm", {})

    # --- Basic LLM Settings (API Key, Model, Temp, etc.) ---
    controls["llm.enabled"] = ft.Switch(
        label="启用 LLM 处理",
        value=llm_conf.get("enabled", False),
        tooltip="是否将识别/翻译后的文本发送给 LLM 进行处理",
        on_change=save_callback, # Assign callback
    )
    controls["llm.api_key"] = ft.TextField(
        label="API Key",
        on_change=save_callback, # Assign callback
        value=llm_conf.get("api_key", ""),
        password=True,
        can_reveal_password=True,
        # on_change=save_callback, # Already added above
        hint_text="从环境变量 OPENAI_API_KEY 覆盖",
        tooltip="用于 LLM 处理的 OpenAI 兼容 API Key (如果启用)",
    )
    controls["llm.base_url"] = ft.TextField(
        label="LLM API Base URL",
        value=llm_conf.get("base_url") or "",
        hint_text="留空则使用 OpenAI 默认 URL",
        tooltip="用于本地 LLM 或代理 (例如 http://localhost:11434/v1)",
        on_change=save_callback, # Assign callback
    )
    # --- LLM Model Dropdown and Refresh Button ---
    initial_model = llm_conf.get("model", "gpt-3.5-turbo") # Get initial value
    controls["llm.model_dropdown"] = ft.Dropdown(
        label="LLM 模型",
        value=initial_model, # Set initial value
        options=[
            # Start with only the configured model as an option
            # User needs to refresh to see others
            ft.dropdown.Option(key=initial_model, text=initial_model)
        ],
        tooltip="选择要使用的 OpenAI 兼容模型 (点击右侧按钮刷新列表)",
        expand=True, # Allow dropdown to expand in Row
        on_change=save_callback, # Assign callback
    )
    controls["llm.model_refresh_button"] = ft.IconButton(
        icon=ft.icons.REFRESH,
        tooltip="刷新 LLM 模型列表 (需要 API Key 和 Base URL)",
        on_click=None, # Handler assigned later in gui.py or via partial
    )
    # --- End LLM Model Dropdown ---

    # REMOVED: System Prompt TextField
    # controls["llm.system_prompt"] = ft.TextField(...)

    # REMOVED: Few-Shot Examples Label
    # controls["llm.few_shot_examples_label"] = ft.Text(...)

    controls["llm.temperature"] = ft.TextField(
        label="LLM Temperature",
        value=str(llm_conf.get("temperature", 0.7)),
        keyboard_type=ft.KeyboardType.NUMBER,
        tooltip="控制 LLM 输出的随机性 (0.0-2.0)",
        on_change=save_callback, # Assign callback
    )
    controls["llm.max_tokens"] = ft.TextField(
        label="LLM Max Tokens",
        on_change=save_callback, # Assign callback
        value=str(llm_conf.get("max_tokens", 150)),
        keyboard_type=ft.KeyboardType.NUMBER,
        tooltip="LLM 响应的最大长度",
    )
    # --- Add extract/marker controls here ---
    controls["llm.extract_final_answer"] = ft.Switch(
        label="提取最终答案",
        value=llm_conf.get("extract_final_answer", False),
        tooltip="尝试从 LLM 输出中提取标记后的最终答案",
        on_change=save_callback, # Assign callback
    )
    controls["llm.final_answer_marker"] = ft.TextField(
        label="最终答案标记",
        on_change=save_callback, # Assign callback
        value=llm_conf.get("final_answer_marker", "Final Answer:"),
        tooltip="用于标识最终答案开始的文本 (仅当 '提取最终答案' 启用时)", # Clarify tooltip
    )

    # --- Preset Management (REMOVED from here, moved to Preset Tab) ---
    # controls["llm.active_preset_name_label"] = ft.Text(...)
    # controls["llm.manage_presets_button"] = ft.ElevatedButton(...)

    # --- Prompt Display/Editing Area (REMOVED from Config Tab) ---
    # REMOVED: Few-shot examples UI elements placeholders
    # controls["llm.few_shot_examples_column"] = ft.Column(...)
    # controls["llm.add_example_button"] = ft.TextButton(...)

    return controls


# --- New Helper Function to Update LLM UI Section ---
def update_llm_config_ui(
    page: ft.Page, # Need page for update
    all_config_controls: Dict[str, ft.Control], # Config tab controls
    # REMOVED: system_prompt_value: str,
    # REMOVED: few_shot_examples_list: List[Dict[str, str]],
    active_preset_name_label_ctrl: ft.Text, # Pass the label control from Preset Tab
    active_preset_name_value: str,
    # REMOVED: create_example_row_func: Callable[[str, str], ft.Row],
) -> None:
    """Updates the LLM Active Preset Label in the Preset Tab."""
    logger.debug(f"Updating LLM active preset label (Preset Tab) for preset: '{active_preset_name_value}'")

    # REMOVED: Update System Prompt TextField logic
    # system_prompt_tf = all_config_controls.get("llm.system_prompt")
    # if isinstance(system_prompt_tf, ft.TextField): ...

    # Update Active Preset Name Label (in the Preset Tab) - This remains
    if isinstance(active_preset_name_label_ctrl, ft.Text):
        active_preset_name_label_ctrl.value = f"当前活动预设: {active_preset_name_value}"
        # REMOVED: active_preset_name_label_ctrl.update() - Rely on page.update() below
    else:
        logger.error("Active preset name label control (from Preset Tab) is invalid.")


    # REMOVED: Update Few-Shot Examples Column logic
    # few_shot_column = all_config_controls.get("llm.few_shot_examples_column")
    # if isinstance(few_shot_column, ft.Column): ...

    # Update the page to reflect changes (only the label update needs this now)
    page.update()


def create_vrc_osc_controls(
    initial_config: Dict[str, Any],
    save_callback: Optional[Callable[[ft.ControlEvent], Awaitable[None]]] = None, # Add save callback param
) -> Dict[str, ft.Control]:
    """Creates controls for the VRC OSC Output section."""
    controls = {}
    output_conf = initial_config.get("outputs", {})
    vrc_conf = output_conf.get("vrc_osc", {})
    controls["outputs.vrc_osc.enabled"] = ft.Switch(
        label="启用 VRChat OSC 输出",
        value=vrc_conf.get("enabled", True),
        on_change=save_callback, # Assign callback
    )
    controls["outputs.vrc_osc.address"] = ft.TextField(
        label="VRC OSC 地址",
        value=vrc_conf.get("address", "127.0.0.1"),
        on_change=save_callback, # Assign callback
    )
    controls["outputs.vrc_osc.port"] = ft.TextField(
        label="VRC OSC 端口",
        on_change=save_callback, # Assign callback
        value=str(vrc_conf.get("port", 9000)),
        keyboard_type=ft.KeyboardType.NUMBER,
    )
    controls["outputs.vrc_osc.message_interval"] = ft.TextField(
        label="VRC OSC 消息间隔 (秒)",
        value=str(vrc_conf.get("message_interval", 1.333)),
        keyboard_type=ft.KeyboardType.NUMBER,
        tooltip="发送到 VRChat 的最小时间间隔",
        on_change=save_callback, # Assign callback
    )
    controls["outputs.vrc_osc.format"] = ft.TextField(
        label="VRC OSC 消息格式",
        on_change=save_callback, # Assign callback
        value=vrc_conf.get("format", "{text}"),  # Default format from example
        tooltip="发送到 VRChat 的消息格式。可用占位符: {text}",
    )
    return controls


def create_console_output_controls(
    initial_config: Dict[str, Any],
    save_callback: Optional[Callable[[ft.ControlEvent], Awaitable[None]]] = None, # Add save callback param
) -> Dict[str, ft.Control]:
    """Creates controls for the Console Output section."""
    controls = {}
    output_conf = initial_config.get("outputs", {})
    console_conf = output_conf.get("console", {})
    controls["outputs.console.enabled"] = ft.Switch(
        label="启用控制台输出",
        value=console_conf.get("enabled", True),
        on_change=save_callback, # Assign callback
    )
    controls["outputs.console.prefix"] = ft.TextField(
        label="控制台输出前缀",
        value=console_conf.get("prefix", "[Final Text]"),
        on_change=save_callback, # Assign callback
    )
    return controls


def create_file_output_controls(
    initial_config: Dict[str, Any],
    save_callback: Optional[Callable[[ft.ControlEvent], Awaitable[None]]] = None, # Add save callback param
) -> Dict[str, ft.Control]:
    """Creates controls for the File Output section."""
    controls = {}
    output_conf = initial_config.get("outputs", {})
    file_conf = output_conf.get("file", {})
    controls["outputs.file.enabled"] = ft.Switch(
        label="启用文件输出",
        value=file_conf.get("enabled", False),
        on_change=save_callback, # Assign callback
    )
    controls["outputs.file.path"] = ft.TextField(
        label="文件输出路径",
        value=file_conf.get("path", "output_log.txt"),
        on_change=save_callback, # Assign callback
    )
    controls["outputs.file.format"] = ft.TextField(
        label="文件输出格式",
        on_change=save_callback, # Assign callback
        value=file_conf.get("format", "{timestamp} - {text}"),
        tooltip="可用占位符: {timestamp}, {text}",
    )
    return controls


def create_logging_controls(
    initial_config: Dict[str, Any],
    save_callback: Optional[Callable[[ft.ControlEvent], Awaitable[None]]] = None, # Add save callback param
) -> Dict[str, ft.Control]:
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
        on_change=save_callback, # Assign callback
    )
    # Add controls for application log file
    log_file_conf = log_conf.get("file", {})  # Get file sub-dict
    controls["logging.file.enabled"] = ft.Switch(
        label="启用应用程序日志文件",
        value=log_file_conf.get("enabled", False),
        tooltip="将应用程序的详细运行日志 (DEBUG, INFO, ERROR 等) 输出到文件",
        on_change=save_callback, # Assign callback
    )
    controls["logging.file.path"] = ft.TextField(
        label="应用程序日志文件路径",
        value=log_file_conf.get("path", "vrcmeow_app.log"),
        tooltip="指定保存应用程序运行日志的文件路径",
        on_change=save_callback, # Assign callback
    )
    return controls


# --- Configuration Layout Function ---
# Takes created controls and buttons as arguments


def create_config_tab_content(
    # REMOVED: save_button parameter
    reload_button: ft.ElevatedButton,
    open_folder_button: ft.ElevatedButton, # Change type to ElevatedButton
    all_controls: Dict[str, ft.Control],  # Pass the complete dictionary
) -> ft.Column:
    """Creates the layout Column for the Configuration tab."""

    # Helper to get control by key, avoids KeyError
    def get_ctrl(key: str) -> Optional[ft.Control]:
        ctrl = all_controls.get(key)
        if ctrl is None:
            logger.warning(f"Control '{key}' not found for config layout.")
        return ctrl

    # REMOVED: Extract few-shot elements
    # few_shot_column = ...
    # add_example_btn = ...

    # Filter None controls before passing to section
    dashscope_controls = [
        c for c in [
            get_ctrl("dashscope.api_key"),
            ft.Divider(height=5),
            get_ctrl("dashscope.stt.selected_model"), # Use updated key
            get_ctrl("dashscope.stt.translation_target_language"),
            get_ctrl("dashscope.stt.intermediate_result_behavior"),
            ft.Divider(height=5),
            ft.Text("音频输入", style=ft.TextThemeStyle.TITLE_SMALL), # Keep static text
            get_ctrl("audio.device"),  # Add device dropdown here
            # REMOVED: get_ctrl("audio.sample_rate"),
            get_ctrl("audio.channels"), # Keep disabled field
            get_ctrl("audio.dtype"), # Keep disabled field
            get_ctrl("audio.debug_echo_mode"),
        ] if c is not None # Filter out None values
    ]
    dashscope_section = create_config_section(
        "Dashscope 设置",
        dashscope_controls,
    )

    # Filter None controls before passing to section
    llm_model_row = ft.Row(
        [
            get_ctrl("llm.model_dropdown"), # Use dropdown
            get_ctrl("llm.model_refresh_button"), # Add refresh button
        ],
        alignment=ft.MainAxisAlignment.START,
        vertical_alignment=ft.CrossAxisAlignment.CENTER, # Align items vertically
    )

    llm_controls = [
        c for c in [
            get_ctrl("llm.enabled"),
            get_ctrl("llm.api_key"),
            get_ctrl("llm.base_url"),
            llm_model_row, # Add the row containing dropdown and button
            # REMOVED: get_ctrl("llm.system_prompt"),
            get_ctrl("llm.temperature"),
            get_ctrl("llm.max_tokens"),
            get_ctrl("llm.extract_final_answer"), # Add missing controls
            get_ctrl("llm.final_answer_marker"), # Add missing controls
            # REMOVED: Divider, Preset management row, Few-shot label, column, button
            # ft.Divider(height=10),
            # get_ctrl("llm.few_shot_examples_label"),
            # few_shot_column,
            # add_example_btn,
        ] if c is not None # Filter out None values from the main list
    ]
    llm_section = create_config_section(
        "语言模型 (LLM)",
        llm_controls,
    )

    # Filter None controls before passing to section
    vrc_osc_controls = [
        c for c in [
            get_ctrl("outputs.vrc_osc.enabled"),
            get_ctrl("outputs.vrc_osc.address"),
            get_ctrl("outputs.vrc_osc.port"),
            get_ctrl("outputs.vrc_osc.message_interval"),
            get_ctrl("outputs.vrc_osc.format"),  # Add the format control here
        ] if c is not None # Filter out None values
    ]
    vrc_osc_output_section = create_config_section(
        "输出: VRChat OSC",
        vrc_osc_controls,
    )

    # Filter None controls before passing to section
    console_controls = [
        c for c in [
            get_ctrl("outputs.console.enabled"),
            get_ctrl("outputs.console.prefix"),
        ] if c is not None # Filter out None values
    ]
    console_output_section = create_config_section(
        "输出: 控制台",
        console_controls,
    )

    # Filter None controls before passing to section
    file_controls = [
        c for c in [
            get_ctrl("outputs.file.enabled"),
            get_ctrl("outputs.file.path"),
            get_ctrl("outputs.file.format"),
        ] if c is not None # Filter out None values
    ]
    file_output_section = create_config_section(
        "输出: 文件",
        file_controls,
    )

    # Filter None controls before passing to section
    logging_controls = [
        c for c in [
            get_ctrl("logging.level"),
            ft.Divider(height=5),
            get_ctrl("logging.file.enabled"),  # Add file logging enabled switch
            get_ctrl("logging.file.path"),  # Add file logging path field
        ] if c is not None # Filter out None values
    ]
    logging_section = create_config_section(
        "日志记录",
        logging_controls,
    )

    # --- Create the top button row (reload and open folder buttons) ---
    button_row = ft.Row(
        [
            open_folder_button, # Use the ElevatedButton directly
            reload_button,      # Place reload button next to it
        ],
        alignment=ft.MainAxisAlignment.END, # Align buttons to the right
        spacing=10, # Add spacing between buttons
    )

    # --- Create the scrollable column for config sections ---
    # Filter out None sections before adding to the scrollable column
    scrollable_sections = [
        ctrl
        for ctrl in [
            dashscope_section,
            llm_section,
            vrc_osc_output_section,
            console_output_section,
            file_output_section,
            logging_section,
        ]
        if ctrl is not None
    ]

    # Ensure sections are valid before adding
    valid_scrollable_sections = [c for c in scrollable_sections if c is not None]
    if len(valid_scrollable_sections) != len(scrollable_sections):
        logger.warning("Some config section controls were None and excluded.")

    scrollable_column = ft.Column(
        controls=valid_scrollable_sections,
        expand=True,  # Make this column take available vertical space
        scroll=ft.ScrollMode.ADAPTIVE,
        spacing=15,
    )

    # --- Combine the fixed button row and the scrollable column ---
    return ft.Column(
        controls=[
            button_row,
            ft.Divider(height=10),  # Separator between buttons and content
            scrollable_column,
        ],
        expand=True,  # Ensure the outer column also expands
        spacing=0,  # Adjust spacing if needed
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
    # REMOVED: create_example_row_func: Callable,
    dashboard_update_callback: Optional[Callable[[], None]],  # Dashboard update callback
    # Update callback signature (no longer needs prompt/examples)
    update_llm_ui_callback: Callable[[ft.Text, str], None],
    # REMOVED: app_state: "AppState",
    # REMOVED: restart_callback: Callable[[], Awaitable[None]],
    # Add active_preset_name_label control reference (passed from gui.py)
    active_preset_name_label_ctrl: Optional[ft.Text] = None,
    # Add the new callback for text input info
    text_input_info_update_callback: Optional[Callable[[], None]] = None,
    e: Optional[ft.ControlEvent] = None,  # Add optional event argument
):
    """
    保存按钮点击事件处理程序 (配置选项卡)。
    保存成功后将重新加载配置并更新 UI。不再触发重启。
    """
    # Add logging to indicate if triggered by event (auto-save) or manually (if button existed)
    trigger_source = "auto-save (on_change)" if e else "programmatic call"
    logger.info(f"Save configuration triggered by: {trigger_source}.")

    # Prevent saving if called without a valid event object from a control change
    # This helps avoid accidental saves if the function is called incorrectly elsewhere.
    # Allow programmatic calls (e=None) for now, but be mindful.
    # if not e:
    #     logger.warning("save_config_handler called without a ControlEvent. Skipping save.")
    #     return

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
        # Use updated key for selected model
        update_nested_dict(
            new_config_data,
            "dashscope.stt.selected_model",
            get_control_value(
                all_config_controls, "dashscope.stt.selected_model", str, None # Default to None if control missing
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
            "audio.device",  # Save selected device
            get_control_value(all_config_controls, "audio.device", str, "Default"),
        )
        # REMOVED: Sample rate is no longer saved from GUI
        # update_nested_dict(
        #     new_config_data,
        #     "audio.sample_rate",
        #     get_control_value(all_config_controls, "audio.sample_rate", int, None),
        # )
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
        # --- Get model value from Dropdown ---
        update_nested_dict(
            new_config_data,
            "llm.model",
            get_control_value(all_config_controls, "llm.model_dropdown", str, "gpt-3.5-turbo"), # Use dropdown key
        )
        # --- End get model value ---
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
            get_control_value(all_config_controls, "llm.max_tokens", int, 256), # Match default
        )
        # --- Include existing extract/marker logic in search ---
        # Note: The following lines were likely added in a previous step and need to be part of the search
        # update_nested_dict(
        #     new_config_data,
        #     "llm.extract_final_answer",
        #     get_control_value(all_config_controls, "llm.extract_final_answer", bool, False),
        # )
        # update_nested_dict(
        #     new_config_data,
        #     "llm.final_answer_marker",
        #     get_control_value(all_config_controls, "llm.final_answer_marker", str, "Final Answer:"),
        # )
        # --- End existing logic ---

        # --- Save Active Preset Name ---
        # Read the active preset name from the label control in the Preset Tab
        active_preset_name = "Default" # Default if label not found or invalid
        if active_preset_name_label_ctrl and isinstance(active_preset_name_label_ctrl, ft.Text):
             # Extract name after "当前活动预设: "
             label_text = active_preset_name_label_ctrl.value
             prefix = "当前活动预设: "
             if label_text and label_text.startswith(prefix):
                 loaded_preset_name = label_text[len(prefix):].strip()
                 if loaded_preset_name:
                     active_preset_name = loaded_preset_name
                 else:
                     logger.warning("Active preset name label (Preset Tab) is empty after prefix, saving 'Default'.")
             else:
                 logger.warning(f"Active preset name label (Preset Tab) has unexpected format ('{label_text}'), saving 'Default'.")
        else:
             logger.warning("Active preset name label control (from Preset Tab) not found or invalid, saving 'Default'.")

        update_nested_dict(new_config_data, "llm.active_preset_name", active_preset_name)
        logger.debug(f"Saving active_preset_name to config: {active_preset_name}")

        # --- IMPORTANT: Do NOT save system_prompt or few_shot_examples to config.yaml ---
        # These are now managed in prompt_presets.json
        # Remove them from the dict before saving to config.yaml
        if "llm" in new_config_data and isinstance(new_config_data["llm"], dict):
            new_config_data["llm"].pop("system_prompt", None)
            new_config_data["llm"].pop("few_shot_examples", None)
            logger.debug("Removed system_prompt and few_shot_examples from data being saved to config.yaml")


        # REMOVED: Saving few-shot examples from UI (they are not in this tab anymore)
        # examples_list = [] ...
        # update_nested_dict(new_config_data, "llm.few_shot_examples", examples_list)


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
            "outputs.vrc_osc.format",
            get_control_value(
                all_config_controls,
                "outputs.vrc_osc.format",
                str,
                "{text}",  # Match default
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
        update_nested_dict(
            new_config_data,
            "logging.file.enabled",  # Save app log file enabled state
            get_control_value(all_config_controls, "logging.file.enabled", bool, False),
        )
        update_nested_dict(
            new_config_data,
            "logging.file.path",  # Save app log file path
            get_control_value(
                all_config_controls, "logging.file.path", str, "vrcmeow_app.log"
            ),
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
        logger.info("Configuration saved to file.")

        # Reload the configuration from the file into the config object
        logger.info("Reloading configuration from file after saving...")
        await asyncio.to_thread(config_instance.reload)
        logger.info("Configuration reloaded into memory.")

        # Update the GUI controls with the newly saved and reloaded values
        logger.info("Updating GUI controls with reloaded configuration...")
        reload_config_controls(
            page,
            all_config_controls,
            config_instance,
            # REMOVED: create_example_row_func,
            update_llm_ui_callback, # Pass the callback here
            active_preset_name_label_ctrl, # Pass the label control here
        )
        logger.info("GUI controls updated.")

        # Show success banner indicating save and reload
        gui_utils.show_success_banner(page, "配置已保存并重新加载到当前应用")

        # Call the dashboard update callback AFTER saving, reloading, and updating controls
        if dashboard_update_callback:
            logger.info("Calling dashboard update callback after save.")
            try:
                # Ensure the callback uses the *latest* config data
                # The partial in gui.py should handle this if it accesses config.data directly
                # Remove await: update_dashboard_info_display uses page.run_thread internally
                dashboard_update_callback()
            except Exception as cb_ex:
                logger.error(
                    f"Error executing dashboard update callback after save: {cb_ex}",
                    exc_info=True,
                )
        else:
            logger.warning(
                "Dashboard update callback not provided to save_config_handler."
            )

        # Call the text input info update callback
        if text_input_info_update_callback:
            logger.info("Calling text input info update callback after save.")
            try:
                text_input_info_update_callback()
            except Exception as cb_ex:
                logger.error(
                    f"Error executing text input info update callback after save: {cb_ex}",
                    exc_info=True,
                )
        else:
            logger.warning(
                "Text input info update callback not provided to save_config_handler."
            )

        # Show banner and update page (page.update() might be redundant if callbacks update)
        # page.update() # Let individual callbacks handle updates

        # REMOVED: Delay and restart logic

    except Exception as ex:
        error_msg = f"保存或重载配置时出错: {ex}"  # Reverted error message context
        logger.critical(error_msg, exc_info=True)
        # Show error banner
        gui_utils.show_error_banner(page, error_msg)


def reload_config_controls(
    page: ft.Page,  # Need page for update
    all_config_controls: Dict[str, ft.Control],  # Need controls dict
    config_instance: "Config",  # Need config instance
    # REMOVED: create_example_row_func: Callable[[str, str], ft.Row],
    # Update callback signature
    update_llm_ui_callback: Callable[[ft.Text, str], None],
    # Need the label control from the Preset Tab to pass to the callback
    active_preset_name_label_ctrl: Optional[ft.Text] = None,
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
        if key in [
            "llm.few_shot_examples_column",
            "llm.add_example_button",
            "llm.model_refresh_button", # Skip refresh button
            # "llm.model_dropdown", # Handle dropdown separately below
        ]:
            continue

        # Handle dropdown separately
        if key == "llm.model_dropdown":
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
                    logger.warning(  # Changed to WARNING
                        f"Config path for '{key}' invalid at '{k}': parent is not a dictionary. Skipping reload for this control."
                    )
                    current_value = None  # Indicate value not found
                    valid_path = False
                    break
            except (KeyError, TypeError, IndexError):
                logger.warning(  # Changed to WARNING
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
            elif isinstance(control, ft.Dropdown) and key != "llm.model_dropdown": # Exclude LLM model dropdown for now
                # --- Special handling for Dropdowns during reload (excluding LLM model) ---
                current_options = control.options or []
                options_changed = False

                # 1. Reload options for STT model dropdown specifically
                if key == "dashscope.stt.selected_model":
                    new_model_options = []
                    stt_models_config = reloaded_config_data.get("dashscope", {}).get("stt", {}).get("models", {})
                    if not stt_models_config:
                         new_model_options.append(ft.dropdown.Option(key="error", text="配置中未定义模型", disabled=True))
                    else:
                        for model_name, model_info in stt_models_config.items():
                            display_text = model_name
                            supports_translation = model_info.get("supports_translation", False)
                            model_type = model_info.get("type", "unknown")
                            display_text += f" ({model_type.capitalize()}"
                            if supports_translation:
                                display_text += ", 支持翻译"
                            display_text += ")"
                            new_model_options.append(ft.dropdown.Option(key=model_name, text=display_text))
                    # Check if options actually changed before updating
                    if str(control.options) != str(new_model_options): # Simple string comparison
                        logger.info(f"Reloading options for STT model dropdown ('{key}').")
                        control.options = new_model_options
                        options_changed = True

                # 2. Set the value for *any* dropdown (including STT model and audio device)
                # Ensure the value from the reloaded config exists in the (potentially updated) options
                final_options = control.options or [] # Use updated options if they changed
                if value is not None and any(opt.key == value for opt in final_options):
                    control.value = value
                else:
                    # Handle invalid/missing value from config
                    if value is not None: # Log only if a value was expected but invalid
                         logger.warning(
                            f"Value '{value}' for dropdown '{key}' not in available options. Attempting fallback."
                        )
                    # Fallback logic:
                    if key == "audio.device":
                        control.value = "Default" # Specific fallback for audio device
                    elif key == "dashscope.stt.selected_model":
                         # Fallback to first available model in the (potentially new) options
                         control.value = next((opt.key for opt in final_options if not getattr(opt, 'disabled', False)), None)
                         if control.value is None: # If still no valid option (e.g., only error option)
                             control.value = "error" if any(opt.key == "error" for opt in final_options) else None
                    else:
                         # Generic fallback: keep current value or set to None if options changed drastically
                         if not options_changed:
                             logger.warning(f"Keeping previous selection '{control.value}' for '{key}'.")
                         else:
                             control.value = None # Reset if options changed and value invalid

            # REMOVED redundant audio.device handling block
            # elif isinstance(control, ft.Dropdown) and key == "audio.device":
            #     ...
            elif isinstance(control, ft.TextField):
                # Handle keys where None should be represented as empty string
                current_options = control.options or []
                if value is not None and any(
                    opt.key == value for opt in current_options
                ):
                    control.value = value
                elif value == "Default":  # Always allow Default
                    control.value = "Default"
                else:
                    logger.warning(
                        f"Reload: Configured audio device '{value}' not found in dropdown options. Setting to Default."
                    )
                    control.value = "Default"  # Fallback if saved device not in list
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

    # --- REMOVED: Reload few-shot examples ---
    # few_shot_column = all_config_controls.get("llm.few_shot_examples_column")
    # if few_shot_column and isinstance(few_shot_column, ft.Column): ...


    # --- Reload LLM Controls (Basic Settings) ---
    if "llm.enabled" in all_config_controls:
        all_config_controls["llm.enabled"].value = config_instance.get("llm.enabled", False)
    if "llm.api_key" in all_config_controls:
        all_config_controls["llm.api_key"].value = config_instance.get("llm.api_key", "")
    if "llm.base_url" in all_config_controls:
        # Handle None correctly for TextField
        base_url_val = config_instance.get("llm.base_url")
        all_config_controls["llm.base_url"].value = str(base_url_val) if base_url_val is not None else ""
    # --- Reload LLM Model Dropdown ---
    llm_model_dropdown = all_config_controls.get("llm.model_dropdown")
    if llm_model_dropdown and isinstance(llm_model_dropdown, ft.Dropdown):
        saved_model_value = config_instance.get("llm.model", "gpt-3.5-turbo")
        current_options = llm_model_dropdown.options or []
        # Check if the saved value exists in the current options
        if any(opt.key == saved_model_value for opt in current_options):
            llm_model_dropdown.value = saved_model_value
            logger.debug(f"Reload: Set LLM model dropdown value to '{saved_model_value}' (found in options).")
        else:
            # Saved value not in current options. Add it as a single option and select it.
            logger.warning(
                f"Reload: Saved LLM model '{saved_model_value}' not found in current dropdown options. "
                f"Adding it temporarily. Consider refreshing the model list."
            )
            # Add the saved model as the only option (or prepend it)
            llm_model_dropdown.options = [ft.dropdown.Option(key=saved_model_value, text=f"{saved_model_value} (Saved)")] + current_options
            # Select the saved value
            llm_model_dropdown.value = saved_model_value

    if "llm.temperature" in all_config_controls:
        # Handle Slider update
        temp_control = all_config_controls["llm.temperature"]
        if isinstance(temp_control, ft.Slider):
            temp_control.value = config_instance.get("llm.temperature", 0.7)
        elif isinstance(temp_control, ft.TextField): # Fallback if type changed
             temp_control.value = str(config_instance.get("llm.temperature", 0.7))
    if "llm.max_tokens" in all_config_controls:
        all_config_controls["llm.max_tokens"].value = str(config_instance.get("llm.max_tokens", 256))
    if "llm.extract_final_answer" in all_config_controls:
         all_config_controls["llm.extract_final_answer"].value = config_instance.get("llm.extract_final_answer", False)
    if "llm.final_answer_marker" in all_config_controls:
         all_config_controls["llm.final_answer_marker"].value = config_instance.get("llm.final_answer_marker", "Final Answer:")

    # --- Update Active Preset Label using the callback ---
    # Get the active preset name from the reloaded config
    active_preset_name = config_instance.get("llm.active_preset_name", "Default")
    logger.info(f"Reloading config: Active LLM preset name is '{active_preset_name}'.")

    # Call the callback to update the label in the Preset Tab
    if active_preset_name_label_ctrl:
        try:
            # Pass the label control and the name to the callback
            update_llm_ui_callback(
                active_preset_name_label_ctrl, # Pass label control
                active_preset_name, # Pass preset name
            )
            logger.info(f"LLM active preset label updated for preset '{active_preset_name}'.")
        except Exception as ui_update_err:
            logger.error(f"Error calling update_llm_ui_callback during reload: {ui_update_err}", exc_info=True)
    else:
        logger.error("Cannot update LLM UI during reload: Active preset name label control is missing.")

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
    # REMOVED: create_example_row_func: Callable,
    dashboard_update_callback: Optional[Callable[[], None]] = None, # Callback type corrected
    # Add the LLM UI update callback and label control needed by reload_config_controls
    update_llm_ui_callback: Optional[Callable] = None, # Signature changed
    active_preset_name_label_ctrl: Optional[ft.Text] = None,
    # Add the new callback for text input info
    text_input_info_update_callback: Optional[Callable[[], None]] = None,
    e: Optional[ft.ControlEvent] = None,  # Add optional event argument
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
        # Need to pass the active_preset_name_label_ctrl here as well
        # Find the label control (assuming it's passed correctly to this handler's partial)
        # This requires modification in gui.py where the partial is created.
        # For now, assume it's available via a mechanism TBD in gui.py or passed directly.
        # Let's modify the signature to expect it.

        # Placeholder: Need to get active_preset_name_label_ctrl here
        active_preset_label_ctrl_for_reload = None # TODO: Get this from gui.py partial binding
        # Find the label control (assuming it's passed correctly to this handler's partial)
        # This requires modification in gui.py where the partial is created.
        # Let's modify the signature to expect it.
        reload_config_controls(
            page,
            all_config_controls,
            config_instance,
            # REMOVED: create_example_row_func,
            update_llm_ui_callback, # Pass the callback
            active_preset_name_label_ctrl, # Pass the label control
        )
        # Show success banner
        gui_utils.show_success_banner(page, "配置已从 config.yaml 重新加载")

        # Call the dashboard update callback AFTER reloading and updating controls
        if dashboard_update_callback:
            logger.info("Calling dashboard update callback after reload.")
            try:
                # Ensure the callback uses the *latest* config data
                # Remove await: update_dashboard_info_display uses page.run_thread internally
                dashboard_update_callback()
            except Exception as cb_ex:
                logger.error(
                    f"Error executing dashboard update callback after reload: {cb_ex}",
                    exc_info=True,
                )
        else:
            logger.warning(
                "Dashboard update callback not provided to reload_config_handler."
            )

        # Call the text input info update callback
        if text_input_info_update_callback:
            logger.info("Calling text input info update callback after reload.")
            try:
                text_input_info_update_callback()
            except Exception as cb_ex:
                logger.error(
                    f"Error executing text input info update callback after reload: {cb_ex}",
                    exc_info=True,
                )
        else:
            logger.warning(
                "Text input info update callback not provided to reload_config_handler."
            )

        # Update the page once after all changes (banner, controls, dashboard, text input info)
        # page.update() # Let individual callbacks handle updates

    except Exception as ex:
        error_msg = f"重新加载配置时出错: {ex}"
        logger.error(error_msg, exc_info=True)
        # Show error banner
        gui_utils.show_error_banner(page, error_msg)


# --- Handler to Open Config Folder ---
async def open_config_folder_handler(
    page: ft.Page, # Need page for banner
    config_instance: "Config", # Need config to get app_dir
    e: Optional[ft.ControlEvent] = None, # Optional event
):
    """Opens the application's configuration folder in the file explorer."""
    logger.info("Open config folder button clicked.")
    if not config_instance:
        logger.error("Cannot open config folder, config object not available.")
        gui_utils.show_error_banner(page, "无法打开配置文件夹：配置对象不可用。")
        return

    try:
        folder_path = config_instance.app_dir
        logger.info(f"Attempting to open folder: {folder_path}")

        if not folder_path.exists() or not folder_path.is_dir():
             logger.error(f"Configuration folder path does not exist or is not a directory: {folder_path}")
             gui_utils.show_error_banner(page, f"错误：配置文件夹不存在 ({folder_path})")
             return

        # Use platform-specific method to open the folder
        if sys.platform == "win32":
            os.startfile(str(folder_path)) # Use os.startfile on Windows
        elif sys.platform == "darwin": # macOS
            subprocess.Popen(["open", str(folder_path)])
        else: # Linux and other Unix-like
            subprocess.Popen(["xdg-open", str(folder_path)])

        logger.info(f"Successfully requested to open folder: {folder_path}")
        # Optional: Show success banner? Usually not needed for opening a folder.
        # gui_utils.show_success_banner(page, f"已在文件浏览器中打开文件夹: {folder_path}")

    except AttributeError:
         logger.error("Config instance does not have 'app_dir' attribute.")
         gui_utils.show_error_banner(page, "错误：无法获取配置文件夹路径。")
    except Exception as ex:
        error_msg = f"打开配置文件夹时出错: {ex}"
        logger.error(error_msg, exc_info=True)
        gui_utils.show_error_banner(page, error_msg)


# --- New Function to Fetch LLM Models ---
async def fetch_and_update_llm_models_dropdown(
    page: ft.Page,
    all_config_controls: Dict[str, ft.Control],
    e: Optional[ft.ControlEvent] = None, # Add optional event argument
):
    """
    Fetches available LLM models from the configured endpoint and updates the dropdown.
    Accepts an optional ControlEvent argument from Flet's on_click.
    """
    logger.info("Attempting to fetch LLM models (triggered by refresh button)...")

    # Get necessary config values from controls
    api_key_ctrl = all_config_controls.get("llm.api_key")
    base_url_ctrl = all_config_controls.get("llm.base_url")
    model_dropdown_ctrl = all_config_controls.get("llm.model_dropdown")
    refresh_button_ctrl = all_config_controls.get("llm.model_refresh_button")

    if not isinstance(model_dropdown_ctrl, ft.Dropdown):
        logger.error("LLM model dropdown control not found or invalid.")
        gui_utils.show_error_banner(page, "内部错误：找不到模型下拉菜单控件。")
        return

    # Disable button during fetch
    if isinstance(refresh_button_ctrl, ft.IconButton):
        refresh_button_ctrl.disabled = True
        page.update() # Show disabled state

    api_key = getattr(api_key_ctrl, "value", None)
    base_url = getattr(base_url_ctrl, "value", None) or None # Ensure None if empty

    if not api_key:
        logger.warning("Cannot fetch LLM models: API Key is missing.")
        gui_utils.show_error_banner(page, "无法获取模型：缺少 LLM API Key。")
        if isinstance(refresh_button_ctrl, ft.IconButton): # Re-enable button
            refresh_button_ctrl.disabled = False
            page.update()
        return

    try:
        logger.debug(f"Fetching models using Base URL: {base_url}, Key: {'*' * (len(api_key) - 4) + api_key[-4:] if api_key else 'None'}")
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        models_response = await client.models.list()
        models = sorted([model.id for model in models_response.data], key=str.lower) # Sort case-insensitively
        logger.info(f"Successfully fetched {len(models)} LLM models.")

        # Preserve current selection if it exists in the new list
        current_value = model_dropdown_ctrl.value
        new_options = [ft.dropdown.Option(key=model_id, text=model_id) for model_id in models]

        model_dropdown_ctrl.options = new_options
        if current_value in models:
            model_dropdown_ctrl.value = current_value # Keep selection
        else:
            # Current value not in new list, maybe select first or leave blank?
            # Let's select the first one if available
            model_dropdown_ctrl.value = models[0] if models else None
            logger.info(f"Previously selected model '{current_value}' not in fetched list. Selecting '{model_dropdown_ctrl.value}'.")

        gui_utils.show_success_banner(page, f"成功获取 {len(models)} 个 LLM 模型。")

    except AuthenticationError:
        logger.error("LLM Authentication Error: Invalid API Key or Base URL.", exc_info=True)
        gui_utils.show_error_banner(page, "LLM 认证错误：无效的 API Key 或 Base URL。")
        model_dropdown_ctrl.options = [ft.dropdown.Option(key="error", text="认证错误", disabled=True)]
        model_dropdown_ctrl.value = "error"
    except APIConnectionError as e:
        logger.error(f"LLM API Connection Error: Could not connect to {base_url or 'default URL'}.", exc_info=True)
        gui_utils.show_error_banner(page, f"LLM 连接错误：无法连接到 API 端点。 {e}")
        model_dropdown_ctrl.options = [ft.dropdown.Option(key="error", text="连接错误", disabled=True)]
        model_dropdown_ctrl.value = "error"
    except OpenAIError as e: # Catch other OpenAI specific errors
        logger.error(f"OpenAI API Error during model fetch: {e}", exc_info=True)
        gui_utils.show_error_banner(page, f"OpenAI API 错误: {e}")
        model_dropdown_ctrl.options = [ft.dropdown.Option(key="error", text="API 错误", disabled=True)]
        model_dropdown_ctrl.value = "error"
    except Exception as e:
        logger.error(f"Unexpected error fetching LLM models: {e}", exc_info=True)
        gui_utils.show_error_banner(page, f"获取模型时发生意外错误: {e}")
        model_dropdown_ctrl.options = [ft.dropdown.Option(key="error", text="未知错误", disabled=True)]
        model_dropdown_ctrl.value = "error"
    finally:
        # Re-enable button
        if isinstance(refresh_button_ctrl, ft.IconButton):
            refresh_button_ctrl.disabled = False
        page.update() # Update dropdown options, value, and button state


# --- REMOVED: Few-Shot Example Add/Remove Logic ---
# def create_config_example_row(...)
# async def add_example_handler(...)
