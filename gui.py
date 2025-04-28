import asyncio
import flet as ft
from typing import Optional, Dict, Any
import os
import pathlib
import sys
import copy  # Needed for handling config data
import logging  # Add missing import for logging constants

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
    print(
        f"[INFO] Initial CWD '{current_cwd}' seems incorrect (config.yaml not found)."
    )
    print(f"[INFO] Changing CWD to detected project root: '{project_root}'")
    os.chdir(project_root)
    print(f"[INFO] Current CWD after change: '{os.getcwd()}'")
    # 如果需要，将项目根目录添加到 sys.path，以确保模块发现
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"[INFO] Added '{project_root}' to sys.path")
else:
    print(
        f"[INFO] Initial CWD '{current_cwd}' seems correct or config.yaml not found at expected root."
    )
# --- 工作目录设置结束 ---


# --- 现在导入依赖于 CWD 或 sys.path 的项目模块 ---
from config import config  # 导入 Config 类用于类型提示 <- 移到 CWD 设置之后
from logger_config import setup_logging, get_logger
from audio_recorder import AudioManager
from output_dispatcher import OutputDispatcher
from llm_client import LLMClient
from osc_client import VRCClient
import gui_utils # Import the utils module


# 初始化日志记录 (在 Flet 应用启动前)
setup_logging()
logger = get_logger("VRCMeowGUI")


class AppState:
    """简单类，用于在回调函数之间共享状态"""

    def __init__(self):
        self.is_running = False
        self.audio_manager: Optional["AudioManager"] = None # Use quotes for forward refs if needed
        self.output_dispatcher: Optional["OutputDispatcher"] = None
        self.llm_client: Optional["LLMClient"] = None
        self.vrc_client: Optional["VRCClient"] = None


def main(page: ft.Page):
    """Flet GUI 主函数"""
    page.title = "VRCMeow Dashboard"
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    # 设置初始窗口大小 (可选)
    page.window_width = 600
    page.window_height = 450
    page.window_resizable = True  # 允许调整大小
    page.padding = 10  # Add some padding around the page content

    app_state = AppState()  # 创建状态实例

    # --- UI 元素 ---
    # UI elements are now largely defined in gui_dashboard.py and gui_config.py
    # We still need to create the actual controls for the config tab here

    # We use a central dictionary to hold all config controls for easy access in handlers
    all_config_controls: Dict[str, ft.Control] = {}

    # Get initial config data safely
    initial_config_data = {}
    try:
        if config:
             initial_config_data = config.data
        else:
             logger.error("Config object not available during GUI initialization.")
             # Potentially show an error message to the user here?
    except Exception as e:
        logger.error(f"Error accessing config data during GUI initialization: {e}", exc_info=True)


    # Import control creation functions from gui_config
    from gui_config import (
        create_dashscope_controls, create_audio_controls, create_llm_controls,
        create_vrc_osc_controls, create_console_output_controls,
        create_file_output_controls, create_logging_controls,
        config_controls as config_controls_dict # Import the shared dict
    )

    # Create controls for each section using helper functions and initial config
    dash_controls = create_dashscope_controls(initial_config_data)
    all_config_controls.update(dash_controls)

    audio_controls = create_audio_controls(initial_config_data)
    all_config_controls.update(audio_controls)

    llm_controls = create_llm_controls(initial_config_data)
    all_config_controls.update(llm_controls)
    # Extract actual few-shot UI elements (created in create_llm_controls)
    # Use .get() for safety in case of issues during creation
    # Ensure the local variables are assigned correctly
    few_shot_examples_column = all_config_controls.get("llm.few_shot_examples_column", ft.Column())
    add_example_button = all_config_controls.get("llm.add_example_button", ft.TextButton())


    vrc_osc_controls = create_vrc_osc_controls(initial_config_data)
    all_config_controls.update(vrc_osc_controls)

    console_controls = create_console_output_controls(initial_config_data)
    all_config_controls.update(console_controls)

    file_controls = create_file_output_controls(initial_config_data)
    all_config_controls.update(file_controls)

    logging_controls = create_logging_controls(initial_config_data)
    all_config_controls.update(logging_controls)

    # Create Config Save/Reload buttons
    save_config_button = ft.ElevatedButton(
        "保存配置",
        on_click=None, # Handlers assigned later
        icon=ft.icons.SAVE,
        tooltip="将当前设置写入 config.yaml",
    )
    reload_config_button = ft.ElevatedButton(
        "从文件重载",
        on_click=None, # Handlers assigned later
        icon=ft.icons.REFRESH,
        tooltip="放弃当前更改并从 config.yaml 重新加载",
    )

    # --- Create Dashboard UI Elements ---
    # These were previously defined at module level in gui_dashboard.py
    # Default state: Red status, Green button
    status_icon = ft.Icon(name=ft.icons.CIRCLE_OUTLINED, color=ft.colors.RED_ACCENT_700)
    status_label = ft.Text("未启动", selectable=True, color=ft.colors.RED_ACCENT_700)
    status_row = ft.Row(
        [status_icon, status_label],
        alignment=ft.MainAxisAlignment.CENTER,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=5,
    )
    output_text = ft.TextField(
        hint_text="最终输出将显示在这里...",
        multiline=True,
        read_only=True,
        expand=True,
        min_lines=5,
        border_radius=ft.border_radius.all(8),
        border_color=ft.colors.with_opacity(0.5, ft.colors.OUTLINE),
        filled=True,
        bgcolor=ft.colors.with_opacity(0.02, ft.colors.ON_SURFACE),
        content_padding=15,
    )
    toggle_button = ft.IconButton(
        icon=ft.icons.PLAY_ARROW_ROUNDED,
        tooltip="启动",
        on_click=None, # Handler assigned later
        disabled=False,
        icon_size=30,
        style=ft.ButtonStyle(color=ft.colors.GREEN_ACCENT_700),
    )
    progress_indicator = ft.ProgressRing(width=20, height=20, stroke_width=2, visible=False)


    # --- 回调函数 (用于更新 UI 和处理事件) ---
    # These callbacks now correctly reference the elements created above in main's scope
    def update_status_display(message: str, is_running: Optional[bool] = None, is_processing: bool = False):
        """线程安全地更新状态文本和图标"""
        if page:  # 确保页面仍然存在
            def update_ui():
                status_label.value = message # Update text
                # Update icon and text color based on state
                if is_processing:
                    # Transition State (Starting/Stopping) - Amber status, button handled below
                    status_icon.name = ft.icons.HOURGLASS_EMPTY_ROUNDED # Use a processing icon
                    status_icon.color = ft.colors.AMBER_700
                    status_label.color = ft.colors.AMBER_700
                elif is_running is True:
                    # Running State - Green status, Red button
                    status_icon.name = ft.icons.CHECK_CIRCLE_ROUNDED
                    status_icon.color = ft.colors.GREEN_ACCENT_700
                    status_label.color = ft.colors.GREEN_ACCENT_700
                    # Update button to "Stop" state
                    toggle_button.icon = ft.icons.STOP_ROUNDED
                    toggle_button.tooltip = "停止"
                    toggle_button.style = ft.ButtonStyle(color=ft.colors.RED_ACCENT_700)
                    toggle_button.disabled = False # Enable stop when running
                else: # Stopped State (is_running is False or None) - Red status, Green button
                    status_icon.name = ft.icons.CIRCLE_OUTLINED # Or ft.icons.ERROR_OUTLINE if stopped due to error?
                    status_icon.color = ft.colors.RED_ACCENT_700
                    status_label.color = ft.colors.RED_ACCENT_700
                    # Update button to "Start" state
                    toggle_button.icon = ft.icons.PLAY_ARROW_ROUNDED
                    toggle_button.tooltip = "启动"
                    toggle_button.style = ft.ButtonStyle(color=ft.colors.GREEN_ACCENT_700)
                    toggle_button.disabled = False # Enable start when stopped

                # Show/hide progress indicator
                progress_indicator.visible = is_processing

                # Disable button during processing states (Starting/Stopping)
                if is_processing:
                    toggle_button.disabled = True
                # Re-enable button based on the final state (handled above for True/False)
                elif is_running is not None: # Only re-enable if state is known (True or False) and not processing
                     toggle_button.disabled = False
                # Ensure button color/icon during processing matches the transition state
                if is_processing:
                    # Consistent "Processing" state for the button
                    toggle_button.icon = ft.icons.HOURGLASS_EMPTY_ROUNDED # Use a processing icon for button too
                    toggle_button.tooltip = "处理中..."
                    toggle_button.style = ft.ButtonStyle(color=ft.colors.AMBER_700) # Amber during processing
                    toggle_button.disabled = True # Always disable during processing

                page.update()

            # Run UI updates on the Flet thread
            # Use page.run() if Flet version supports it, or keep page.run_thread()
            # Assuming page.run_thread for compatibility as used elsewhere
            page.run_thread(update_ui) # type: ignore

    def update_output_display(text: str):
        """线程安全地将文本附加到输出区域"""
        if page:  # 确保页面仍然存在
            # 使用 run_thread 执行一个简单的 lambda 来更新 UI
            current_value = output_text.value if output_text.value is not None else ""

            page.run_thread(
                lambda: setattr(output_text, "value", current_value + text + "\n")
                or page.update()  # type: ignore
            )

    # --- (Removed show_snackbar function) ---

    # --- Helper function to get value from a control ---
    def get_control_value(key: str, control_type: type = str, default: Any = None) -> Any:
        """Safely retrieves and converts the value from a GUI control."""
        control = all_config_controls.get(key) # Use the aggregated dict
        if control is None:
            logger.warning(f"Control for config key '{key}' not found in GUI. Returning default: {default}")
            return default

        value = getattr(control, "value", default)

        # Handle specific control types first
        if isinstance(control, ft.Switch):
            return bool(value) if value is not None else bool(default) # Ensure bool conversion
        if isinstance(control, ft.Dropdown):
            # Return the selected value directly, assuming it's the correct type
            # Handle case where value might be None if nothing is selected and no default
            return value if value is not None else default

        # Handle text fields (TextField) and potentially others needing string conversion/validation
        # Treat None or empty string from text field carefully
        if value is None or value == "":
            # Check specific keys that explicitly allow None
            if key in [
                "dashscope.stt.translation_target_language",
                "audio.sample_rate", # Sample rate None means auto-detect
                "llm.base_url",
                "llm.api_key", # Allow empty API key string, handle downstream
                "dashscope.api_key", # Allow empty API key string, handle downstream
            ]:
                 # If the intended type is not string, return None. If string, return empty string or None based on default.
                 return None if control_type != str else (value if value is not None else default)

            # If a default value is provided, return it for empty fields (unless None is allowed above)
            if default is not None:
                return default
            # Otherwise, return None for numeric types or empty string for strings
            elif control_type in [int, float]:
                return None # Cannot convert empty string to number
            else:
                return "" # Default empty string for text

        # Attempt type conversion for non-empty values from text-based inputs
        try:
            if control_type == int:
                return int(value)
            if control_type == float:
                # Handle potential locale issues if needed, assuming standard decimal format
                return float(value)
            if control_type == bool: # Should be handled by Switch, but as fallback
                return str(value).lower() in ['true', '1', 'yes', 'on']
            # Default to string if no other type matches (or if control_type is str)
            return str(value)
        except (ValueError, TypeError) as convert_err:
            logger.error(
                f"Invalid value '{value}' for '{key}'. Expected type {control_type}. Error: {convert_err}. Returning default value: {default}"
            )
            return default # Return default on conversion error

    # --- 配置保存/重载逻辑 ---
    async def save_config_handler(e: ft.ControlEvent):
        """保存按钮点击事件处理程序 (配置选项卡)"""
        logger.info("Save configuration button clicked.")
        # Start with a deep copy of the *current live* config data
        if not config:
            logger.error("Cannot save config, config object not available.")
            # Show error banner?
            return
        new_config_data = copy.deepcopy(config.data)

        # Define Helper function inside save_config_handler's scope or keep it in main's scope?
        try:
            # Use the get_control_value function defined in the main scope

            # Recursively update the new_config_data dictionary
            def update_nested_dict(data_dict: Dict, key: str, value: Any):
                keys = key.split('.')
                temp_dict = data_dict
                for i, k in enumerate(keys[:-1]):
                    # Ensure intermediate level is a dictionary
                    if not isinstance(temp_dict.get(k), dict):
                         temp_dict[k] = {} # Create or overwrite if not a dict
                    temp_dict = temp_dict[k]

                # Set the final value
                temp_dict[keys[-1]] = value

            # Update dictionary from controls
            # Update dictionary from controls using explicit types/defaults
            update_nested_dict(
                new_config_data,
                "dashscope.api_key",
                get_control_value("dashscope.api_key", str, ""), # API keys are strings
            )
            update_nested_dict(
                new_config_data,
                "dashscope.stt.model",
                get_control_value("dashscope.stt.model", str, "gummy-realtime-v1"), # Dropdown returns string
            )
            update_nested_dict(
                new_config_data,
                "dashscope.stt.translation_target_language",
                get_control_value("dashscope.stt.translation_target_language", str, None), # Allow None
            )
            update_nested_dict(
                new_config_data,
                "dashscope.stt.intermediate_result_behavior",
                get_control_value("dashscope.stt.intermediate_result_behavior", str, "ignore"), # Dropdown returns string
            )
            update_nested_dict(
                new_config_data,
                "audio.sample_rate",
                get_control_value("audio.sample_rate", int, None), # Allow None (auto-detect)
            )
            update_nested_dict(
                new_config_data,
                "audio.channels",
                get_control_value("audio.channels", int, 1),
            )
            update_nested_dict(
                new_config_data,
                "audio.dtype",
                get_control_value("audio.dtype", str, "int16"), # Dropdown returns string
            )
            update_nested_dict(
                new_config_data,
                "audio.debug_echo_mode",
                get_control_value("audio.debug_echo_mode", bool, False), # Switch returns bool
            )
            update_nested_dict(
                new_config_data,
                "llm.enabled",
                get_control_value("llm.enabled", bool, False), # Switch returns bool
            )
            update_nested_dict(
                new_config_data,
                "llm.api_key",
                get_control_value("llm.api_key", str, ""), # API keys are strings
            )
            update_nested_dict(
                new_config_data,
                "llm.base_url",
                get_control_value("llm.base_url", str, None), # Allow None
            )
            update_nested_dict(
                new_config_data,
                "llm.model",
                get_control_value("llm.model", str, ""),
            )
            update_nested_dict(
                new_config_data,
                "llm.system_prompt",
                get_control_value("llm.system_prompt", str, ""),
            )
            update_nested_dict(
                new_config_data,
                "llm.temperature",
                get_control_value("llm.temperature", float, 0.7),
            )
            update_nested_dict(
                new_config_data,
                "llm.max_tokens",
                get_control_value("llm.max_tokens", int, 150),
            )
            update_nested_dict(
                new_config_data,
                "outputs.vrc_osc.enabled",
                get_control_value("outputs.vrc_osc.enabled", bool, True), # Switch returns bool
            )
            update_nested_dict(
                new_config_data,
                "outputs.vrc_osc.address",
                get_control_value("outputs.vrc_osc.address", str, "127.0.0.1"),
            )
            update_nested_dict(
                new_config_data,
                "outputs.vrc_osc.port",
                get_control_value("outputs.vrc_osc.port", int, 9000),
            )
            update_nested_dict(
                new_config_data,
                "outputs.vrc_osc.message_interval",
                get_control_value("outputs.vrc_osc.message_interval", float, 1.333),
            )
            update_nested_dict(
                new_config_data,
                "outputs.console.enabled",
                get_control_value("outputs.console.enabled", bool, True), # Switch returns bool
            )
            update_nested_dict(
                new_config_data,
                "outputs.console.prefix",
                get_control_value("outputs.console.prefix", str, "[VRCMeow]"),
            )
            update_nested_dict(
                new_config_data,
                "outputs.file.enabled",
                get_control_value("outputs.file.enabled", bool, False), # Switch returns bool
            )
            update_nested_dict(
                new_config_data,
                "outputs.file.path",
                get_control_value("outputs.file.path", str, "output.txt"),
            )
            update_nested_dict(
                new_config_data,
                "outputs.file.format",
                get_control_value("outputs.file.format", str, "{timestamp} - {text}"), # Dropdown returns string
            )
            update_nested_dict(
                new_config_data,
                "logging.level",
                get_control_value("logging.level", str, "INFO"), # Dropdown returns string
            )

            # Update few-shot examples from the dynamic rows
            examples_list = []
            # Use the correct column variable reference if available
            active_few_shot_column = all_config_controls.get("llm.few_shot_examples_column", few_shot_examples_column) # Use local variable if possible
            if active_few_shot_column and isinstance(active_few_shot_column, ft.Column):
                 for row in active_few_shot_column.controls:
                    if isinstance(row, ft.Row) and len(row.controls) >= 3: # Expect 3 controls: user, assistant, remove_button
                        # Check control types more robustly before accessing value
                        user_tf = row.controls[0] if isinstance(row.controls[0], ft.TextField) else None
                        assistant_tf = row.controls[1] if isinstance(row.controls[1], ft.TextField) else None

                        if user_tf and assistant_tf:
                            user_text = user_tf.value or ""
                            assistant_text = assistant_tf.value or ""
                            # Only save if at least one field has text
                            if user_text or assistant_text:
                                examples_list.append({"user": user_text, "assistant": assistant_text})
                        else:
                            logger.warning(f"Unexpected control types in few-shot row: {row.controls}")
            else:
                 logger.warning("Few-shot examples column control not found or invalid during save.")


            logger.debug(f"Saving {len(examples_list)} few-shot examples.")
            update_nested_dict(new_config_data, "llm.few_shot_examples", examples_list)

            # Directly update the singleton's internal data BEFORE saving
            config._config_data = new_config_data
            # Recalculate derived values like logging level int after update
            log_level_str = (
                new_config_data.get("logging", {}).get("level", "INFO").upper()
            )
            log_level = getattr(logging, log_level_str, logging.INFO)
            config._config_data["logging"]["level_int"] = log_level

            # Call the save method on the config instance
            await asyncio.to_thread(config.save)  # Run synchronous save in thread

            # Show success banner
            page.banner = ft.Banner(
                bgcolor=ft.colors.GREEN_100,
                leading=ft.Icon(
                    ft.icons.CHECK_CIRCLE_OUTLINE, color=ft.colors.GREEN_800
                ),
                content=ft.Text("配置已成功保存到 config.yaml", color=ft.colors.BLACK),
                actions=[
                    ft.TextButton(
                        "关闭",
                        on_click=lambda _: gui_utils.close_banner(page), # Use imported function
                        style=ft.ButtonStyle(color=ft.colors.GREEN_900),
                    )
                ],
            )
            page.banner.open = True
            page.update()

        except Exception as ex:
            error_msg = f"保存配置时出错: {ex}"
            logger.critical(error_msg, exc_info=True)
            # Show error banner
            page.banner = ft.Banner(
                bgcolor=ft.colors.RED_100,
                leading=ft.Icon(ft.icons.ERROR_OUTLINE, color=ft.colors.RED_800),
                content=ft.Text(error_msg, color=ft.colors.BLACK),
                actions=[
                    ft.TextButton(
                        "关闭",
                        on_click=lambda _: gui_utils.close_banner(page), # Use imported function
                        style=ft.ButtonStyle(color=ft.colors.RED_900),
                    )
                ],
            )
            page.banner.open = True
            page.update()

    def reload_config_controls():
        """Updates the GUI controls with values from the reloaded config."""
        logger.info("Reloading config values into GUI controls.")
        reloaded_config_data = config.data if config else {} # Get reloaded data safely

        # Use the aggregated dictionary of controls
        for key, control in all_config_controls.items():
            # Skip special controls that don't map directly to config keys
            if key in ["llm.few_shot_examples_column", "llm.add_example_button"]:
                 continue

            # Get value from the *reloaded* data using nested access if needed
            keys = key.split('.')
            current_value = reloaded_config_data
            for k in keys:
                try:
                    # Check if current_value is a dictionary before indexing
                    if isinstance(current_value, dict):
                        current_value = current_value[k]
                    else:
                        logger.warning(f"Config path for '{key}' invalid at '{k}': parent is not a dictionary. Skipping.")
                        current_value = None
                        break
                except (KeyError, TypeError, IndexError):
                    logger.warning(f"Key '{key}' path invalid or key missing at '{k}' in reloaded config data. Skipping control update.")
                    current_value = None # Mark value as not found
                    break # Stop traversing this key

            # Assign the final retrieved value (or None if path was invalid)
            value = current_value

            try:
                if control is None: # Should not happen if loop continues, but check anyway
                     logger.debug(f"Skipping reload for key '{key}' as control is None.")
                     continue

                # --- Update control based on type ---
                if isinstance(control, ft.Switch):
                    control.value = bool(value) if value is not None else False
                elif isinstance(control, ft.Dropdown):
                    # Ensure the value exists in options before setting
                    if value is not None and hasattr(control, 'options') and isinstance(control.options, list) and any(opt.key == value for opt in control.options):
                        control.value = value
                    else:
                        # Log warning but don't change value if invalid or not found
                        if value is not None: # Log only if there was a value expected
                             logger.warning(
                                 f"Value '{value}' for dropdown '{key}' not in options or invalid. Keeping previous selection: {control.value}"
                             )
                        # Consider setting to None or a default if value is invalid? For now, keep existing.
                elif isinstance(control, ft.TextField):
                     # Check specific keys that allow None/empty representation
                    if key in [
                            "dashscope.stt.translation_target_language",
                            "audio.sample_rate",
                            "llm.base_url",
                        ] and value is None:
                         control.value = "" # Use empty string for None
                    else:
                         # Ensure value is converted to string for TextField
                         control.value = str(value) if value is not None else ""
                # Add other control types if necessary
                else:
                     logger.debug(f"Control for key '{key}' has unhandled type '{type(control)}' during reload.")

            except Exception as ex:
                logger.error(
                    f"Error reloading control for key '{key}' with value '{value}': {ex}", exc_info=True # Add exc_info
                )

        # Reload few-shot examples (ensure column control exists)
        # Use the correct column variable reference if available
        active_few_shot_column = all_config_controls.get("llm.few_shot_examples_column", few_shot_examples_column) # Use local if possible
        reloaded_config_data = config.data if config else {} # Get reloaded data again safely

        if active_few_shot_column and isinstance(active_few_shot_column, ft.Column):
            active_few_shot_column.controls.clear()  # Remove existing rows
            # Safely get examples from reloaded data
            loaded_examples = reloaded_config_data.get("llm", {}).get("few_shot_examples", [])
            if isinstance(loaded_examples, list):
                logger.info(f"Reloading {len(loaded_examples)} few-shot examples into GUI.")
                for example in loaded_examples:
                    if (
                        isinstance(example, dict)
                        and "user" in example
                        and "assistant" in example
                    ):
                        # Call the internal row creation function (defined below in main)
                        new_row = _create_example_row_internal(
                            example.get("user", ""), example.get("assistant", "")
                        )
                        active_few_shot_column.controls.append(new_row)
                    else:
                        logger.warning(
                            f"Skipping invalid few-shot example during reload: {example}"
                        )
        else:
            logger.warning(
                "'llm.few_shot_examples' in config is not a list. Cannot reload examples."
            )

        page.update()

    async def reload_config_handler(e: ft.ControlEvent):
        """Reloads configuration from file and updates the GUI."""
        logger.info("Reload configuration button clicked.")
        try:
            await asyncio.to_thread(config.reload)  # Run synchronous reload in thread
            reload_config_controls()  # Update GUI fields with new values
            # Show success banner
            page.banner = ft.Banner(
                bgcolor=ft.colors.GREEN_100,
                leading=ft.Icon(
                    ft.icons.CHECK_CIRCLE_OUTLINE, color=ft.colors.GREEN_800
                ),
                content=ft.Text("配置已从 config.yaml 重新加载", color=ft.colors.BLACK),
                actions=[
                    ft.TextButton(
                        "关闭",
                        on_click=lambda _: gui_utils.close_banner(page), # Use imported function
                        style=ft.ButtonStyle(color=ft.colors.GREEN_900),
                    )
                ],
            )
            page.banner.open = True
            page.update()
        except Exception as ex:
            error_msg = f"重新加载配置时出错: {ex}"
            logger.error(error_msg, exc_info=True)
            # Show error banner
            page.banner = ft.Banner(
                bgcolor=ft.colors.RED_100,
                leading=ft.Icon(ft.icons.ERROR_OUTLINE, color=ft.colors.RED_800),
                content=ft.Text(error_msg, color=ft.colors.BLACK),
                actions=[
                    ft.TextButton(
                        "关闭",
                        on_click=lambda _: gui_utils.close_banner(page), # Use imported function
                        style=ft.ButtonStyle(color=ft.colors.RED_900),
                    )
                ],
            )
            page.banner.open = True
            page.update()

    # --- Few-Shot Example Add/Remove Logic (defined within main) ---
    # close_banner function is now in gui_utils.py

    def _create_example_row_internal(user_text: str = "", assistant_text: str = "") -> ft.Row:
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

         # Helper function for remove button handler
         async def remove_this_row(e_remove: ft.ControlEvent):
            row_to_remove = e_remove.control.data # Get the Row associated with the button
            # Use all_config_controls to get the column reference robustly
            active_few_shot_column = all_config_controls.get("llm.few_shot_examples_column")
            if active_few_shot_column and isinstance(active_few_shot_column, ft.Column) and row_to_remove in active_few_shot_column.controls:
                active_few_shot_column.controls.remove(row_to_remove)
                logger.debug("Removed few-shot example row.")
                page.update()
            else:
                logger.warning("Attempted to remove a row not found in the column or column control not found/invalid.")

         remove_button = ft.IconButton(
            icon=ft.icons.DELETE_OUTLINE,
            tooltip="删除此示例",
            on_click=remove_this_row,
            icon_color=ft.colors.RED_ACCENT_400,
        )

         new_row = ft.Row(
            controls=[user_input, assistant_output, remove_button],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.START,
        )
         remove_button.data = new_row  # Associate the row with the button for removal
         return new_row

    async def add_example_handler(e: ft.ControlEvent):
        """Adds a new, empty example row to the column."""
        new_row = _create_example_row_internal() # Use the correct internal function
        # Use all_config_controls to get the column reference robustly
        active_few_shot_column = all_config_controls.get("llm.few_shot_examples_column")
        if active_few_shot_column and isinstance(active_few_shot_column, ft.Column):
            active_few_shot_column.controls.append(new_row)
            logger.debug("Added new few-shot example row.")
            page.update()
        else:
            logger.error("Could not add few-shot example row: Column control not found or invalid.")

    # Assign the handler to the button
    add_example_button.on_click = add_example_handler

    # --- Combined Start/Stop Logic ---
    async def _start_recording_internal():
        """Internal logic for starting the process."""
        # Status update now handles button state during processing
        update_status_display("正在启动...", is_running=None, is_processing=True)
        # page.update() # update_status_display calls page.update()

        try:
            # --- 初始化组件 ---
            logger.info("GUI 请求启动，正在初始化组件...")

            # 检查关键配置 (可以在这里再次检查或依赖 main.py 中的早期检查)
            # Use the new nested key 'dashscope.api_key'
            dashscope_api_key = config.get("dashscope.api_key")
            if not dashscope_api_key:
                error_msg = "错误：Dashscope API Key 未设置。"
                logger.error(error_msg)
                update_status_display(error_msg, is_running=False, is_processing=False) # Show error state, this will reset button via callback
                # No need to manually reset button here, callback handles it
                # toggle_button.icon = ft.icons.PLAY_ARROW_ROUNDED
                # toggle_button.tooltip = "启动"
                # toggle_button.style = ft.ButtonStyle(color=ft.colors.GREEN_ACCENT_700)
                # toggle_button.disabled = False
                # page.update()
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
                await app_state.vrc_client.start()  # VRCClient 现在需要显式启动
                logger.info("VRCClient 已初始化并启动。")
            else:
                logger.info("VRC OSC 输出已禁用，跳过 VRCClient 初始化。")
                app_state.vrc_client = None

            # 2. LLMClient (如果启用)
            llm_enabled = config.get("llm.enabled", False)
            if llm_enabled:
                app_state.llm_client = LLMClient()
                if not app_state.llm_client.enabled:
                    logger.warning(
                        "LLMClient 初始化失败或 API Key 缺失，LLM 处理将被禁用。"
                    )
                    app_state.llm_client = None
                else:
                    logger.info("LLMClient 已初始化。")
            else:
                app_state.llm_client = None

            # 3. OutputDispatcher (传递 VRC 客户端和 GUI 输出回调)
            app_state.output_dispatcher = OutputDispatcher(
                vrc_client_instance=app_state.vrc_client,
                gui_output_callback=update_output_display,  # 传递新的回调
            )
            logger.info("OutputDispatcher 已初始化。")

            # 4. AudioManager (传递 LLM 客户端、调度器和状态回调)
            app_state.audio_manager = AudioManager(
                llm_client=app_state.llm_client,
                output_dispatcher=app_state.output_dispatcher,
                status_callback=update_status_display,  # 传递状态更新回调
            )
            logger.info("AudioManager 已初始化。")

            # --- 启动 AudioManager ---
            # AudioManager.start() 会启动后台线程
            app_state.audio_manager.start()
            # The threads inside AudioManager will update the status via callback.
            app_state.is_running = True
            logger.info("AudioManager start requested. Threads are running.")
            # Initial status update comes from AudioManager threads now.
            # The status callback will update the button to the correct 'Running' state (stop icon, red color).
            # No explicit button update needed here after successful start request.
            # toggle_button.icon = ft.icons.STOP_ROUNDED
            # toggle_button.tooltip = "停止"
            # toggle_button.style = ft.ButtonStyle(color=ft.colors.RED_ACCENT_700)
            # toggle_button.disabled = False
            # page.update()

        except Exception as ex:
            error_msg = f"启动过程中出错: {ex}"
            logger.critical(error_msg, exc_info=True)
            # Update status display to show error, which will reset the button state via callback
            update_status_display(f"启动错误: {ex}", is_running=False, is_processing=False)
            # Attempt cleanup even on startup error
            await _stop_recording_internal(is_error=True) # Call internal stop logic for cleanup

    async def _stop_recording_internal(is_error: bool = False):
        """Internal logic for stopping the process."""
        # Only proceed if actually running or if called due to an error needing cleanup
        if not app_state.is_running and not is_error:
             # If already stopped and not an error, ensure button/status is in 'Stopped' state (Red/Green)
             update_status_display("已停止", is_running=False, is_processing=False) # This updates the button/status
             return

        # Status update now handles button state during processing
        update_status_display("正在停止...", is_running=True, is_processing=True)
        # page.update() # update_status_display calls page.update()

        logger.info("GUI Requesting Stop...")
        # tasks_to_await = [] # Not used

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

        app_state.is_running = False # Mark as not running logically first
        logger.info("All components requested to stop.")
        # Final status update ("Stopped" or "Stopped (with issues)") comes from AudioManager callback
        # update_status_display("已停止", is_running=False, is_processing=False) # Removed, handled by callback
        # Button state update also handled by AudioManager callback upon final stop status
        # start_button.disabled = False # Handled by status callback
        # stop_button.disabled = True # Handled by status callback
        # page.update() # Update handled by callback

    async def toggle_recording(e: ft.ControlEvent):
        """Handles clicks on the combined Start/Stop button."""
        if app_state.is_running:
            await _stop_recording_internal()
        else:
            await _start_recording_internal()

    # --- 绑定事件 ---
    toggle_button.on_click = toggle_recording # Assign the new handler

    # --- 页面关闭处理 ---
    async def on_window_event(e: ft.ControlEvent):
        if e.data == "close":
            logger.info("检测到窗口关闭事件。")
            # Ensure processes are stopped before closing
            if app_state.is_running:
                logger.info("Window closing, stopping background processes...")
                await _stop_recording_internal() # Call internal stop logic
            # Now destroy the window
            page.window_destroy()


    # --- Bind event handlers ---
    toggle_button.on_click = toggle_recording      # Dashboard button
    save_config_button.on_click = save_config_handler    # Config button
    reload_config_button.on_click = reload_config_handler # Config button
    add_example_button.on_click = add_example_handler    # Config button (already assigned above, but good to be explicit)
    page.on_window_event = on_window_event               # Page event


    # --- Layout using Tabs ---
    # Import layout functions from the new modules
    # Remove direct element imports from gui_dashboard
    from gui_dashboard import create_dashboard_tab_content
    from gui_config import create_config_tab_content

    # Create tab content by calling functions from imported modules
    dashboard_tab_layout = create_dashboard_tab_content(
        # Pass the elements created locally in main
        status_row_control=status_row,             # Pass the local status_row
        toggle_button_control=toggle_button,       # Pass the local toggle_button
        progress_indicator_control=progress_indicator, # Pass the local progress_indicator
        output_text_control=output_text            # Pass the local output_text
    )

    config_tab_layout = create_config_tab_content(
        save_button=save_config_button,
        reload_button=reload_config_button,
        all_controls=all_config_controls # Pass the aggregated dictionary
    )

    page.add(
        ft.Tabs(
            [
                ft.Tab(
                    text="仪表盘",
                    icon=ft.icons.DASHBOARD,
                    content=dashboard_tab_layout, # Use correct variable
                ),
                ft.Tab(text="配置", icon=ft.icons.SETTINGS, content=config_tab_layout), # Use correct variable
            ],
            expand=True,  # Make tabs fill the page width
        )
    )

    # Initial population of few-shot examples on first load
    logger.debug("Initial population of few-shot examples UI.")
    # Use safe access to initial config data
    initial_config_data = config.data if config else {}
    initial_examples = initial_config_data.get('llm', {}).get('few_shot_examples', [])
    # Use the correct column variable reference
    active_few_shot_column = all_config_controls.get("llm.few_shot_examples_column", few_shot_examples_column)
    if isinstance(initial_examples, list) and active_few_shot_column and isinstance(active_few_shot_column, ft.Column):
        for example in initial_examples:
            if isinstance(example, dict) and 'user' in example and 'assistant' in example:
                # Use the internal function defined within main()
                initial_row = _create_example_row_internal( # Ensure this call uses the correct function
                    example.get('user', ''), example.get('assistant', '')
                )
                active_few_shot_column.controls.append(initial_row)
            else:
                logger.warning(f"Skipping invalid few-shot example during initial load: {example}")
    else:
        logger.warning("'llm.few_shot_examples' in initial config is not a list.")


    # Initial page update
    page.update()

# 注意：此文件不再包含 if __name__ == "__main__": ft.app(...)
# 这将移至 main.py
