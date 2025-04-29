import flet as ft
from typing import Optional, Dict
import logging
from datetime import datetime # Import datetime

logger = logging.getLogger(__name__)

from config import Config # Import Config for type hinting

# --- Dashboard UI Element Creation and Layout ---

def _create_info_row(icon: str, text_control: ft.Control) -> ft.Row:
    """Helper to create a consistent row for info display."""
    return ft.Row(
        [
            ft.Icon(name=icon, size=16, opacity=0.7),
            text_control,
        ],
        spacing=8,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )

def create_dashboard_elements() -> Dict[str, ft.Control]:
    """Creates the core UI elements for the dashboard tab, including info display."""
    elements = {}
    # Default state: Red status, Green button
    elements["status_icon"] = ft.Icon(
        name=ft.icons.CIRCLE_OUTLINED, color=ft.colors.RED_ACCENT_700
    )
    elements["status_label"] = ft.Text(
        "未启动", selectable=True, color=ft.colors.RED_ACCENT_700
    )
    elements["status_row"] = ft.Row(
        [elements["status_icon"], elements["status_label"]],
        alignment=ft.MainAxisAlignment.CENTER,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=5,
    )
    elements["output_text"] = ft.TextField(
        hint_text="最终输出将显示在这里...",
        multiline=True,
        read_only=True,
        # expand=True, # Remove expand
        height=200, # Set a fixed height
        # min_lines=5, # Remove min_lines
        border_radius=ft.border_radius.all(8),
        border_color=ft.colors.with_opacity(0.5, ft.colors.OUTLINE),
        filled=True,
        bgcolor=ft.colors.with_opacity(0.02, ft.colors.ON_SURFACE),
        content_padding=ft.padding.only(top=15, bottom=10, left=10, right=10), # Apply padding directly here
    )
    elements["toggle_button"] = ft.IconButton(
        icon=ft.icons.PLAY_ARROW_ROUNDED,
        tooltip="启动",
        on_click=None,  # Handler assigned in gui.py
        disabled=False,
        icon_size=30,
        style=ft.ButtonStyle(color=ft.colors.GREEN_ACCENT_700),
    )
    elements["progress_indicator"] = ft.ProgressRing(
        width=20, height=20, stroke_width=2, visible=False
    )

    # --- Add elements for displaying configuration info ---
    default_info_text_style = {"size": 12, "opacity": 0.9}
    elements["info_mic_label"] = ft.Text("麦克风: -", **default_info_text_style, selectable=True)
    elements["info_stt_label"] = ft.Text("STT: -", **default_info_text_style, selectable=True)
    elements["info_llm_label"] = ft.Text("LLM: -", **default_info_text_style, selectable=True)
    elements["info_vrc_label"] = ft.Text("VRC OSC: -", **default_info_text_style, selectable=True)
    elements["info_file_label"] = ft.Text("文件输出: -", **default_info_text_style, selectable=True)

    return elements


def create_dashboard_tab_content(elements: Dict[str, ft.Control]) -> ft.Column:
    """Creates the layout Column for the Dashboard tab using pre-created elements."""
    # Validate required elements exist
    required_keys = ["status_row", "toggle_button", "progress_indicator", "output_text"]
    if not all(key in elements for key in required_keys):
        logger.error("Missing required elements for dashboard layout.")
        return ft.Column(
            [ft.Text("Error: Dashboard layout failed.", color=ft.colors.RED)]
        )

    return ft.Column(
        [
            # Status display at the top
            ft.Container(
                elements["status_row"], padding=ft.padding.only(top=15, bottom=5)
            ),
            # Combined Start/Stop button below status, centered
            ft.Row( # Add ft.Row() here
                [elements["toggle_button"], elements["progress_indicator"]],
                alignment=ft.MainAxisAlignment.CENTER,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=15,
            ),
            ft.Divider(height=10, thickness=1), # Separator

            # --- Configuration Info Section ---
            ft.Column(
                [
                    _create_info_row(ft.icons.MIC_NONE_OUTLINED, elements["info_mic_label"]),
                    _create_info_row(ft.icons.RECORD_VOICE_OVER_OUTLINED, elements["info_stt_label"]),
                    _create_info_row(ft.icons.TEXT_SNIPPET_OUTLINED, elements["info_llm_label"]),
                    _create_info_row(ft.icons.SEND_AND_ARCHIVE_OUTLINED, elements["info_vrc_label"]),
                    _create_info_row(ft.icons.SAVE_ALT_OUTLINED, elements["info_file_label"]),
                ],
                spacing=5,
                alignment=ft.MainAxisAlignment.START,
                horizontal_alignment=ft.CrossAxisAlignment.START, # Align info text left
            ),
            ft.Divider(height=10, thickness=1), # Separator

            # Output text area with fixed height
            elements["output_text"], # Place TextField directly in the Column
        ],
       # expand=True, # Remove expand from column
       alignment=ft.MainAxisAlignment.START,
       horizontal_alignment=ft.CrossAxisAlignment.CENTER,
       spacing=10,
       # scroll=ft.ScrollMode.ADAPTIVE, # Remove scroll from column, TextField handles its own
   )


# --- Dashboard Callback Functions ---

# Function to update the static info display
def update_dashboard_info_display(
    page: ft.Page,
    elements: Dict[str, ft.Control],
    config_data: Dict, # Pass the config data dictionary
):
    """线程安全地更新仪表盘上的静态配置信息显示"""
    if not page:
        logger.warning("update_dashboard_info_display called without a valid page object.")
        return
    if not elements:
        logger.warning("update_dashboard_info_display called without elements.")
        return
    if not config_data:
        logger.warning("update_dashboard_info_display called without config_data.")
        return

    # Extract info from config_data (use .get() for safety)
    audio_conf = config_data.get("audio", {})
    mic_device = audio_conf.get("device", "Default")
    mic_info = f"麦克风: {mic_device}"

    dash_conf = config_data.get("dashscope", {}).get("stt", {})
    stt_model = dash_conf.get("model", "未知")
    stt_translate = dash_conf.get("translation_target_language")
    stt_info = f"STT: {stt_model}"
    if stt_translate:
        stt_info += f" (翻译: {stt_translate})"

    llm_conf = config_data.get("llm", {})
    llm_enabled = llm_conf.get("enabled", False)
    llm_model = llm_conf.get("model", "未知")
    llm_info = f"LLM: {'启用' if llm_enabled else '禁用'}"
    if llm_enabled:
        llm_info += f" ({llm_model})"

    vrc_conf = config_data.get("outputs", {}).get("vrc_osc", {})
    vrc_enabled = vrc_conf.get("enabled", False)
    vrc_addr = vrc_conf.get("address", "未知")
    vrc_port = vrc_conf.get("port", "未知")
    vrc_info = f"VRC OSC: {'启用' if vrc_enabled else '禁用'}"
    if vrc_enabled:
        vrc_info += f" ({vrc_addr}:{vrc_port})"

    file_conf = config_data.get("outputs", {}).get("file", {})
    file_enabled = file_conf.get("enabled", False)
    file_path = file_conf.get("path", "未知")
    file_info = f"文件输出: {'启用' if file_enabled else '禁用'}"
    if file_enabled:
        file_info += f" ({file_path})"

    # Get control references
    mic_label = elements.get("info_mic_label")
    stt_label = elements.get("info_stt_label")
    llm_label = elements.get("info_llm_label")
    vrc_label = elements.get("info_vrc_label")
    file_label = elements.get("info_file_label")

    def update_info_ui():
        if mic_label and isinstance(mic_label, ft.Text): mic_label.value = mic_info
        if stt_label and isinstance(stt_label, ft.Text): stt_label.value = stt_info
        if llm_label and isinstance(llm_label, ft.Text): llm_label.value = llm_info
        if vrc_label and isinstance(vrc_label, ft.Text): vrc_label.value = vrc_info
        if file_label and isinstance(file_label, ft.Text): file_label.value = file_info

        try:
            if page and page.controls:
                # Update only the specific controls that changed
                controls_to_update = [ctrl for ctrl in [mic_label, stt_label, llm_label, vrc_label, file_label] if ctrl]
                if controls_to_update:
                    page.update(*controls_to_update)
            elif page:
                logger.warning("Page has no controls, skipping update in update_dashboard_info_display.")
        except Exception as e:
            logger.error(f"Error during page.update in update_dashboard_info_display: {e}", exc_info=True)

    try:
        if page and page.controls is not None:
            page.run_thread(update_info_ui) # type: ignore
        elif page:
            logger.warning("Page object seems invalid, skipping run_thread in update_dashboard_info_display.")
    except Exception as e:
        logger.error(f"Error calling page.run_thread in update_dashboard_info_display: {e}", exc_info=True)


def update_status_display(
    page: ft.Page,
    status_icon: ft.Icon,
    status_label: ft.Text,
    toggle_button: ft.IconButton,
    progress_indicator: ft.ProgressRing,
    message: str,
    is_running: Optional[bool] = None,
    is_processing: bool = False,
):
    """线程安全地更新状态文本和图标 (需要传入 UI 元素)"""
    if not page:
        logger.warning("update_status_display called without a valid page object.")
        return

    def update_ui():
        status_label.value = message  # Update text
        # Update icon and text color based on state
        if is_processing:
            # Transition State (Starting/Stopping) - Amber status, button handled below
            status_icon.name = ft.icons.HOURGLASS_EMPTY_ROUNDED
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
            toggle_button.disabled = False  # Enable stop when running
        else:  # Stopped State (is_running is False or None) - Red status, Green button
            status_icon.name = ft.icons.CIRCLE_OUTLINED
            status_icon.color = ft.colors.RED_ACCENT_700
            status_label.color = ft.colors.RED_ACCENT_700
            # Update button to "Start" state
            toggle_button.icon = ft.icons.PLAY_ARROW_ROUNDED
            toggle_button.tooltip = "启动"
            toggle_button.style = ft.ButtonStyle(color=ft.colors.GREEN_ACCENT_700)
            toggle_button.disabled = False  # Enable start when stopped

        # Show/hide progress indicator
        progress_indicator.visible = is_processing

        # Disable button during processing states (Starting/Stopping)
        if is_processing:
            toggle_button.disabled = True
        # Re-enable button based on the final state (handled above for True/False)
        elif is_running is not None:
            toggle_button.disabled = False
        # Ensure button color/icon during processing matches the transition state
        if is_processing:
            # Consistent "Processing" state for the button
            toggle_button.icon = ft.icons.HOURGLASS_EMPTY_ROUNDED
            toggle_button.tooltip = "处理中..."
            toggle_button.style = ft.ButtonStyle(color=ft.colors.AMBER_700)
            toggle_button.disabled = True  # Always disable during processing

        try:
            # Check if page and controls are still valid before updating
            if page and page.controls:
                page.update()
            elif page:
                logger.warning(
                    "Page has no controls, skipping update in update_status_display."
                )
            # No need for else, page check already handled
        except Exception as e:
            # Catch errors during update (e.g., page closed unexpectedly)
            logger.error(
                f"Error during page.update in update_status_display: {e}", exc_info=True
            )

    # Run UI updates on the Flet thread
    try:
        # Check if page is still valid before running thread
        if page and page.controls is not None:
            page.run_thread(update_ui)  # type: ignore
        elif page:
            logger.warning(
                "Page object seems invalid (no controls), skipping run_thread in update_status_display."
            )
        # No need for else, page check already handled
    except Exception as e:
        # Catch potential errors if page becomes invalid between check and run_thread
        logger.error(
            f"Error calling page.run_thread in update_status_display: {e}",
            exc_info=True,
        )


def update_output_display(page: ft.Page, output_text_control: ft.TextField, text: str):
    """线程安全地将文本附加到输出区域 (需要传入 UI 元素)"""
    if not page:
        logger.warning("update_output_display called without a valid page object.")
        return
    if not output_text_control:
        logger.warning(
            "update_output_display called without a valid output_text_control."
        )
        return

    def append_text():
        current_value = (
            output_text_control.value if output_text_control.value is not None else ""
        )
        # Get current time and format it
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_text = f"[{timestamp}] {text}"

        # Limit the amount of text to prevent performance issues
        max_len = 10000  # Example limit, adjust as needed
        new_value = current_value + formatted_text + "\n" # Add formatted text
        if len(new_value) > max_len:
            # Keep the last max_len characters
            # Find the first newline after the part to be truncated to avoid partial lines
            cutoff_point = new_value.find('\n', len(new_value) - max_len)
            if cutoff_point != -1:
                new_value = "[... log truncated ...]\n" + new_value[cutoff_point + 1:]
            else: # If no newline found (very long single line), just truncate
                new_value = "[... log truncated ...]\n" + new_value[-max_len:]
            # Optional: Add a marker indicating truncation
            # new_value = "[...]\n" + new_value

        output_text_control.value = new_value
        try:
            # Check if page and controls are still valid before updating
            if page and page.controls:
                page.update(output_text_control)  # Update only the specific control
            elif page:
                logger.warning(
                    "Page has no controls, skipping update in update_output_display."
                )
            # No need for else, page check already handled
        except Exception as e:
            # Catch errors during update
            logger.error(
                f"Error during page.update in update_output_display: {e}", exc_info=True
            )

    try:
        # Check if page is still valid before running thread
        if page and page.controls is not None:
            page.run_thread(append_text)  # type: ignore
        elif page:
            logger.warning(
                "Page object seems invalid (no controls), skipping run_thread in update_output_display."
            )
        # No need for else, page check already handled
    except Exception as e:
        # Catch potential errors if page becomes invalid
        logger.error(f"Error calling page.run_thread in update_output_display: {e}", exc_info=True)
