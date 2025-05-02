import logging
from typing import Dict, Optional

import flet as ft

logger = logging.getLogger(__name__)


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
    # REMOVED: elements["output_text"] = ft.TextField(...)
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
    elements["info_mic_label"] = ft.Text(
        "麦克风: -", **default_info_text_style, selectable=True
    )
    elements["info_stt_label"] = ft.Text(
        "STT: -", **default_info_text_style, selectable=True
    )
    elements["info_llm_label"] = ft.Text(
        "LLM: -", **default_info_text_style, selectable=True
    )
    elements["info_vrc_label"] = ft.Text(
        "VRC OSC: -", **default_info_text_style, selectable=True
    )
    elements["info_file_label"] = ft.Text( # Fix indentation and add missing assignment
        "文件输出: -", **default_info_text_style, selectable=True
    )
    elements["info_preset_label"] = ft.Text( # Add preset label element
        "LLM 预设: -", **default_info_text_style, selectable=True
    )

    # --- Add element for audio level visualization ---
    elements["audio_level_bar"] = ft.ProgressBar(
        width=150,  # Adjust width as needed
        height=8,  # Adjust height for thickness
        value=0.0,  # Start empty
        bar_height=8,
        color=ft.colors.with_opacity(0.7, ft.colors.BLUE_ACCENT),  # Color for the bar
        bgcolor=ft.colors.with_opacity(0.2, ft.colors.OUTLINE),  # Background color
        border_radius=ft.border_radius.all(4),
        tooltip="当前麦克风音量",
        # visible=False # Initially hidden, shown when running? Or always visible? Let's keep visible.
    )
    elements["info_config_path_label"] = ft.Text(
        "配置文件: Loading...",
        **default_info_text_style,
        selectable=True,
        color=ft.colors.SECONDARY,
    )  # Add config path text

    return elements


def create_dashboard_tab_content(elements: Dict[str, ft.Control]) -> ft.Column:
    """Creates the layout Column for the Dashboard tab using pre-created elements."""
    # Validate required elements exist (removed output_text)
    required_keys = ["status_row", "toggle_button", "progress_indicator"]
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
            ft.Row(  # Add ft.Row() here
                [elements["toggle_button"], elements["progress_indicator"]],
                alignment=ft.MainAxisAlignment.CENTER,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=15,
            ),
            ft.Divider(height=10, thickness=1),  # Separator
            # --- Configuration Info Section ---
            ft.Column(
                [
                    # Combine Mic info and volume bar in one row
                    ft.Row(
                        [
                            ft.Icon(
                                name=ft.icons.MIC_NONE_OUTLINED, size=16, opacity=0.7
                            ),
                            elements["info_mic_label"],
                            ft.Container(width=10),  # Spacer
                            elements["audio_level_bar"],  # Add volume bar here
                        ],
                        spacing=8,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    # Other info rows remain the same
                    _create_info_row(
                        ft.icons.RECORD_VOICE_OVER_OUTLINED, elements["info_stt_label"]
                    ),
                    _create_info_row(
                        ft.icons.TEXT_SNIPPET_OUTLINED, elements["info_llm_label"]
                    ),
                    _create_info_row(  # Add preset info row
                        ft.icons.EDIT_NOTE_OUTLINED, elements["info_preset_label"]
                    ),
                    _create_info_row(
                        ft.icons.SEND_AND_ARCHIVE_OUTLINED, elements["info_vrc_label"]
                    ),
                    _create_info_row(
                        ft.icons.SAVE_ALT_OUTLINED, elements["info_file_label"]
                    ),
                    _create_info_row(
                        ft.icons.FOLDER_OPEN_OUTLINED,
                        elements["info_config_path_label"],
                    ),  # Add config file path row
                ],
                spacing=5,  # Adjust spacing if needed after adding the bar
                alignment=ft.MainAxisAlignment.START,
                horizontal_alignment=ft.CrossAxisAlignment.START,  # Align info text left
            ),
            ft.Divider(height=10, thickness=1),  # Separator
            # REMOVED: elements["output_text"],
        ],
        # expand=True, # Remove expand from column
        alignment=ft.MainAxisAlignment.START,  # Keep column alignment
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=10,
        # scroll=ft.ScrollMode.ADAPTIVE, # Remove scroll from column, TextField handles its own
    )


# --- Dashboard Callback Functions ---

# Function to update the static info display
# Forward reference Config for type hinting
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import Config  # Import Config class for type hinting


def update_dashboard_info_display(
    page: ft.Page,
    elements: Dict[str, ft.Control],
    config_instance: "Config",  # Pass the config instance
):
    """线程安全地更新仪表盘上的静态配置信息显示"""
    if not page:
        logger.warning(
            "update_dashboard_info_display called without a valid page object."
        )
        return
    if not elements:
        logger.warning("update_dashboard_info_display called without elements.")
        return
    if not config_instance:
        logger.warning("update_dashboard_info_display called without config_instance.")
        return

    # Get the latest config data from the instance
    try:
        config_data = config_instance.data
        if not config_data:
            logger.warning("Config instance provided but its data is empty.")
            return  # Or handle appropriately
    except Exception as e:
        logger.error(f"Error accessing config_instance.data: {e}", exc_info=True)
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
    llm_preset = llm_conf.get("active_preset_name", "Default") # Get active preset name
    llm_info = f"LLM: {'启用' if llm_enabled else '禁用'}"
    if llm_enabled:
        llm_info += f" ({llm_model})"
    preset_info = f"LLM 预设: {llm_preset}" # Create preset info string

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
    preset_label = elements.get("info_preset_label") # Get preset label control
    vrc_label = elements.get("info_vrc_label")
    file_label = elements.get("info_file_label")
    config_path_label = elements.get("info_config_path_label") # Get config path label

    def update_info_ui():
        if mic_label and isinstance(mic_label, ft.Text):
            mic_label.value = mic_info
        if stt_label and isinstance(stt_label, ft.Text):
            stt_label.value = stt_info
        if llm_label and isinstance(llm_label, ft.Text):
            llm_label.value = llm_info
        if preset_label and isinstance(preset_label, ft.Text): # Update preset label
            preset_label.value = preset_info
        if vrc_label and isinstance(vrc_label, ft.Text):
            vrc_label.value = vrc_info
        if file_label and isinstance(file_label, ft.Text):
            file_label.value = file_info
        # Update config file path display
        if config_path_label and isinstance(config_path_label, ft.Text):
            loaded_path = getattr(
                config_instance, "loaded_config_path", "Unknown"
            )  # Use getattr for safety with FallbackConfig
            config_path_label.value = (
                f"配置文件: {loaded_path}"  # Display the path/status
            )

        try:
            if page and page.controls:
                # Update only the specific controls that changed
                controls_to_update = [
                    ctrl
                    for ctrl in [
                        mic_label,
                        stt_label,
                        llm_label,
                        preset_label, # Add preset label to update list
                        vrc_label,
                        file_label,
                        config_path_label,
                    ]
                    if ctrl
                ]
                if controls_to_update:
                    page.update(*controls_to_update)
            elif page:
                logger.warning(
                    "Page has no controls, skipping update in update_dashboard_info_display."
                )
        except Exception as e:
            logger.error(
                f"Error during page.update in update_dashboard_info_display: {e}",
                exc_info=True,
            )

    try:
        if page and page.controls is not None:
            page.run_thread(update_info_ui)  # type: ignore
        elif page:
            logger.warning(
                "Page object seems invalid, skipping run_thread in update_dashboard_info_display."
            )
    except Exception as e:
        logger.error(
            f"Error calling page.run_thread in update_dashboard_info_display: {e}",
            exc_info=True,
        )


def update_audio_level_display(
    page: ft.Page,
    audio_level_bar: ft.ProgressBar,
    level: float,  # Expect normalized level 0.0 to 1.0
):
    """线程安全地更新音频电平指示器 (需要传入 UI 元素)"""
    if not page:
        # logger.warning("update_audio_level_display called without a valid page object.") # Can be noisy
        return
    if not audio_level_bar:
        logger.warning("update_audio_level_display called without audio_level_bar.")
        return

    # Clamp level just in case
    level = max(0.0, min(1.0, level))

    def update_ui():
        if audio_level_bar:
            audio_level_bar.value = level
            try:
                # Check if page and controls are still valid before updating
                if page and page.controls:
                    # Update only the specific control
                    page.update(audio_level_bar)
                # No need for else, page check already handled
            except Exception:
                # Catch errors during update (e.g., page closed unexpectedly)
                # logger.error(f"Error during page.update in update_audio_level_display: {e}", exc_info=True) # Can be noisy
                pass  # Ignore update errors for level bar

    # Run UI updates on the Flet thread
    try:
        # Check if page is still valid before running thread
        if page and page.controls is not None:
            page.run_thread(update_ui)  # type: ignore
        # No need for else, page check already handled
    except Exception:
        # Catch potential errors if page becomes invalid between check and run_thread
        # logger.error(f"Error calling page.run_thread in update_audio_level_display: {e}", exc_info=True) # Can be noisy
        pass  # Ignore run_thread errors for level bar


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

        # --- Button State Logic ---
        # Default: Stopped state
        toggle_button.icon = ft.icons.PLAY_ARROW_ROUNDED
        toggle_button.tooltip = "启动"
        toggle_button.style = ft.ButtonStyle(color=ft.colors.GREEN_ACCENT_700)
        toggle_button.disabled = False

        # Running state
        if is_running is True:
            toggle_button.icon = ft.icons.STOP_ROUNDED
            toggle_button.tooltip = "停止"
            toggle_button.style = ft.ButtonStyle(color=ft.colors.RED_ACCENT_700)
            toggle_button.disabled = False

        # Processing state (overrides running/stopped for button appearance and disabled state)
        if is_processing:
            toggle_button.icon = ft.icons.HOURGLASS_EMPTY_ROUNDED
            toggle_button.tooltip = "处理中..."
            toggle_button.style = ft.ButtonStyle(color=ft.colors.AMBER_700)
            toggle_button.disabled = True

        # --- Status Icon/Label Logic ---
        if is_processing:
            status_icon.name = ft.icons.HOURGLASS_EMPTY_ROUNDED
            status_icon.color = ft.colors.AMBER_700
            status_label.color = ft.colors.AMBER_700
        elif is_running is True:
            status_icon.name = ft.icons.CHECK_CIRCLE_ROUNDED
            status_icon.color = ft.colors.GREEN_ACCENT_700
            status_label.color = ft.colors.GREEN_ACCENT_700
        else: # Stopped
            status_icon.name = ft.icons.CIRCLE_OUTLINED
            status_icon.color = ft.colors.RED_ACCENT_700
            status_label.color = ft.colors.RED_ACCENT_700

        # Show/hide progress indicator
        progress_indicator.visible = is_processing

        try:
            # Check if page and controls are still valid before updating
            if page and page.controls:
                # Update only the specific controls that changed
                controls_to_update = [
                    ctrl for ctrl in [
                        status_icon,
                        status_label,
                        toggle_button,
                        progress_indicator,
                    ] if ctrl
                ]
                if controls_to_update:
                    page.update(*controls_to_update)
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

# REMOVED: update_output_display function
