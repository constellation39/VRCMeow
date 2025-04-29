import flet as ft
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

# --- Dashboard UI Element Creation and Layout ---

def create_dashboard_elements() -> Dict[str, ft.Control]:
    """Creates the core UI elements for the dashboard tab."""
    elements = {}
    # Default state: Red status, Green button
    elements["status_icon"] = ft.Icon(name=ft.icons.CIRCLE_OUTLINED, color=ft.colors.RED_ACCENT_700)
    elements["status_label"] = ft.Text("未启动", selectable=True, color=ft.colors.RED_ACCENT_700)
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
        expand=True,
        min_lines=5,
        border_radius=ft.border_radius.all(8),
        border_color=ft.colors.with_opacity(0.5, ft.colors.OUTLINE),
        filled=True,
        bgcolor=ft.colors.with_opacity(0.02, ft.colors.ON_SURFACE),
        content_padding=15,
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
    return elements


def create_dashboard_tab_content(
    elements: Dict[str, ft.Control]
) -> ft.Column:
    """Creates the layout Column for the Dashboard tab using pre-created elements."""
    # Validate required elements exist
    required_keys = ["status_row", "toggle_button", "progress_indicator", "output_text"]
    if not all(key in elements for key in required_keys):
        logger.error("Missing required elements for dashboard layout.")
        return ft.Column([ft.Text("Error: Dashboard layout failed.", color=ft.colors.RED)])

    return ft.Column(
        [
            # Status display at the top
            ft.Container(elements["status_row"], padding=ft.padding.only(top=15, bottom=5)),

            # Combined Start/Stop button below status, centered
            ft.Row(
                [elements["toggle_button"], elements["progress_indicator"]],
                alignment=ft.MainAxisAlignment.CENTER,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=15,
            ),

            # Output text area taking remaining space
            ft.Container(
                elements["output_text"],
                expand=True,
                padding=ft.padding.only(top=15, bottom=10, left=10, right=10)
            ),
        ],
        expand=True,
        alignment=ft.MainAxisAlignment.START,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=10,
    )


# --- Dashboard Callback Functions ---

def update_status_display(
    page: ft.Page,
    status_icon: ft.Icon,
    status_label: ft.Text,
    toggle_button: ft.IconButton,
    progress_indicator: ft.ProgressRing,
    message: str,
    is_running: Optional[bool] = None,
    is_processing: bool = False
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
                 logger.warning("Page has no controls, skipping update in update_status_display.")
            # No need for else, page check already handled
        except Exception as e:
            # Catch errors during update (e.g., page closed unexpectedly)
            logger.error(f"Error during page.update in update_status_display: {e}", exc_info=True)


    # Run UI updates on the Flet thread
    try:
        # Check if page is still valid before running thread
        if page and page.controls is not None:
             page.run_thread(update_ui) # type: ignore
        elif page:
             logger.warning("Page object seems invalid (no controls), skipping run_thread in update_status_display.")
        # No need for else, page check already handled
    except Exception as e:
        # Catch potential errors if page becomes invalid between check and run_thread
        logger.error(f"Error calling page.run_thread in update_status_display: {e}", exc_info=True)


def update_output_display(page: ft.Page, output_text_control: ft.TextField, text: str):
    """线程安全地将文本附加到输出区域 (需要传入 UI 元素)"""
    if not page:
        logger.warning("update_output_display called without a valid page object.")
        return
    if not output_text_control:
        logger.warning("update_output_display called without a valid output_text_control.")
        return

    def append_text():
        current_value = output_text_control.value if output_text_control.value is not None else ""
        # Limit the amount of text to prevent performance issues
        max_len = 10000 # Example limit, adjust as needed
        new_value = current_value + text + "\n"
        if len(new_value) > max_len:
             # Keep the last max_len characters
             new_value = new_value[-max_len:]
             # Optional: Add a marker indicating truncation
             # new_value = "[...]\n" + new_value

        output_text_control.value = new_value
        try:
            # Check if page and controls are still valid before updating
            if page and page.controls:
                 page.update(output_text_control) # Update only the specific control
            elif page:
                 logger.warning("Page has no controls, skipping update in update_output_display.")
            # No need for else, page check already handled
        except Exception as e:
            # Catch errors during update
            logger.error(f"Error during page.update in update_output_display: {e}", exc_info=True)

    try:
        # Check if page is still valid before running thread
        if page and page.controls is not None:
             page.run_thread(append_text) # type: ignore
        elif page:
             logger.warning("Page object seems invalid (no controls), skipping run_thread in update_output_display.")
        # No need for else, page check already handled
    except Exception as e:
        # Catch potential errors if page becomes invalid
        logger.error(f"Error calling page.run_thread in update_output_display: {e}", exc_info=True)
