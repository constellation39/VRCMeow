import flet as ft

# --- Dashboard UI Element Definitions ---
# Define elements as module-level variables for easy import and access

# Default state: Red status, Green button
status_icon = ft.Icon(name=ft.icons.CIRCLE_OUTLINED, color=ft.colors.RED_ACCENT_700) # Default RED
status_label = ft.Text("未启动", selectable=True, color=ft.colors.RED_ACCENT_700) # Default RED
status_row = ft.Row(
    [status_icon, status_label],
    alignment=ft.MainAxisAlignment.CENTER,
    vertical_alignment=ft.CrossAxisAlignment.CENTER,
    spacing=5, # Add spacing between icon and text
)

output_text = ft.TextField(
    # Removed label for minimalism
    hint_text="最终输出将显示在这里...", # Add hint text
    multiline=True,
    read_only=True,
    expand=True,
    min_lines=5,
    border_radius=ft.border_radius.all(8), # Add slight rounding
    border_color=ft.colors.with_opacity(0.5, ft.colors.OUTLINE), # Subtle border
    filled=True, # Use a filled background
    bgcolor=ft.colors.with_opacity(0.02, ft.colors.ON_SURFACE), # Very subtle background
    content_padding=15, # Add padding inside the text field
)

# Combined Start/Stop button (initialized for Start state)
toggle_button = ft.IconButton(
    icon=ft.icons.PLAY_ARROW_ROUNDED,
    tooltip="启动",
    on_click=None, # Handler assigned in main gui.py
    disabled=False,
    icon_size=30, # Make icon slightly larger
    style=ft.ButtonStyle(color=ft.colors.GREEN_ACCENT_700), # Default GREEN button
)

# Progress indicator
progress_indicator = ft.ProgressRing(width=20, height=20, stroke_width=2, visible=False)


# --- Dashboard Layout Function ---

def create_dashboard_tab_content(
    status_row_control: ft.Row,
    toggle_button_control: ft.IconButton,
    progress_indicator_control: ft.ProgressRing,
    output_text_control: ft.TextField
) -> ft.Column:
    """Creates the layout Column for the Dashboard tab."""
    return ft.Column(
        [
            # Status display at the top
            ft.Container(status_row_control, padding=ft.padding.only(top=15, bottom=5)),

            # Combined Start/Stop button below status, centered
            ft.Row(
                [toggle_button_control, progress_indicator_control],
                alignment=ft.MainAxisAlignment.CENTER,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=15,
            ),

            # Output text area taking remaining space
            ft.Container(
                output_text_control,
                expand=True,
                padding=ft.padding.only(top=15, bottom=10, left=10, right=10)
            ),
        ],
        expand=True,
        alignment=ft.MainAxisAlignment.START,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=10,
    )
