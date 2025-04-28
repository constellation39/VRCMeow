import flet as ft

# --- Dashboard Layout Function ---
# UI elements (status_icon, status_label, etc.) are now created in gui.py
# and passed to this function.

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
