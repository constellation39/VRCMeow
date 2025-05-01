import logging
import queue
from typing import Dict, Optional

import flet as ft

# Use a thread-safe queue to pass log records from logging threads to the Flet UI thread
log_queue = queue.Queue(maxsize=1000)  # Limit queue size


class FletLogHandler(logging.Handler):
    """A custom logging handler that puts log records into a thread-safe queue."""

    def __init__(self, level=logging.NOTSET):
        super().__init__(level=level)
        self.formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s",
            datefmt="%H:%M:%S",  # Use a shorter timestamp for GUI logs
        )

    def emit(self, record: logging.LogRecord):
        """Format the record and put it into the queue."""
        log_entry = self.format(record)
        try:
            # Use non-blocking put to avoid blocking the logging thread if the queue is full
            log_queue.put_nowait(log_entry)
        except queue.Full:
            # Handle queue full scenario if necessary (e.g., drop the log)
            # print("Log queue is full, dropping log message.", file=sys.stderr)
            pass


def create_log_elements() -> Dict[str, ft.Control]:
    """Creates the UI elements for the Log tab."""
    elements = {}
    elements["log_output"] = ft.ListView(
        expand=True,
        spacing=2,
        auto_scroll=True,  # Keep scrolled to the bottom
        # divider_thickness=1, # Optional divider
    )
    elements["clear_log_button"] = ft.IconButton(
        icon=ft.icons.DELETE_SWEEP_OUTLINED,
        tooltip="清除日志",
        on_click=None,  # Handler assigned in gui.py
    )
    elements["log_level_dropdown"] = ft.Dropdown(
        label="日志级别",
        hint_text="选择要显示的最低日志级别",
        options=[
            ft.dropdown.Option("DEBUG"),
            ft.dropdown.Option("INFO"),
            ft.dropdown.Option("WARNING"),
            ft.dropdown.Option("ERROR"),
            ft.dropdown.Option("CRITICAL"),
        ],
        value="INFO",  # Default level
        width=150,
        dense=True,
        on_change=None,  # Handler assigned in gui.py
    )
    return elements


def create_log_tab_content(elements: Dict[str, ft.Control]) -> ft.Column:
    """Creates the layout Column for the Log tab."""
    log_output = elements.get("log_output")
    clear_button = elements.get("clear_log_button")
    level_dropdown = elements.get("log_level_dropdown")

    if not log_output or not clear_button or not level_dropdown:
        return ft.Column(
            [ft.Text("Error creating log tab layout.", color=ft.colors.RED)]
        )

    return ft.Column(
        [
            ft.Divider(height=5, color=ft.colors.TRANSPARENT), # Add spacer at the top
            ft.Row(
                [level_dropdown, clear_button],
                alignment=ft.MainAxisAlignment.END,
            ),
            ft.Divider(height=1, thickness=1),
            ft.Container(  # Container to hold the ListView and allow expansion
                content=log_output,
                expand=True,
                border=ft.border.all(1, ft.colors.with_opacity(0.5, ft.colors.OUTLINE)),
                border_radius=ft.border_radius.all(5),
                padding=5,
            ),
        ],
        expand=True,
        spacing=5,
    )


# --- Log Update Logic ---

# Store the full log history and the filtered list separately
_full_log_history = []
_filtered_log_controls = []
_current_log_level = logging.INFO  # Default level


def set_log_level_filter(level_name: str):
    """Sets the minimum level for logs displayed in the GUI."""
    global _current_log_level
    level = logging.getLevelName(level_name.upper())
    if isinstance(level, int):
        _current_log_level = level
        # Trigger a refresh of the displayed logs
        _refilter_log_display()
    else:
        print(f"Warning: Invalid log level selected: {level_name}")


def _refilter_log_display():
    """Re-populates the filtered log controls based on the current level."""
    global _filtered_log_controls
    _filtered_log_controls = []
    for level, text_control in _full_log_history:
        if level >= _current_log_level:
            _filtered_log_controls.append(text_control)


def _get_log_color(level_name: str) -> Optional[str]:
    """Returns a color based on the log level."""
    level_name = level_name.upper()
    if level_name == "CRITICAL":
        return ft.colors.RED_ACCENT_700
    elif level_name == "ERROR":
        return ft.colors.RED_400
    elif level_name == "WARNING":
        return ft.colors.AMBER_700
    elif level_name == "DEBUG":
        return ft.colors.with_opacity(0.7, ft.colors.ON_SURFACE)  # Dim debug messages
    # INFO and others default to None (default text color)
    return None


def update_log_display(
    page: ft.Page,
    log_output_listview: ft.ListView,
    max_log_entries: int = 500,  # Limit displayed entries
):
    """
    Checks the log queue and updates the Flet ListView.
    This function should be called periodically from the Flet UI thread.
    """
    if not page or not log_output_listview:
        return

    global _full_log_history, _filtered_log_controls

    try:
        # Process all available logs in the queue
        logs_added = False
        while not log_queue.empty():
            try:
                log_entry = log_queue.get_nowait()
                logs_added = True

                # Parse level name for coloring and filtering (simple parsing)
                level_name = "INFO"  # Default
                parts = log_entry.split(" - ", 3)
                if len(parts) > 1:
                    level_name = parts[1]  # e.g., "INFO", "WARNING"

                log_level_int = logging.getLevelName(level_name.upper())
                if not isinstance(log_level_int, int):
                    log_level_int = logging.INFO  # Fallback

                # Create the Text control for the log entry
                log_text_control = ft.Text(
                    log_entry,
                    size=11,
                    selectable=True,
                    color=_get_log_color(level_name),
                    # Optional: Add width/overflow for long lines
                    # width=page.width - 40 if page.width else 800, # Adjust width based on page
                    # overflow=ft.TextOverflow.ELLIPSIS,
                )

                # Add to full history
                _full_log_history.append((log_level_int, log_text_control))

                # Add to filtered list if it meets the current level
                if log_level_int >= _current_log_level:
                    _filtered_log_controls.append(log_text_control)

            except queue.Empty:
                break  # Should not happen with the loop condition, but good practice
            except Exception as e:
                print(f"Error processing log entry: {e}")  # Log error processing error

        # Limit history size (both full and filtered)
        if len(_full_log_history) > max_log_entries:
            num_to_remove = len(_full_log_history) - max_log_entries
            _full_log_history = _full_log_history[num_to_remove:]
            # Refilter after trimming full history
            _refilter_log_display()
            logs_added = True  # Force UI update if logs were trimmed

        # Update the ListView only if logs were added or trimmed/refiltered
        if logs_added:
            log_output_listview.controls = _filtered_log_controls
            # Check page validity before updating
            if page and page.controls is not None:
                log_output_listview.update()  # Update the ListView itself
                # page.update(log_output_listview) # Alternative: update via page

    except Exception as e:
        # Catch errors during the update process itself
        print(f"Error updating log display: {e}")


def clear_log_display(page: ft.Page, log_output_listview: ft.ListView):
    """Clears the log display in the GUI."""
    global _full_log_history, _filtered_log_controls
    _full_log_history = []
    _filtered_log_controls = []
    log_output_listview.controls = []
    if page and page.controls is not None:
        log_output_listview.update()
