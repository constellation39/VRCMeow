import functools
import logging
from typing import TYPE_CHECKING, Callable, Dict, Optional

import flet as ft

from prompt_presets import delete_preset, get_preset, load_presets, save_preset

if TYPE_CHECKING:
    from config import Config  # For type hinting

logger = logging.getLogger(__name__)


# --- Few-Shot Example Row Creation/Deletion Logic (Adapted for Preset Tab) ---


# This function now needs page and the column reference within the preset tab.
# It defines the remove handler internally, capturing the necessary scope.
def create_preset_example_row(
    page: ft.Page,  # Need page for update
    few_shot_column: ft.Column,  # Need column ref within preset tab
    user_text: str = "",
    assistant_text: str = "",
) -> ft.Row:
    """Creates a Flet Row for a single few-shot example within the preset tab."""
    # Create controls for the row
    user_input = ft.TextField(
        label="用户输入 (User)",
        value=user_text,
        multiline=True,
        max_lines=3,
        expand=True,
        dense=True,  # Make denser for preset tab
    )
    assistant_output = ft.TextField(
        label="助手响应 (Assistant)",
        value=assistant_text,
        multiline=True,
        max_lines=3,
        expand=True,
        dense=True,  # Make denser
    )

    # Define remove handler within this scope to capture page and column
    async def remove_this_row(e_remove: ft.ControlEvent):
        row_to_remove = e_remove.control.data  # Get the Row associated with the button
        if few_shot_column and row_to_remove in few_shot_column.controls:
            few_shot_column.controls.remove(row_to_remove)
            logger.debug("Removed few-shot example row from preset tab.")
            try:
                # Update the column directly, page update might not be needed if contained
                few_shot_column.update()
                # if page and page.controls: page.update() # Avoid full page update if possible
            except Exception as e:
                logger.error(
                    f"Error updating preset few-shot column after removing row: {e}",
                    exc_info=True,
                )
        else:
            logger.warning(
                "Attempted to remove a row not found in the preset column or column is invalid."
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


async def add_example_handler_preset(
    page: ft.Page,  # Need page for row creation
    few_shot_column: ft.Column,  # Need column ref within preset tab
    e: Optional[ft.ControlEvent] = None,  # Add optional event argument
) -> None:  # Explicitly type return as None
    """Adds a new, empty example row to the preset tab's column."""
    if few_shot_column and isinstance(few_shot_column, ft.Column):
        # Pass page and column ref to the internal row creation function
        try:
            new_row = create_preset_example_row(
                page, few_shot_column
            )  # Create row with handler
            few_shot_column.controls.append(new_row)
            logger.debug("Added new few-shot example row to preset tab.")
            few_shot_column.update()  # Update the column directly
            # if page and page.controls: page.update() # Avoid full page update
        except Exception as e:
            logger.error(
                f"Error adding or updating preset few-shot column: {e}", exc_info=True
            )
            # Show banner? Maybe just log for preset tab internal error.
            # gui_utils.show_error_banner(page, f"添加示例时出错: {e}")
    else:
        logger.error(
            "Could not add few-shot example row to preset tab: Column control not found or invalid."
        )
        # gui_utils.show_error_banner(page, "无法添加示例：UI 元素丢失。")


# Define the type for the callback function (partial) to update the main config UI's preset label
# It receives: active_preset_name_value
UpdateConfigUICallback = Callable[[str], None]


def create_preset_tab_content(
    page: ft.Page,
    config_instance: "Config",  # Add config instance parameter
    update_config_ui_callback: UpdateConfigUICallback,  # Callback to update main Config Tab UI's label
) -> Dict[str, ft.Control]:
    """
    Creates the content Column for the Preset Management Tab, initializes it
    with the currently active preset from config, and returns key controls.
    """

    presets_data = load_presets()
    preset_names = sorted(list(presets_data.keys()))

    # --- Get the initially active preset name from config ---
    initial_active_preset_name = config_instance.get(
        "llm.active_preset_name", "Default"
    )
    logger.info(
        f"Preset Tab: Initial active preset name from config: '{initial_active_preset_name}'"
    )

    # --- Load the initial preset's data ---
    initial_preset_data = get_preset(initial_active_preset_name)
    if not initial_preset_data:
        logger.warning(
            f"Preset Tab: Initial active preset '{initial_active_preset_name}' not found. Falling back to 'Default'."
        )
        initial_active_preset_name = "Default"  # Fallback to Default name
        initial_preset_data = get_preset(initial_active_preset_name)
        if not initial_preset_data:
            logger.error(
                "Preset Tab: CRITICAL - Failed to load even the 'Default' preset data. UI will be empty."
            )
            # Set empty defaults if even 'Default' fails
            initial_preset_data = {"system_prompt": "", "few_shot_examples": []}

    initial_system_prompt = initial_preset_data.get("system_prompt", "")
    initial_few_shot_examples = initial_preset_data.get("few_shot_examples", [])

    # --- UI Controls (Initialized with loaded preset data) ---
    preset_select_dd = ft.Dropdown(
        label="选择预设",
        options=[ft.dropdown.Option(name) for name in preset_names],
        # Set initial value if it exists in the options
        value=initial_active_preset_name
        if initial_active_preset_name in preset_names
        else None,
        tooltip="选择一个预设以加载其设置到下方的编辑区域",  # Updated tooltip
        expand=True,
    )

    # Label to display the currently active preset name
    active_preset_name_label = ft.Text(
        f"当前活动预设: {initial_active_preset_name}",  # Set initial text based on loaded preset
        italic=True,
        color=ft.colors.SECONDARY,
        size=12,
    )

    # --- Preset Content Editing Controls ---
    system_prompt_tf = ft.TextField(
        label="系统提示 (System Prompt)",
        value=initial_system_prompt,  # Set initial value
        multiline=True,
        min_lines=4,
        max_lines=8,
        tooltip="指导 LLM 如何回应 (编辑此处以修改当前选定预设)",
        expand=True,
    )

    # Populate initial few-shot examples
    initial_few_shot_rows = []
    if isinstance(initial_few_shot_examples, list):
        for example in initial_few_shot_examples:
            if (
                isinstance(example, dict)
                and "user" in example
                and "assistant" in example
            ):
                # Need a temporary column reference here for row creation,
                # will be replaced by the actual column below.
                # This is slightly awkward, maybe refactor create_preset_example_row later.
                temp_col_ref = ft.Column()  # Dummy column for now
                row = create_preset_example_row(
                    page,
                    temp_col_ref,
                    example.get("user", ""),
                    example.get("assistant", ""),
                )
                initial_few_shot_rows.append(row)

    few_shot_column = ft.Column(
        controls=initial_few_shot_rows,  # Set initial controls
        spacing=5,
        scroll=ft.ScrollMode.ADAPTIVE,
    )
    # Now, fix the column reference in the already created rows
    for row in few_shot_column.controls:
        if isinstance(row, ft.Row) and len(row.controls) > 0:
            remove_button = row.controls[-1]  # Assume remove button is last
            if isinstance(remove_button, ft.IconButton):
                # Re-create the remove handler with the correct column reference
                # This requires modifying create_preset_example_row or handling it here.
                # Let's handle it here for now by re-assigning on_click.
                async def remove_this_row_wrapper(
                    e_remove: ft.ControlEvent, row_to_remove=row
                ):
                    if row_to_remove in few_shot_column.controls:
                        few_shot_column.controls.remove(row_to_remove)
                        logger.debug("Removed few-shot example row from preset tab.")
                        try:
                            few_shot_column.update()
                        except Exception as e:
                            logger.error(
                                f"Error updating preset few-shot column after removing row: {e}",
                                exc_info=True,
                            )
                    else:
                        logger.warning(
                            "Attempted to remove a row not found in the preset column."
                        )

                remove_button.on_click = remove_this_row_wrapper
                remove_button.data = row  # Ensure data is still set

    add_example_button = ft.TextButton(
        "添加 Few-Shot 示例",
        icon=ft.icons.ADD,
        on_click=None,  # Assigned later after column is defined
        tooltip="向当前预设添加用户/助手交互示例",
    )
    # --- End Preset Content Editing Controls ---

    new_preset_name_tf = ft.TextField(
        label="新预设名称",
        value=initial_active_preset_name,  # Set initial value to loaded preset name
        tooltip="输入要保存的新预设的名称 (将使用下方编辑区域中的当前值)",
        expand=True,
    )

    status_text = ft.Text(
        "", size=11, color=ft.colors.SECONDARY, height=30
    )  # For feedback, reserve height

    # --- Handlers ---
    # Assign add example handler now that the column exists
    add_example_partial = functools.partial(
        add_example_handler_preset, page, few_shot_column
    )
    add_example_button.on_click = add_example_partial

    def update_dropdown():
        """Reloads presets and updates the dropdown options."""
        nonlocal presets_data, preset_names  # Allow modification
        presets_data = load_presets()
        preset_names = sorted(list(presets_data.keys()))
        preset_select_dd.options = [ft.dropdown.Option(name) for name in preset_names]
        # Try to keep selection if possible, otherwise reset
        current_value = preset_select_dd.value
        if current_value not in preset_names:
            preset_select_dd.value = None
        # Update the controls within this tab
        preset_select_dd.update()
        # active_preset_name_label.update() # No need to update label here

    async def load_selected_preset(e: ft.ControlEvent):
        """Loads the selected preset's content into this tab's editing controls."""
        selected_name = preset_select_dd.value
        if not selected_name:
            status_text.value = "请先选择一个预设。"
            status_text.color = ft.colors.ERROR
            status_text.update()  # Update status text within the tab
            return

        logger.info(f"Loading preset '{selected_name}' content into Preset Tab UI...")
        preset_data = get_preset(selected_name)
        if preset_data:
            try:
                # 1. Update the editing controls in this tab
                system_prompt_tf.value = preset_data.get("system_prompt", "")
                few_shot_column.controls.clear()  # Clear existing examples
                loaded_examples = preset_data.get("few_shot_examples", [])
                if isinstance(loaded_examples, list):
                    for example in loaded_examples:
                        if (
                            isinstance(example, dict)
                            and "user" in example
                            and "assistant" in example
                        ):
                            new_row = create_preset_example_row(
                                page,
                                few_shot_column,
                                example.get("user", ""),
                                example.get("assistant", ""),
                            )
                            few_shot_column.controls.append(new_row)
                        else:
                            logger.warning(
                                f"Skipping invalid few-shot example structure in preset '{selected_name}'."
                            )
                else:
                    logger.warning(
                        f"Few-shot examples in preset '{selected_name}' is not a list."
                    )

                # Update the controls visually
                system_prompt_tf.update()
                few_shot_column.update()

                # 2. Update the "New Preset Name" field with the loaded preset's name
                new_preset_name_tf.value = selected_name
                new_preset_name_tf.update()

                # 3. Update the active preset label in this tab
                active_preset_name_label.value = f"当前活动预设: {selected_name}"
                active_preset_name_label.update()

                # 4. Call the callback to update the label in the *other* tabs (via gui.py -> gui_config.py)
                # This indicates which preset *will be saved* if the user hits save in the config tab.
                update_config_ui_callback(selected_name)

                status_text.value = f"预设 '{selected_name}' 的内容已加载到编辑区域。"
                status_text.color = ft.colors.GREEN_700
                logger.info(
                    f"Preset '{selected_name}' content loaded into Preset Tab UI."
                )
                status_text.update()
                return  # Exit after successful load

            except Exception as load_err:
                error_msg = (
                    f"加载预设 '{selected_name}' 内容到预设编辑区域时出错: {load_err}"
                )
                logger.error(error_msg, exc_info=True)
                status_text.value = error_msg
                status_text.color = ft.colors.ERROR
        else:
            status_text.value = f"错误：无法找到预设 '{selected_name}' 的数据。"
            status_text.color = ft.colors.ERROR
            logger.error(f"Preset data not found for '{selected_name}' during load.")

        status_text.update()  # Update status text within the tab

    async def save_current_as_preset(e: ft.ControlEvent):
        """Saves the current values from this tab's editing controls as a preset."""
        preset_name_to_save = new_preset_name_tf.value
        if not preset_name_to_save or not preset_name_to_save.strip():
            status_text.value = "请输入要保存的预设名称。"
            status_text.color = ft.colors.ERROR
            status_text.update()
            return

        preset_name_to_save = preset_name_to_save.strip()

        # --- Get current values from the controls within THIS tab ---
        current_system_prompt = system_prompt_tf.value or ""

        current_few_shot_examples = []
        try:
            for row_control in few_shot_column.controls:
                if isinstance(row_control, ft.Row) and len(row_control.controls) >= 2:
                    user_tf = row_control.controls[0]
                    assistant_tf = row_control.controls[1]
                    if isinstance(user_tf, ft.TextField) and isinstance(
                        assistant_tf, ft.TextField
                    ):
                        user_text = user_tf.value or ""
                        assistant_text = assistant_tf.value or ""
                        if (
                            user_text.strip() or assistant_text.strip()
                        ):  # Only save non-empty examples
                            current_few_shot_examples.append(
                                {
                                    "user": user_text,
                                    "assistant": assistant_text,
                                }
                            )
        except Exception as ex:
            logger.error(
                f"Error reading few-shot examples from Preset Tab UI: {ex}",
                exc_info=True,
            )
            status_text.value = "错误：从预设编辑区域读取 Few-Shot 示例时出错。"
            status_text.color = ft.colors.ERROR
            status_text.update()
            return

        logger.info(
            f"Attempting to save current Preset Tab UI values as preset '{preset_name_to_save}'..."
        )
        if save_preset(
            preset_name_to_save, current_system_prompt, current_few_shot_examples
        ):
            status_text.value = f"预设 '{preset_name_to_save}' 已保存。"
            status_text.color = ft.colors.GREEN_700
            new_preset_name_tf.value = ""  # Clear name field
            update_dropdown()  # Refresh dropdown list in this tab
            preset_select_dd.value = (
                preset_name_to_save  # Select the newly saved preset
            )
            logger.info(f"Preset '{preset_name_to_save}' saved successfully.")
            # Also update the active preset label in this tab immediately
            active_preset_name_label.value = f"当前活动预设: {preset_name_to_save}"
            # Update controls within this tab
            status_text.update()
            new_preset_name_tf.update()
            preset_select_dd.update()
            active_preset_name_label.update()
            # The active preset name *in the config file* will be updated on next save/reload
            # based on the active_preset_name_label.

        else:
            status_text.value = f"错误：保存预设 '{preset_name_to_save}' 失败。"
            status_text.color = ft.colors.ERROR
            logger.error(f"Failed to save preset '{preset_name_to_save}'.")

        status_text.update()  # Update status text within the tab

    async def delete_selected_preset(e: ft.ControlEvent):
        """Deletes the preset selected in the dropdown."""
        selected_name = preset_select_dd.value
        if not selected_name:
            status_text.value = "请先选择要删除的预设。"
            status_text.color = ft.colors.ERROR
            status_text.update()
            return

        if selected_name == "Default":
            status_text.value = "无法删除 'Default' 预设。"
            status_text.color = ft.colors.ERROR
            status_text.update()
            return

        logger.warning(f"Attempting to delete preset '{selected_name}'...")
        # Optional: Add confirmation dialog here? For now, direct delete.
        if delete_preset(selected_name):
            status_text.value = f"预设 '{selected_name}' 已删除。"
            status_text.color = ft.colors.GREEN_700
            update_dropdown()  # Refresh dropdown list in this tab
            page.update()  # Explicitly update the page to ensure dropdown refresh
            logger.info(f"Preset '{selected_name}' deleted successfully.")
            # Check if the deleted preset was the one currently active (shown in this tab's label)
            if active_preset_name_label.value == f"当前活动预设: {selected_name}":
                logger.info(
                    f"Deleted preset '{selected_name}' was active. Loading 'Default' preset content into Preset Tab UI."
                )
                # Load the 'Default' preset content into this tab's UI
                default_preset_data = get_preset("Default")
                if default_preset_data:
                    # Update this tab's editing controls
                    system_prompt_tf.value = default_preset_data.get(
                        "system_prompt", ""
                    )
                    few_shot_column.controls.clear()
                    default_examples = default_preset_data.get("few_shot_examples", [])
                    if isinstance(default_examples, list):
                        for example in default_examples:
                            if (
                                isinstance(example, dict)
                                and "user" in example
                                and "assistant" in example
                            ):
                                new_row = create_preset_example_row(
                                    page,
                                    few_shot_column,
                                    example.get("user", ""),
                                    example.get("assistant", ""),
                                )
                                few_shot_column.controls.append(new_row)
                    system_prompt_tf.update()
                    few_shot_column.update()

                    # Update the active label in this tab
                    active_preset_name_label.value = "当前活动预设: Default"
                    active_preset_name_label.update()

                    # Call the callback to update the label elsewhere
                    update_config_ui_callback("Default")
                else:
                    logger.error(
                        "Could not load 'Default' preset data after deleting active one. Clearing preset UI."
                    )
                    # Clear editing controls if Default fails to load
                    system_prompt_tf.value = ""
                    few_shot_column.controls.clear()
                    system_prompt_tf.update()
                    few_shot_column.update()
                    # Update label to indicate no active preset loaded
                    active_preset_name_label.value = (
                        "当前活动预设: None (Default missing!)"
                    )
                    active_preset_name_label.update()
                    # Update label elsewhere
                    update_config_ui_callback("None")

        else:
            # Error message handled within delete_preset logging
            status_text.value = f"错误：删除预设 '{selected_name}' 失败。"
            status_text.color = ft.colors.ERROR
            logger.error(f"Failed to delete preset '{selected_name}'.")

        status_text.update()  # Update status text within the tab

    # --- Tab Content Layout ---
    content_column = ft.Column(
        [
            # --- Preset Selection/Loading ---
            ft.Text("选择和加载预设", weight=ft.FontWeight.BOLD),
            ft.Row(
                [
                    preset_select_dd,
                    ft.ElevatedButton(
                        "加载选中",
                        icon=ft.icons.INPUT,
                        on_click=load_selected_preset,
                        tooltip="将选定预设的内容加载到下方的编辑区域",
                    ),
                ],
                alignment=ft.MainAxisAlignment.START,
            ),
            active_preset_name_label,  # Display active preset name here
            ft.Divider(),
            # --- Preset Editing Area ---
            ft.Text("编辑预设内容", weight=ft.FontWeight.BOLD),
            system_prompt_tf,
            ft.Text("Few-Shot 示例:", style=ft.TextThemeStyle.BODY_MEDIUM),
            few_shot_column,  # Column for examples
            add_example_button,  # Button to add examples
            ft.Divider(),
            # --- Saving New/Existing Preset ---
            ft.Text("保存预设", weight=ft.FontWeight.BOLD),
            ft.Row(
                [
                    new_preset_name_tf,
                    ft.ElevatedButton(
                        "保存",
                        icon=ft.icons.SAVE,
                        on_click=save_current_as_preset,
                        tooltip="使用上方输入的名称保存当前编辑区域的内容 (覆盖同名预设)",
                    ),
                ],
                alignment=ft.MainAxisAlignment.START,
            ),
            ft.Divider(),
            # --- Deleting Preset ---
            ft.Text("删除预设", weight=ft.FontWeight.BOLD),
            ft.ElevatedButton(
                "删除选中预设",
                icon=ft.icons.DELETE_FOREVER,
                on_click=delete_selected_preset,
                color=ft.colors.RED,
                tooltip="删除上方下拉框中选定的预设 ('Default' 除外)",
            ),
            ft.Divider(),
            status_text,  # Feedback area
        ],
        spacing=15,  # Increased spacing
        # Remove fixed width/height, let tab handle sizing
        # width=500,
        # height=350,
        scroll=ft.ScrollMode.ADAPTIVE,  # Allow scrolling if content overflows
        expand=True,  # Allow column to expand within the tab
    )

    # Return a dictionary containing the main content and key controls
    return {
        "content": content_column,
        "active_preset_name_label": active_preset_name_label,
        "preset_select_dd": preset_select_dd,
        # Add other controls if they need to be accessed from gui.py
    }
