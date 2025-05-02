import flet as ft
import logging
from typing import Dict, Any, Callable, List, Optional, TYPE_CHECKING

# Import preset logic functions
from prompt_presets import load_presets, get_preset, save_preset, delete_preset

# Import GUI utils for banners
import gui_utils

if TYPE_CHECKING:
    from config import Config # For type hinting if needed

logger = logging.getLogger(__name__)

# Define the type for the callback function to update the main config UI
# It receives: system_prompt, few_shot_examples, active_preset_name
UpdateConfigUICallback = Callable[[str, List[Dict[str, str]], str], None]


def create_preset_management_dialog_content(
    page: ft.Page,
    all_config_controls: Dict[str, ft.Control], # Need controls to read current values
    update_config_ui_callback: UpdateConfigUICallback, # Callback to update main UI
    dialog_ref: ft.AlertDialog, # Need reference to close/update the dialog
) -> ft.Column:
    """Creates the content Column for the Preset Management AlertDialog."""

    presets_data = load_presets()
    preset_names = sorted(list(presets_data.keys()))

    # --- UI Controls ---
    preset_select_dd = ft.Dropdown(
        label="选择预设",
        options=[ft.dropdown.Option(name) for name in preset_names],
        value=None, # Start with no selection
        tooltip="选择一个预设以加载其设置到主配置表单",
        expand=True,
    )

    new_preset_name_tf = ft.TextField(
        label="新预设名称",
        tooltip="输入要保存的新预设的名称 (将使用主配置表单中的当前值)",
        expand=True,
    )

    status_text = ft.Text("", size=11, color=ft.colors.SECONDARY, height=30) # For feedback, reserve height

    # --- Handlers ---
    def update_dropdown():
        """Reloads presets and updates the dropdown options."""
        nonlocal presets_data, preset_names # Allow modification
        presets_data = load_presets()
        preset_names = sorted(list(presets_data.keys()))
        preset_select_dd.options = [ft.dropdown.Option(name) for name in preset_names]
        # Try to keep selection if possible, otherwise reset
        current_value = preset_select_dd.value
        if current_value not in preset_names:
            preset_select_dd.value = None
        # Update the dialog content to show the refreshed dropdown
        # Removed window_exists check
        dialog_ref.update()

    async def load_selected_preset(e: ft.ControlEvent):
        """Loads the selected preset into the main config UI."""
        selected_name = preset_select_dd.value
        if not selected_name:
            status_text.value = "请先选择一个预设。"
            status_text.color = ft.colors.ERROR
            # Removed window_exists check
            dialog_ref.update() # Update dialog content
            return

        logger.info(f"Loading preset '{selected_name}' into main config UI...")
        preset_data = get_preset(selected_name)
        if preset_data:
            try:
                # Call the callback provided by gui.py to update the main UI controls
                update_config_ui_callback(
                    preset_data.get("system_prompt", ""),
                    preset_data.get("few_shot_examples", []),
                    selected_name, # Pass the name of the loaded preset
                )
                status_text.value = f"预设 '{selected_name}' 已加载到配置表单。"
                status_text.color = ft.colors.GREEN_700
                logger.info(f"Preset '{selected_name}' loaded successfully into UI.")
                # Close the dialog after loading
                dialog_ref.open = False
                # Removed window_exists check
                page.update() # Update page to close dialog
                return # Exit after closing dialog
            except Exception as load_err:
                error_msg = f"加载预设 '{selected_name}' 到 UI 时出错: {load_err}"
                logger.error(error_msg, exc_info=True)
                status_text.value = error_msg
                status_text.color = ft.colors.ERROR
        else:
            status_text.value = f"错误：无法找到预设 '{selected_name}' 的数据。"
            status_text.color = ft.colors.ERROR
            logger.error(f"Preset data not found for '{selected_name}' during load.")

        # Removed window_exists check
        dialog_ref.update() # Update dialog content if not closed

    async def save_current_as_preset(e: ft.ControlEvent):
        """Saves the current values from the main config UI as a preset."""
        preset_name_to_save = new_preset_name_tf.value
        if not preset_name_to_save or not preset_name_to_save.strip():
            status_text.value = "请输入要保存的预设名称。"
            status_text.color = ft.colors.ERROR
            # Removed window_exists check
            dialog_ref.update()
            return

        preset_name_to_save = preset_name_to_save.strip()

        # --- Get current values from the main config UI controls ---
        system_prompt_control = all_config_controls.get("llm.system_prompt")
        few_shot_column_control = all_config_controls.get("llm.few_shot_examples_column")

        current_system_prompt = ""
        if isinstance(system_prompt_control, ft.TextField):
            current_system_prompt = system_prompt_control.value or ""
        else:
            logger.error("Could not find system prompt control to save preset.")
            status_text.value = "错误：找不到系统提示控件。"
            status_text.color = ft.colors.ERROR
            # Removed window_exists check
            dialog_ref.update()
            return

        current_few_shot_examples = []
        if isinstance(few_shot_column_control, ft.Column):
            try:
                for row_control in few_shot_column_control.controls:
                    if isinstance(row_control, ft.Row) and len(row_control.controls) >= 2:
                        user_tf = row_control.controls[0]
                        assistant_tf = row_control.controls[1]
                        if isinstance(user_tf, ft.TextField) and isinstance(assistant_tf, ft.TextField):
                            user_text = user_tf.value or ""
                            assistant_text = assistant_tf.value or ""
                            if user_text.strip() or assistant_text.strip(): # Only save non-empty examples
                                current_few_shot_examples.append({
                                    "user": user_text,
                                    "assistant": assistant_text,
                                })
            except Exception as ex:
                logger.error(f"Error reading few-shot examples from UI: {ex}", exc_info=True)
                status_text.value = "错误：读取 Few-Shot 示例时出错。"
                status_text.color = ft.colors.ERROR
                # Removed window_exists check
                dialog_ref.update()
                return
        else:
            logger.error("Could not find few-shot examples column control to save preset.")
            status_text.value = "错误：找不到 Few-Shot 示例控件。"
            status_text.color = ft.colors.ERROR
            # Removed window_exists check
            dialog_ref.update()
            return

        logger.info(f"Attempting to save current UI values as preset '{preset_name_to_save}'...")
        if save_preset(preset_name_to_save, current_system_prompt, current_few_shot_examples):
            status_text.value = f"预设 '{preset_name_to_save}' 已保存。"
            status_text.color = ft.colors.GREEN_700
            new_preset_name_tf.value = "" # Clear name field
            update_dropdown() # Refresh dropdown list (updates dialog)
            preset_select_dd.value = preset_name_to_save # Select the newly saved preset
            logger.info(f"Preset '{preset_name_to_save}' saved successfully.")
            # Also update the main UI's active preset label immediately
            active_preset_label = all_config_controls.get("llm.active_preset_name_label")
            if isinstance(active_preset_label, ft.Text):
                 active_preset_label.value = f"当前预设: {preset_name_to_save}"
                 # Removed window_exists check
                 page.update() # Update main page UI
            else:
                 logger.warning("Could not update active preset label on main page after save.")

        else:
            status_text.value = f"错误：保存预设 '{preset_name_to_save}' 失败。"
            status_text.color = ft.colors.ERROR
            logger.error(f"Failed to save preset '{preset_name_to_save}'.")

        # Removed window_exists check
        dialog_ref.update() # Update dialog content

    async def delete_selected_preset(e: ft.ControlEvent):
        """Deletes the preset selected in the dropdown."""
        selected_name = preset_select_dd.value
        if not selected_name:
            status_text.value = "请先选择要删除的预设。"
            status_text.color = ft.colors.ERROR
            # Removed window_exists check
            dialog_ref.update()
            return

        if selected_name == "Default":
            status_text.value = "无法删除 'Default' 预设。"
            status_text.color = ft.colors.ERROR
            # Removed window_exists check
            dialog_ref.update()
            return

        logger.warning(f"Attempting to delete preset '{selected_name}'...")
        # Optional: Add confirmation dialog here? For now, direct delete.
        if delete_preset(selected_name):
            status_text.value = f"预设 '{selected_name}' 已删除。"
            status_text.color = ft.colors.GREEN_700
            update_dropdown() # Refresh dropdown list (updates dialog)
            logger.info(f"Preset '{selected_name}' deleted successfully.")
            # Check if the deleted preset was the one currently loaded in the main UI
            active_preset_label = all_config_controls.get("llm.active_preset_name_label")
            if isinstance(active_preset_label, ft.Text) and active_preset_label.value == f"当前预设: {selected_name}":
                 logger.info(f"Deleted preset '{selected_name}' was active. Loading 'Default' preset into UI.")
                 # Load the 'Default' preset into the main UI
                 default_preset_data = get_preset("Default")
                 if default_preset_data:
                     update_config_ui_callback(
                         default_preset_data.get("system_prompt", ""),
                         default_preset_data.get("few_shot_examples", []),
                         "Default",
                     )
                 else:
                      logger.error("Could not load 'Default' preset after deleting active one.")
                      # Clear UI or show error? Clear for now.
                      update_config_ui_callback("", [], "None")


        else:
            # Error message handled within delete_preset logging
            status_text.value = f"错误：删除预设 '{selected_name}' 失败。"
            status_text.color = ft.colors.ERROR
            logger.error(f"Failed to delete preset '{selected_name}'.")

        # Removed window_exists check
        dialog_ref.update() # Update dialog content

    # --- Dialog Content Layout ---
    content = ft.Column(
        [
            ft.Text("加载预设到主配置表单", weight=ft.FontWeight.BOLD),
            ft.Row([preset_select_dd, ft.ElevatedButton("加载选中", icon=ft.icons.INPUT, on_click=load_selected_preset, tooltip="将选定预设的系统提示和示例加载到主配置表单中，并关闭此对话框。")]),
            ft.Divider(),
            ft.Text("保存当前主配置表单中的提示/示例为预设", weight=ft.FontWeight.BOLD),
            ft.Row([new_preset_name_tf, ft.ElevatedButton("保存", icon=ft.icons.SAVE, on_click=save_current_as_preset, tooltip="使用上面输入的名称保存主配置表单中当前的系统提示和示例。")]),
            ft.Divider(),
            ft.Text("删除预设", weight=ft.FontWeight.BOLD),
            ft.ElevatedButton("删除选中预设", icon=ft.icons.DELETE_FOREVER, on_click=delete_selected_preset, color=ft.colors.RED, tooltip="删除上面下拉框中选定的预设 ('Default' 除外)"),
            ft.Divider(),
            status_text, # Feedback area
        ],
        spacing=15, # Increased spacing
        tight=True, # Make column take minimum vertical space
        width=500, # Set a width for the dialog content
        height=350, # Set a height to prevent excessive growth
        scroll=ft.ScrollMode.ADAPTIVE, # Allow scrolling if content overflows
    )

    return content
