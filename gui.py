# --- Standard Library Imports ---
import asyncio
import functools
import os
import pathlib
import sys
from typing import Optional, Dict  # Added Callable here

# --- Third-Party Imports ---
import flet as ft

# --- Project Setup & CWD Adjustment ---
# This block MUST run before local imports to ensure modules are found,
# especially when running as a packaged application.
script_dir = pathlib.Path(__file__).parent.resolve()
project_root = script_dir
config_file_path = project_root / "config.yaml"
current_cwd = pathlib.Path.cwd()

if not (current_cwd / "config.yaml").exists() and config_file_path.exists():
    print(
        f"[INFO] Initial CWD '{current_cwd}' seems incorrect (config.yaml not found)."
    )
    print(f"[INFO] Changing CWD to detected project root: '{project_root}'")
    os.chdir(project_root)
    print(f"[INFO] Current CWD after change: '{os.getcwd()}'")
    # Add project root to sys.path if not already present
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"[INFO] Added '{project_root}' to sys.path")
else:
    print(
        f"[INFO] Initial CWD '{current_cwd}' seems correct or config.yaml not found at expected root."
    )
# --- End Project Setup ---

# --- Local Project Imports (after CWD setup) ---
from audio_recorder import AudioManager
from config import config  # Import singleton and class
import gui_utils
from gui_config import (
    create_audio_controls,
    create_config_example_row,
    create_config_tab_content,
    create_console_output_controls,
    create_dashscope_controls,
    create_file_output_controls,
    create_llm_controls,
    create_logging_controls,
    create_vrc_osc_controls,
    add_example_handler,
    reload_config_handler,
    save_config_handler,
)
from gui_dashboard import (
    create_dashboard_elements,
    create_dashboard_tab_content,
    update_output_display,
    update_status_display,
)
from llm_client import LLMClient
from logger_config import get_logger, setup_logging
from osc_client import VRCClient
from output_dispatcher import OutputDispatcher


# --- Initialize Logging (after imports and CWD setup) ---
setup_logging()
logger = get_logger("VRCMeowGUI")


class AppState:
    """简单类，用于在回调函数之间共享状态"""

    def __init__(self):
        self.is_running = False
        self.audio_manager: Optional["AudioManager"] = (
            None  # Use quotes for forward refs if needed
        )
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
    page.window_resizable = True
    page.padding = 10

    app_state = AppState()

    # --- Central Dictionary for Config Controls ---
    all_config_controls: Dict[str, ft.Control] = {}

    # --- Get Initial Config ---
    initial_config_data = {}
    try:
        if config:
            initial_config_data = config.data
        else:
            logger.error("Config object not available during GUI initialization.")
            # Potentially show an error message to the user here?
    except Exception as e:
        logger.error(
            f"Error accessing config data during GUI initialization: {e}", exc_info=True
        )
    # Ensure config object is available, even if fallback is used
    # The 'config' variable is imported from config.py
    if not config:
        logger.critical(
            "CRITICAL: Config object failed to initialize. GUI cannot function correctly."
        )
        # Display a critical error message in the UI if possible
        page.add(
            ft.Text(
                "CRITICAL ERROR: Failed to load configuration. Please check logs.",
                color=ft.colors.RED,
            )
        )
        page.update()  # Try to show the error
        return  # Stop GUI setup

    # --- Create Config Tab Controls ---
    # These are created here because the layout function in gui_config needs them.
    # Functions imported from gui_config are used.
    all_config_controls.update(create_dashscope_controls(initial_config_data))

    all_config_controls.update(create_audio_controls(initial_config_data))
    all_config_controls.update(create_llm_controls(initial_config_data))
    all_config_controls.update(create_vrc_osc_controls(initial_config_data))
    all_config_controls.update(create_console_output_controls(initial_config_data))
    all_config_controls.update(create_file_output_controls(initial_config_data))
    all_config_controls.update(create_logging_controls(initial_config_data))

    # Extract key config controls needed for handlers/logic
    few_shot_examples_column = all_config_controls.get("llm.few_shot_examples_column")
    add_example_button = all_config_controls.get("llm.add_example_button")
    # Ensure they are the correct type or provide fallbacks
    if not isinstance(few_shot_examples_column, ft.Column):
        logger.warning(
            "Few-shot column control not found or invalid type, creating fallback."
        )
        few_shot_examples_column = ft.Column(visible=False)  # Fallback, hide it
        all_config_controls["llm.few_shot_examples_column"] = (
            few_shot_examples_column  # Add fallback to dict
        )
    if not isinstance(add_example_button, ft.TextButton):
        logger.warning(
            "Add example button control not found or invalid type, creating fallback."
        )
        add_example_button = ft.TextButton(
            "添加示例 (Error)", visible=False
        )  # Fallback, hide it
        all_config_controls["llm.add_example_button"] = (
            add_example_button  # Add fallback to dict
        )

    # --- Create Config Save/Reload Buttons ---
    save_config_button = ft.ElevatedButton(
        "保存配置",
        on_click=None,  # Handlers assigned later
        icon=ft.icons.SAVE,
        tooltip="将当前设置写入 config.yaml",
    )
    reload_config_button = ft.ElevatedButton(
        "从文件重载",
        on_click=None,  # Handlers assigned later
        icon=ft.icons.REFRESH,
        tooltip="放弃当前更改并从 config.yaml 重新加载",
    )

    # --- Create Dashboard UI Elements ---
    # Elements are created by a function in gui_dashboard
    dashboard_elements = create_dashboard_elements()
    # Extract key elements needed locally for callbacks and logic
    status_icon = dashboard_elements.get("status_icon")
    status_label = dashboard_elements.get("status_label")
    # status_row = dashboard_elements.get("status_row") # Not directly needed
    output_text = dashboard_elements.get("output_text")
    toggle_button = dashboard_elements.get("toggle_button")
    progress_indicator = dashboard_elements.get("progress_indicator")

    # Validate that essential dashboard elements were created
    if not all(
        [status_icon, status_label, output_text, toggle_button, progress_indicator]
    ):
        logger.critical(
            "CRITICAL: Failed to create essential dashboard UI elements. GUI cannot function."
        )
        page.add(
            ft.Text(
                "CRITICAL ERROR: Failed to create dashboard UI. Check logs.",
                color=ft.colors.RED,
            )
        )
        page.update()
        return  # Stop GUI setup

    # --- Partial Callbacks (Binding arguments) ---
    # Create partial functions for callbacks that need page and UI elements
    # These are passed to AudioManager and OutputDispatcher
    update_status_callback = functools.partial(
        update_status_display,  # Function from gui_dashboard
        page,
        status_icon,
        status_label,
        toggle_button,
        progress_indicator,
    )
    update_output_callback = functools.partial(
        update_output_display,  # Function from gui_dashboard
        page,
        output_text,
    )

    # --- Core Application Logic Handlers (Start/Stop) ---
    # These remain in gui.py as they orchestrate multiple components
    # REMOVED: Definitions of update_status_display, update_output_display
    # REMOVED: Definition of get_control_value
    # REMOVED: Definitions of save_config_handler, reload_config_controls, reload_config_handler
    # REMOVED: Definitions of _create_example_row_internal, add_example_handler
    async def _start_recording_internal():
        """Internal logic for starting the process."""
        # Status update now handles button state during processing
        update_status_callback(
            "正在启动...", is_running=None, is_processing=True
        )  # Use the callback
        # page.update() # update_status_callback calls page.update()

        try:
            # --- Initialize Components ---
            logger.info("GUI requesting start, initializing components...")

            # Check critical config (can re-check here or rely on earlier checks)
            # Use the config object directly (imported singleton)
            dashscope_api_key = config.get("dashscope.api_key")
            if not dashscope_api_key:
                error_msg = "错误：Dashscope API Key 未设置。"
                logger.error(error_msg)
                update_status_callback(  # Use the callback
                    error_msg, is_running=False, is_processing=False
                )
                # Show banner as well
                gui_utils.show_error_banner(page, error_msg)
                return

            # 1. VRCClient (if enabled)
            vrc_osc_enabled = config.get(
                "outputs.vrc_osc.enabled", False
            )  # Access via singleton
            if vrc_osc_enabled:
                osc_address = config.get(
                    "outputs.vrc_osc.address", "127.0.0.1"
                )  # Access via singleton
                osc_port = config.get(
                    "outputs.vrc_osc.port", 9000
                )  # Access via singleton
                osc_interval = config.get(
                    "outputs.vrc_osc.message_interval", 1.333
                )  # Access via singleton
                try:
                    app_state.vrc_client = VRCClient(
                        address=osc_address, port=osc_port, interval=osc_interval
                    )
                    await app_state.vrc_client.start()
                    logger.info("VRCClient initialized and started.")
                except Exception as vrc_err:
                    error_msg = f"Error initializing VRCClient: {vrc_err}"
                    logger.error(error_msg, exc_info=True)
                    update_status_callback(
                        error_msg, is_running=False, is_processing=False
                    )  # Use partial
                    gui_utils.show_error_banner(page, error_msg)
                    return  # Stop the start process
            else:
                logger.info(
                    "VRC OSC output disabled, skipping VRCClient initialization."
                )
                app_state.vrc_client = None

            # 2. LLMClient (if enabled)
            llm_enabled = config.get("llm.enabled", False)  # Access via singleton
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

            # 3. OutputDispatcher (pass VRC client and GUI output callback)
            app_state.output_dispatcher = OutputDispatcher(
                vrc_client_instance=app_state.vrc_client,
                gui_output_callback=update_output_callback,  # Pass the partial callback
            )
            logger.info("OutputDispatcher initialized.")

            # 4. AudioManager (pass LLM client, dispatcher, and status callback)
            app_state.audio_manager = AudioManager(
                llm_client=app_state.llm_client,
                output_dispatcher=app_state.output_dispatcher,
                status_callback=update_status_callback,  # Pass the partial callback
            )
            logger.info("AudioManager initialized.")

            # --- Start AudioManager ---
            # AudioManager.start() 会启动后台线程
            app_state.audio_manager.start()
            # AudioManager.start() is synchronous but starts background threads.
            # The threads inside AudioManager will update the status via the callback.
            app_state.audio_manager.start()
            app_state.is_running = True  # Mark logical state immediately
            logger.info("AudioManager start requested. Background threads initiated.")
            # Initial status update ("Starting...") was already sent.
            # Subsequent updates ("Running", "Error", etc.) will come from AudioManager via callback.

        except Exception as ex:
            error_msg = f"启动过程中出错: {ex}"
            logger.critical(error_msg, exc_info=True)
            # Update status display to show error using the callback
            update_status_callback(
                "启动错误", is_running=False, is_processing=False
            )  # Simplified error message
            gui_utils.show_error_banner(page, error_msg)  # Show details in banner
            # Attempt cleanup even on startup error
            await _stop_recording_internal(is_error=True)  # Call internal stop logic

    async def _stop_recording_internal(is_error: bool = False):
        """Internal logic for stopping the process."""
        # Use the status callback to indicate stopping
        # Check if already stopped to prevent redundant calls, unless it's an error cleanup
        if not app_state.is_running and not is_error:
            logger.info("Stop requested but already stopped.")
            # Ensure UI is in correct stopped state via callback
            update_status_callback("已停止", is_running=False, is_processing=False)
            return

        logger.info("GUI Requesting Stop...")
        update_status_callback(
            "正在停止...", is_running=app_state.is_running, is_processing=True
        )  # Indicate processing
        # tasks_to_await = [] # Not used

        # Stop AudioManager (this handles stopping STT and audio stream)
        if app_state.audio_manager:
            logger.info("Stopping AudioManager...")
            try:
                # Run the potentially blocking stop() in a thread
                await asyncio.to_thread(app_state.audio_manager.stop)
                logger.info("AudioManager stop completed.")
            except Exception as am_stop_err:
                logger.error(
                    f"Error stopping AudioManager: {am_stop_err}", exc_info=True
                )
                # Update status even if error occurs during stop
                update_status_callback(
                    "停止时出错", is_running=False, is_processing=False
                )
                gui_utils.show_error_banner(
                    page, f"停止 AudioManager 时出错: {am_stop_err}"
                )
            # Don't nullify audio_manager here, let the status callback handle final state update
            # app_state.audio_manager = None # Removed

        # Stop VRCClient (if exists)
        if app_state.vrc_client:
            logger.info("正在停止 VRCClient...")
            try:
                await app_state.vrc_client.stop()
                logger.info("VRCClient 已停止。")
            except Exception as vrc_stop_err:
                logger.error(f"停止 VRCClient 时出错: {vrc_stop_err}", exc_info=True)
            app_state.vrc_client = None

        # Clean up other resources
        app_state.llm_client = None  # LLMClient has no explicit stop needed currently
        app_state.output_dispatcher = None  # Dispatcher has no explicit stop needed

        # Mark logical state *after* attempting stops
        app_state.is_running = False
        logger.info(
            "All components requested to stop. Final status update relies on AudioManager callback."
        )
        # The final "Stopped" or "Error" status update, along with button state reset,
        # should come from the AudioManager's status_callback when its threads fully exit.
        # If AudioManager fails to send a final status, the UI might remain in "Stopping...".

    async def toggle_recording(e: ft.ControlEvent):
        """Handles clicks on the combined Start/Stop button."""
        # Prevent rapid clicking if already processing
        if progress_indicator.visible:
            logger.warning(
                "Toggle button clicked while already processing (start/stop). Ignoring."
            )
            return

        if app_state.is_running:
            await _stop_recording_internal()
        else:
            await _start_recording_internal()

    # --- Page Close Handler ---
    async def on_window_event(e: ft.ControlEvent):
        if e.data == "close":
            logger.info("检测到窗口关闭事件。")
            # Ensure processes are stopped before closing
            if app_state.is_running:
                logger.info("Window closing, stopping background processes before restart...")
                await _stop_recording_internal()  # Call internal stop logic
            else:
                logger.info("Window closing, no active processes to stop.")

            # Don't destroy the window, restart the application instead
            logger.info("Executing application restart after window close...")
            try:
                # Ensure sys.executable and sys.argv are valid
                if not sys.executable or not sys.argv:
                    raise RuntimeError("sys.executable or sys.argv is not available for restart.")
                # Use os.execv to replace the current process
                os.execv(sys.executable, [sys.executable] + sys.argv)
            except Exception as restart_ex:
                # If restart fails, log critical error and maybe destroy window as fallback?
                logger.critical(f"重启应用程序时出错: {restart_ex}", exc_info=True)
                # Fallback: Destroy the window if restart fails to prevent hanging
                try:
                    page.window_destroy()
                except Exception as destroy_ex:
                    logger.error(f"Error destroying window after failed restart: {destroy_ex}")

    # --- Bind Event Handlers ---
    toggle_button.on_click = toggle_recording  # Dashboard button (local handler)
    page.on_window_event = on_window_event  # Page event (local handler)

    # --- Define Stop Wrapper for Save Handler ---
    # This wrapper provides a simple async function reference to _stop_recording_internal
    # without needing to pass the 'is_error' argument from the save handler.
    async def stop_wrapper():
        logger.debug("stop_wrapper called, invoking _stop_recording_internal.")
        await _stop_recording_internal()

    # Config tab buttons - Use functools.partial to bind arguments to async handlers
    # Flet will automatically run the async handler in its event loop.
    save_handler_partial = functools.partial(
        save_config_handler,
        page,
        all_config_controls,
        config,
        stop_wrapper, # Pass the stop function wrapper
    )
    save_config_button.on_click = save_handler_partial


    # Define a wrapper for create_config_example_row needed by reload_config_handler
    # This wrapper matches the signature expected by reload_config_controls
    def create_row_wrapper_for_reload(user_text: str, assistant_text: str) -> ft.Row:
        # Call the actual row creation function from gui_config, passing page and column
        # Ensure few_shot_examples_column is valid before calling
        if few_shot_examples_column and isinstance(few_shot_examples_column, ft.Column):
            return create_config_example_row(
                page, few_shot_examples_column, user_text, assistant_text
            )
        else:
            logger.error(
                "Cannot create example row for reload: few_shot_examples_column is invalid."
            )
            # Return a dummy row or raise an error? Returning dummy for now.
            return ft.Row([ft.Text("Error creating row", color=ft.colors.RED)])

    reload_handler_partial = functools.partial(
        reload_config_handler,
        page,
        all_config_controls,
        config,
        create_row_wrapper_for_reload,
    )
    reload_config_button.on_click = reload_handler_partial

    # Ensure add_example_button is valid before assigning handler
    if add_example_button:
        add_example_partial = functools.partial(
            add_example_handler, page, all_config_controls
        )
        add_example_button.on_click = add_example_partial
    else:
        logger.error("Cannot assign handler: Add example button is invalid.")

    # --- Create Tab Layouts ---
    dashboard_tab_layout = create_dashboard_tab_content(
        dashboard_elements
    )  # Pass created elements dict

    config_tab_layout = create_config_tab_content(
        save_button=save_config_button,
        reload_button=reload_config_button,
        all_controls=all_config_controls,  # Pass controls dict (includes few-shot column/button)
    )

    # --- Add Tabs to Page ---
    page.add(
        ft.Tabs(
            [
                ft.Tab(
                    text="仪表盘",
                    icon=ft.icons.DASHBOARD,
                    content=dashboard_tab_layout,  # Use correct variable
                ),
                ft.Tab(
                    text="配置", icon=ft.icons.SETTINGS, content=config_tab_layout
                ),  # Use correct variable
            ],
            expand=True,  # Make tabs fill the page width
        )
    )

    # --- Initial Population of Few-Shot Examples ---
    logger.debug("Initial population of few-shot examples UI.")
    initial_examples = initial_config_data.get("llm", {}).get("few_shot_examples", [])
    # Check if the column control exists and is the correct type
    if few_shot_examples_column and isinstance(few_shot_examples_column, ft.Column):
        if isinstance(initial_examples, list):
            for example in initial_examples:
                if (
                    isinstance(example, dict)
                    and "user" in example
                    and "assistant" in example
                ):
                    # Use the imported function, passing page and column
                    try:
                        initial_row = create_config_example_row(
                            page,  # Pass page
                            few_shot_examples_column,  # Pass column ref
                            example.get("user", ""),
                            example.get("assistant", ""),
                        )
                        few_shot_examples_column.controls.append(initial_row)
                    except Exception as row_ex:
                        logger.error(
                            f"Error creating initial few-shot row for example {example}: {row_ex}",
                            exc_info=True,
                        )
                else:
                    logger.warning(f"Skipping invalid few-shot example during initial load: {example}")
        else:
            logger.warning("'llm.few_shot_examples' in initial config is not a list.")
    else:
         logger.error("Cannot populate few-shot examples: Column control not found or invalid.")


    # --- Final Page Update ---
    page.update()

# 注意：此文件不再包含 if __name__ == "__main__": ft.app(...)
# 这将移至 main.py
