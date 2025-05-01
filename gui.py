# --- Standard Library Imports ---
import asyncio
import functools
import os
import pathlib
import subprocess # <-- Import subprocess
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
    # REMOVED: update_output_display,
    update_status_display,
    update_dashboard_info_display,  # Import the dashboard info update function
    update_audio_level_display,  # Import the audio level update function
)
# --- New Log Tab Imports ---
from gui_log import (
    create_log_elements,
    create_log_tab_content,
    update_log_display,
    clear_log_display,
    set_log_level_filter,
)
from llm_client import LLMClient
from logger_config import get_logger, setup_logging
from osc_client import VRCClient
from output_dispatcher import OutputDispatcher


# --- Logging Setup Placeholder ---
# Logging will be configured later after the page and log UI elements are created
logger = get_logger("VRCMeowGUI") # Get logger instance early, but setup is deferred


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

    # --- Create Dashboard UI Elements ---
    # Elements are created by a function in gui_dashboard
    dashboard_elements = create_dashboard_elements()
    # Extract key elements needed locally for callbacks and logic
    status_icon = dashboard_elements.get("status_icon")
    status_label = dashboard_elements.get("status_label")
    # status_row = dashboard_elements.get("status_row") # Not directly needed
    # REMOVED: output_text = dashboard_elements.get("output_text")
    toggle_button = dashboard_elements.get("toggle_button")
    progress_indicator = dashboard_elements.get("progress_indicator")
    audio_level_bar = dashboard_elements.get("audio_level_bar")  # Get the new element

    # Validate that essential dashboard elements were created
    if not all(
        [
            status_icon,
            status_label,
            # REMOVED: output_text,
            toggle_button,
            progress_indicator,
            audio_level_bar,
        ]
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
    # REMOVED: update_output_callback = functools.partial(...)
    update_audio_level_callback = functools.partial(
        update_audio_level_display,  # Function from gui_dashboard
        page,
        audio_level_bar,  # Pass the progress bar element
    )

    # --- Create Log Tab UI Elements ---
    log_elements = create_log_elements()
    log_output_listview = log_elements.get("log_output")
    clear_log_button = log_elements.get("clear_log_button")
    log_level_dropdown = log_elements.get("log_level_dropdown")

    # Validate essential log elements
    if not all([log_output_listview, clear_log_button, log_level_dropdown]):
         logger.critical("CRITICAL: Failed to create essential log UI elements.")
         # Handle error appropriately, maybe show message and exit
         page.add(ft.Text("CRITICAL ERROR: Failed to create log UI.", color=ft.colors.RED))
         page.update()
         return

    # --- Create Log Update Callback ---
    # This will be called periodically to pull logs from the queue
    log_update_callback = functools.partial(
        update_log_display, # Function from gui_log
        page,
        log_output_listview,
    )

    # --- Initialize Logging (NOW that page and log elements exist) ---
    # Pass the log update callback to the setup function
    setup_logging(log_update_callback=log_update_callback)
    logger.info("Logging setup complete with Flet handler.") # Now logger is fully configured

    # --- Initialize Core Components (based on config) ---
    logger.info("Initializing core components...")
    try:
        # 1. LLMClient (if enabled)
        if config.get("llm.enabled", False):
            app_state.llm_client = LLMClient()
            if not app_state.llm_client.enabled:
                logger.warning(
                    "LLMClient initialization failed or API Key missing, LLM processing will be disabled."
                )
                app_state.llm_client = None  # Ensure it's None if disabled
            else:
                logger.info("LLMClient initialized.")
        else:
            logger.info("LLM processing disabled by config.")
            app_state.llm_client = None

        # 2. VRCClient (if enabled) - Defer initialization and start
        # We will initialize and start it in initialize_async_components below

        # 3. OutputDispatcher will be initialized asynchronously below.

        # 4. AudioManager will be initialized asynchronously below.
        # app_state.audio_manager = AudioManager(...) # Removed from here

    except Exception as init_err: # Keep the overall try/except for LLMClient init
        logger.critical(
            f"CRITICAL ERROR during core component initialization: {init_err}",
            exc_info=True,
        )
        gui_utils.show_error_banner(page, f"核心组件初始化失败: {init_err}")
        # Depending on the error, we might want to return or disable features
        # For now, log and show banner, hoping some parts might still work.

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
    # REMOVED: update_output_callback = functools.partial(...)
    update_audio_level_callback = functools.partial(
        update_audio_level_display,  # Function from gui_dashboard
        page,
        audio_level_bar,  # Pass the progress bar element
    )
    # Log update callback was created earlier before setup_logging

    # --- Core Application Logic Handlers (Start/Stop) ---
    # These remain in gui.py as they orchestrate multiple components
    # REMOVED: Definitions of update_status_display, update_output_display
    # REMOVED: Definition of get_control_value
    # REMOVED: Definitions of save_config_handler, reload_config_controls, reload_config_handler
    # REMOVED: Definitions of _create_example_row_internal, add_example_handler

    # --- Application Restart Logic ---
    async def restart_application(page: ft.Page, app_state: AppState):
        """Performs cleanup and restarts the application."""
        logger.info("Initiating application restart sequence...")

        # Ensure audio processes are stopped
        if app_state.is_running:  # is_running now refers to audio state
            logger.info("Restart: Stopping active audio recording...")
            try:
                await _stop_recording_internal()  # Stop AudioManager if running
                logger.info("Restart: AudioManager stop requested (callback will confirm final state).")
                # Add a slightly longer delay to allow AudioManager's threads to potentially finish cleanup
                logger.info("Restart: Waiting briefly for audio cleanup...")
                await asyncio.sleep(0.5) # Increased from 0.2
                logger.info("Restart: Finished waiting after audio stop request.")
            except Exception as audio_stop_err:
                 logger.error(f"Restart: Error during _stop_recording_internal: {audio_stop_err}", exc_info=True)
        else:
            logger.info("Restart: Audio recording not active, skipping audio stop.")

        # Stop VRCClient if it exists and is running
        if app_state.vrc_client:
            logger.info("Restart: Stopping VRCClient...")
            try:
                await app_state.vrc_client.stop()
                logger.info("Restart: VRCClient stopped successfully.")
            except Exception as vrc_stop_err:
                logger.error(
                    f"Restart: Error stopping VRCClient: {vrc_stop_err}",
                    exc_info=True,
                )
            # Still clear reference even if stop failed
            app_state.vrc_client = None
        else:
            logger.info("Restart: VRCClient not active, skipping VRC stop.")


        # Attempt restart by launching a new detached process and exiting the current one
        logger.critical(">>> Preparing to launch new application instance and exit <<<")
        try:
            # Ensure sys.executable and sys.argv are valid
            exec_path = sys.executable
            exec_args = [exec_path] + sys.argv
            logger.debug(f"Restart: Launching executable: {exec_path}")
            logger.debug(f"Restart: Using arguments: {exec_args}")

            if not exec_path or not exec_args:
                raise RuntimeError(
                    "sys.executable or sys.argv is not available for restart."
                )

            # Platform-specific flags for detaching the process
            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.DETACHED_PROCESS # Detach on Windows

            # Launch the new process
            logger.info(f"Launching new process: {exec_args} with creationflags={creationflags}")
            subprocess.Popen(exec_args, creationflags=creationflags)

            # Successfully launched, now exit the current process by closing the window
            logger.info("New process launched. Closing current application window...")
            try:
                page.window_destroy()
                # Note: Code execution might stop here as the window closes.
                logger.info("Window close requested.") # This log might not always appear
            except Exception as destroy_ex:
                 # This might happen if the page context is already invalid
                logger.error(f"Error requesting window destroy during restart: {destroy_ex}")
                logger.info("Attempting sys.exit(0) as fallback.")
                sys.exit(0) # Force exit if window destroy fails

        except Exception as launch_ex:
            # If launching the new process fails
            logger.critical(f"启动新应用程序实例时出错: {launch_ex}", exc_info=True)
            gui_utils.show_error_banner(page, f"启动新实例失败: {launch_ex}")
            # Do not destroy the window here, allow the current instance to continue running

    # --- Log Tab Handlers ---
    async def clear_log_handler(e: ft.ControlEvent):
        """Handles clicks on the clear log button."""
        logger.info("Clearing GUI log display.")
        clear_log_display(page, log_output_listview) # Call function from gui_log

    async def log_level_change_handler(e: ft.ControlEvent):
        """Handles changes in the log level dropdown."""
        selected_level = e.control.value
        logger.info(f"GUI log level filter changed to: {selected_level}")
        set_log_level_filter(selected_level) # Call function from gui_log
        # The update_log_display function (called periodically) will use the new filter
        # We might force an immediate update here if desired, but periodic is usually fine.
        log_update_callback() # Force immediate update after level change


    async def _start_recording_internal():
        """Internal logic for starting the process."""
        # Status update now handles button state during processing
        update_status_callback("正在启动...", is_running=None, is_processing=True)
        # page.update() # update_status_callback calls page.update()

        try:
            # --- Start AudioManager ---
            # Components (LLM, VRC, Dispatcher, AudioManager) are now initialized earlier.
            # We just need to start the AudioManager here.
            if not app_state.audio_manager:
                # This shouldn't happen if initialization worked, but handle defensively.
                error_msg = "错误：AudioManager 未初始化。"
                logger.error(error_msg)
                update_status_callback(error_msg, is_running=False, is_processing=False)
                gui_utils.show_error_banner(page, error_msg)
                return

            # Check for Dashscope API key before starting audio, as STT needs it.
            dashscope_api_key = config.get("dashscope.api_key")
            if not dashscope_api_key:
                error_msg = "错误：Dashscope API Key 未设置，无法启动语音识别。"
                logger.error(error_msg)
                update_status_callback(error_msg, is_running=False, is_processing=False)
                gui_utils.show_error_banner(page, error_msg)
                return

            logger.info("Requesting AudioManager start...")
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
            logger.info("Requesting AudioManager stop...")
            try:
                # Run the potentially blocking stop() in a thread via asyncio.to_thread
                # Wrap this await in asyncio.wait_for to prevent hangs during restart
                stop_timeout = 10.0 # Seconds to wait for AudioManager.stop() to complete
                logger.info(f"Waiting up to {stop_timeout}s for AudioManager.stop() to complete...")
                await asyncio.wait_for(
                    asyncio.to_thread(app_state.audio_manager.stop),
                    timeout=stop_timeout
                )
                logger.info("AudioManager stop request completed within timeout.")
                # The final status update ("Stopped", "Error") should come from
                # the AudioManager's status_callback when its threads fully exit.
            except asyncio.TimeoutError:
                logger.warning(f"AudioManager.stop() did not complete within {stop_timeout}s timeout. Proceeding with stop/restart anyway.")
                # Force UI update to indicate potential issue but still stopped state
                update_status_callback("已停止 (超时)", is_running=False, is_processing=False)
            except Exception as am_stop_err:
                logger.error(
                    f"Error requesting AudioManager stop: {am_stop_err}", exc_info=True
                )
                # Force status update on error during stop request
                update_status_callback(
                    "停止时出错", is_running=False, is_processing=False
                )
                gui_utils.show_error_banner(
                    page, f"停止 AudioManager 时出错: {am_stop_err}"
                )
        else:
            logger.warning("Stop requested, but AudioManager not found in state.")
            # Ensure UI reflects stopped state if manager is missing
            update_status_callback("已停止", is_running=False, is_processing=False)

        # Do NOT stop VRCClient or nullify LLMClient/OutputDispatcher here.
        # They persist for other functions (like text input) or app lifetime.

        # Mark logical *audio recording* state as stopped
        app_state.is_running = (
            False  # This now specifically means audio recording is off
        )
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
            logger.info("Window closing event detected.")
            # Ensure audio processes are stopped
            if app_state.is_running:  # is_running now refers to audio state
                logger.info("Stopping active audio recording before closing...")
                await _stop_recording_internal()  # Stop AudioManager if running
                # Add a small delay to allow AudioManager's threads to potentially finish cleanup
                await asyncio.sleep(0.2)
            else:
                logger.info("Audio recording not active.")

            # Stop VRCClient if it exists and is running
            if app_state.vrc_client:
                logger.info("Stopping VRCClient before closing...")
                try:
                    await app_state.vrc_client.stop()
                    logger.info("VRCClient stopped.")
                except Exception as vrc_stop_err:
                    logger.error(
                        f"Error stopping VRCClient during window close: {vrc_stop_err}",
                        exc_info=True,
                    )
                app_state.vrc_client = None  # Clear reference

            # Don't destroy the window, restart the application instead
            logger.info("Attempting application restart after cleanup...")
            try:
                # Ensure sys.executable and sys.argv are valid
                if not sys.executable or not sys.argv:
                    raise RuntimeError(
                        "sys.executable or sys.argv is not available for restart."
                    )
                # Use os.execv to replace the current process
                os.execv(sys.executable, [sys.executable] + sys.argv)
            except Exception as restart_ex:
                # If restart fails, log critical error and maybe destroy window as fallback?
                logger.critical(f"重启应用程序时出错: {restart_ex}", exc_info=True)
                # Fallback: Destroy the window if restart fails to prevent hanging
                try:
                    page.window_destroy()
                except Exception as destroy_ex:
                    logger.error(
                        f"Error destroying window after failed restart: {destroy_ex}"
                    )

    # --- Bind Event Handlers ---
    toggle_button.on_click = toggle_recording  # Dashboard button
    page.on_window_event = on_window_event  # Page event
    clear_log_button.on_click = clear_log_handler # Log tab button
    log_level_dropdown.on_change = log_level_change_handler # Log tab dropdown

    # --- Define a wrapper for create_config_example_row needed by reload/save handlers ---
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
                "Cannot create example row for reload/save: few_shot_examples_column is invalid."
            )
            # Return a dummy row or raise an error? Returning dummy for now.
            return ft.Row([ft.Text("Error creating row", color=ft.colors.RED)])

    # --- Create Dashboard Info Update Callback ---
    # This partial binds the necessary arguments for updating the dashboard info display
    # --- Create Dashboard Info Update Callback ---
    # This partial binds the necessary arguments for updating the dashboard info display
    # Pass the config *instance* so the callback can access the latest .data
    update_dashboard_info_partial = functools.partial(
        update_dashboard_info_display,
        page,
        dashboard_elements,  # Pass the dashboard elements dict
        config,  # Pass the config instance itself
        # Now, inside update_dashboard_info_display, it will access config.data
        # If not, we might need to pass the config object itself and access .data inside the callback.
        # For now, let's try passing the data dictionary directly.
    )

    # Config tab buttons - Use functools.partial to bind arguments to async handlers
    # Flet will automatically run the async handler in its event loop.
    save_handler_partial = functools.partial(
        save_config_handler,
        page,
        all_config_controls,
        config, # Config instance
        create_row_wrapper_for_reload,  # Function to create few-shot rows
        update_dashboard_info_partial,  # Callback to update dashboard info
        app_state, # Pass the application state for cleanup during restart
        functools.partial(restart_application, page, app_state), # Pass the restart function bound with page and app_state
    )
    save_config_button.on_click = save_handler_partial

    reload_handler_partial = functools.partial(
        reload_config_handler,
        page,
        all_config_controls,
        config,
        create_row_wrapper_for_reload,  # Pass the same row creation function
        update_dashboard_info_partial,  # Pass the dashboard update callback
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

    # --- Create Text Input Tab Elements & Handler ---
    text_input_field = ft.TextField(
        label="在此输入文本",
        multiline=True,
        min_lines=3,
        max_lines=5,
        expand=True,  # Allow vertical expansion
        border_color=ft.colors.OUTLINE,
    )
    text_input_progress = ft.ProgressRing(
        visible=False, width=16, height=16, stroke_width=2
    )
    submit_text_button = ft.ElevatedButton(
        "发送",
        icon=ft.icons.SEND,
        tooltip="处理并发送输入的文本",
        # on_click will be assigned below
    )

    async def submit_text_handler(e: ft.ControlEvent):
        """Handles clicks on the text input submit button."""
        input_text = text_input_field.value
        if not input_text or not input_text.strip():
            gui_utils.show_error_banner(page, "请输入要发送的文本。")
            return

        logger.info(f"Text input submitted: '{input_text[:50]}...'")
        submit_text_button.disabled = True
        text_input_progress.visible = True
        page.update()  # Show progress

        processed_text = input_text  # Default to original text
        try:
            # 1. Process with LLM (if enabled and available)
            if app_state.llm_client:
                logger.debug("Processing text with LLM...")
                llm_result = await app_state.llm_client.process_text(input_text)
                if llm_result is not None:
                    processed_text = llm_result
                    logger.info(f"LLM processed text input: '{processed_text[:50]}...'") # Log LLM result
                else:
                    logger.warning("LLM processing returned None, using original text for dispatch.")
                    # Optionally show a warning banner?
                    # gui_utils.show_banner(page, "LLM 处理失败，使用原始文本。", icon=ft.icons.WARNING_AMBER_ROUNDED, bgcolor=ft.colors.AMBER_100, icon_color=ft.colors.AMBER_700)

            # 2. Dispatch the result (original or processed)
            if app_state.output_dispatcher:
                logger.info(f"Dispatching text input result: '{processed_text[:50]}...'") # Log before dispatch
                await app_state.output_dispatcher.dispatch(processed_text)
                # REMOVED: update_output_callback(...) call
                logger.info("Text input dispatched successfully.") # Log after dispatch
                text_input_field.value = ""  # Clear input on success
            else:
                logger.error("OutputDispatcher not available, cannot dispatch text input.")
                gui_utils.show_error_banner(page, "错误：无法分发文本。")

        except Exception as ex:
            error_msg = f"处理文本输入时出错: {ex}"
            logger.error(error_msg, exc_info=True)
            gui_utils.show_error_banner(page, error_msg)
        finally:
            # Ensure UI is reset
            submit_text_button.disabled = False
            text_input_progress.visible = False
            page.update()

    submit_text_button.on_click = submit_text_handler

    # --- Create Tab Layouts ---
    dashboard_tab_layout = create_dashboard_tab_content(
        dashboard_elements
    )  # Pass created elements dict

    config_tab_layout = create_config_tab_content(
        save_button=save_config_button,
        reload_button=reload_config_button,
        all_controls=all_config_controls,  # Pass controls dict
    )

    log_tab_layout = create_log_tab_content(log_elements) # Create log tab layout

    text_input_tab_content = ft.Column(
        [
            ft.Text(
                "手动输入文本并发送，将通过与语音输入相同的处理流程（LLM -> 输出）。"
            ),
            text_input_field,
            ft.Row(
                [submit_text_button, text_input_progress],
                alignment=ft.MainAxisAlignment.END,
            ),
        ],
        spacing=10,
        # Add scroll if content might overflow, though unlikely with max_lines
        # scroll=ft.ScrollMode.ADAPTIVE,
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
                ft.Tab(text="配置", icon=ft.icons.SETTINGS, content=config_tab_layout),
                ft.Tab(
                    text="文本输入",
                    icon=ft.icons.TEXT_FIELDS,
                    content=text_input_tab_content,
                ),
                 ft.Tab( # Add the new Log tab
                    text="日志",
                    icon=ft.icons.LIST_ALT_ROUNDED,
                    content=log_tab_layout,
                ),
            ],
            expand=True,  # Make tabs fill the page width
        )
    )

    # --- Periodic Log Queue Check ---
    # Schedule a task to periodically check the log queue and update the UI
    async def periodic_log_update():
        while True:
            # Check if the page still exists and the window hasn't been destroyed
            if page and not page.window_destroyed: # Check if window is still active
                 try:
                    log_update_callback() # Call the partial function
                 except Exception as log_update_err:
                     print(f"Error in periodic log update: {log_update_err}") # Print error directly
            else:
                logger.info("Page closed or connection lost, stopping periodic log update.")
                break # Exit the loop if page is gone
            await asyncio.sleep(0.5) # Check every 500ms

    page.run_task(periodic_log_update)

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
                    logger.warning(
                        f"Skipping invalid few-shot example during initial load: {example}"
                    )
        else:
            logger.warning("'llm.few_shot_examples' in initial config is not a list.")
    else:
        logger.error(
            "Cannot populate few-shot examples: Column control not found or invalid."
        )

    # --- Initial Dashboard Info Population ---
    logger.debug("Initial population of dashboard info display.")
    # Call the partial function created earlier
    # Run it in the background as it might involve UI updates
    # --- Initial Dashboard Info Population ---
    logger.debug("Initial population of dashboard info display.")
    # Call the partial function created earlier which now takes the config instance
    # Run it in the background as it might involve UI updates
    # async def initial_dashboard_update(): # No longer needed as partial handles it
    #     try:
    #         # Pass the config instance
    #         await update_dashboard_info_display(page, dashboard_elements, config)
    #     except Exception as e:
    #         logger.error(f"Error during initial dashboard info update: {e}", exc_info=True)

    # Schedule the initial update using the partial
    # Using asyncio.create_task might be better if running within an async context already
    # but page.run_thread is safer if unsure about the context Flet runs main() in.
    # Let's try calling the partial directly first, as it uses run_thread internally.
    try:
        # Directly call the partial which uses page.run_thread internally
        # The partial now correctly holds the config instance.
        update_dashboard_info_partial()
    except Exception as e:
        logger.error(f"Error scheduling initial dashboard update: {e}", exc_info=True)

    # --- Async Component Initialization ---
    async def initialize_async_components():
        """Initialize and start components that require a running event loop."""
        logger.info("Starting async component initialization...")

        # Initialize and start VRCClient if enabled
        if config.get("outputs.vrc_osc.enabled", False):
            osc_address = config.get("outputs.vrc_osc.address", "127.0.0.1")
            osc_port = config.get("outputs.vrc_osc.port", 9000)
            osc_interval = config.get("outputs.vrc_osc.message_interval", 1.333)
            try:
                app_state.vrc_client = VRCClient(
                    address=osc_address, port=osc_port, interval=osc_interval
                )
                # Assign the created client to the dispatcher
                if app_state.output_dispatcher:
                    app_state.output_dispatcher.vrc_client_instance = (
                        app_state.vrc_client
                    )
                    logger.info("VRCClient instance assigned to OutputDispatcher.")
                else:
                    logger.warning(
                        "OutputDispatcher not available when VRCClient initialized."
                    )

                # Start VRCClient using asyncio.create_task now that loop is running
                asyncio.create_task(app_state.vrc_client.start())
                logger.info("VRCClient initialized and start task created.")
            except Exception as vrc_err:
                error_msg = f"Error initializing or starting VRCClient: {vrc_err}"
                logger.error(error_msg, exc_info=True)
                gui_utils.show_error_banner(page, error_msg)
                app_state.vrc_client = None  # Ensure it's None on error
                # Ensure dispatcher doesn't hold a reference if init failed
                if app_state.output_dispatcher:
                    app_state.output_dispatcher.vrc_client_instance = None
        else:
            logger.info("VRC OSC output disabled, skipping VRCClient initialization.")
            app_state.vrc_client = None
            # Ensure dispatcher doesn't hold a reference
            if app_state.output_dispatcher: # Check if dispatcher exists before assigning None
                app_state.output_dispatcher.vrc_client_instance = None

        # --- Initialize OutputDispatcher (now that VRCClient is potentially ready) ---
        try:
            logger.info("Initializing OutputDispatcher asynchronously...")
            app_state.output_dispatcher = OutputDispatcher(
                vrc_client_instance=app_state.vrc_client, # Pass the actual client instance (or None)
                # REMOVED: gui_output_callback=update_output_callback,
            )
            logger.info("OutputDispatcher initialized.")
        except Exception as disp_err:
            logger.critical(f"CRITICAL ERROR initializing OutputDispatcher: {disp_err}", exc_info=True)
            gui_utils.show_error_banner(page, f"OutputDispatcher 初始化失败: {disp_err}")
            # Cannot proceed without dispatcher for audio/text input
            return # Stop further async initialization

        # --- Initialize AudioManager (now that Dispatcher is ready) ---
        try:
            logger.info("Initializing AudioManager asynchronously...")
            app_state.audio_manager = AudioManager(
                llm_client=app_state.llm_client, # LLM client was initialized synchronously
                output_dispatcher=app_state.output_dispatcher, # Pass the newly created dispatcher
                status_callback=update_status_callback,
                audio_level_callback=update_audio_level_callback,
            )
            logger.info("AudioManager initialized.")
        except Exception as am_err:
            logger.critical(f"CRITICAL ERROR initializing AudioManager: {am_err}", exc_info=True)
            gui_utils.show_error_banner(page, f"AudioManager 初始化失败: {am_err}")
            # Audio input will not work, but text input might still function

        logger.info("Async component initialization finished.")

    # Schedule the async initialization task to run
    page.run_task(initialize_async_components)

    # --- Final Page Update ---
    page.update()

# 注意：此文件不再包含 if __name__ == "__main__": ft.app(...)
# 这将移至 main.py
