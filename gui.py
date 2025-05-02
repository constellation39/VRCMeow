# --- Standard Library Imports ---
import asyncio
import functools
import os
import pathlib
import sys
from typing import (
    Optional,
    Dict,
    TYPE_CHECKING,
)  # Added Callable here, Added TYPE_CHECKING

# --- Third-Party Imports ---
import flet as ft

# --- Third-Party Imports ---

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
from config import config  # Import singleton instance

if TYPE_CHECKING:
    from config import Config  # Import class for type hints only
import gui_utils
import prompt_presets  # Import the missing module
import gui_config  # Import the module itself
from gui_config import (
    create_audio_controls,
    # REMOVED: create_config_example_row,
    create_config_tab_content,
    create_console_output_controls,
    create_dashscope_controls,
    create_file_output_controls,
    create_llm_controls,
    create_logging_controls,
    create_vrc_osc_controls,
    # REMOVED: add_example_handler,
    reload_config_handler,
    save_config_handler,
    open_config_folder_handler,  # Import the new handler
)

# --- Preset UI Import ---
from gui_presets import create_preset_tab_content  # Import the new function
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
logger = get_logger("VRCMeowGUI")  # Get logger instance early, but setup is deferred


# --- Helper for Creating Info Rows (can be shared or duplicated) ---
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
        # --- Text Input Timer State ---
        self.text_input_timer_task: Optional[asyncio.Task] = None
        self.text_input_timer_delay: float = 5.0  # Default delay in seconds
        self.is_timer_enabled: bool = False


def main(page: ft.Page):
    """Flet GUI 主函数"""
    page.title = "VRCMeow Dashboard"
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    # 设置初始窗口大小 (可选)
    page.window_width = 600
    page.window_height = 450
    page.window_resizable = True
    # page.window_frameless = True # 启用无边框窗口
    page.padding = 10

    app_state = AppState()

    # --- Central Dictionary for Config Controls ---
    all_config_controls: Dict[str, ft.Control] = {}

    # --- Dictionary for Text Input Info Display Elements ---
    text_input_info_elements: Dict[str, ft.Control] = {}
    default_info_text_style = {"size": 12, "opacity": 0.9}
    text_input_info_elements["info_llm_label"] = ft.Text(
        "LLM: -", **default_info_text_style, selectable=True
    )
    text_input_info_elements["info_preset_label"] = ft.Text(
        "LLM 预设: -", **default_info_text_style, selectable=True
    )
    text_input_info_elements["info_vrc_label"] = ft.Text(
        "VRC OSC: -", **default_info_text_style, selectable=True
    )
    text_input_info_elements["info_file_label"] = ft.Text(
        "文件输出: -", **default_info_text_style, selectable=True
    )
    text_input_info_elements["info_config_path_label"] = ft.Text(
        "配置文件: Loading...",
        **default_info_text_style,
        selectable=True,
        color=ft.colors.SECONDARY,
    )

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
        page.add(
            ft.Text("CRITICAL ERROR: Failed to create log UI.", color=ft.colors.RED)
        )
        page.update()
        return

    # --- Create Log Update Callback ---
    # This will be called periodically to pull logs from the queue
    log_update_callback = functools.partial(
        update_log_display,  # Function from gui_log
        page,
        log_output_listview,
    )

    # --- Initialize Logging (NOW that page and log elements exist) ---
    # Pass the log update callback to the setup function
    setup_logging(log_update_callback=log_update_callback)
    logger.info(
        "Logging setup complete with Flet handler."
    )  # Now logger is fully configured

    # --- Ensure Default Preset Exists ---
    # Call this after logging is set up and before UI relies on presets
    try:
        logger.info("Ensuring default prompt preset exists...")
        prompt_presets.ensure_default_preset()
        logger.info("Default prompt preset check complete.")
    except Exception as preset_err:
        logger.error(f"Error ensuring default preset: {preset_err}", exc_info=True)
        # Optionally show a banner, but the error is logged.

    # --- Initialize Core Components (based on config) ---
    logger.info("Initializing core components (Dispatcher/VRCClient will be async)...")
    try:
        # LLMClient will be created on demand (in _start_recording_internal or submit_text_handler)
        app_state.llm_client = None  # Initialize as None

        # VRCClient will be initialized asynchronously below

        # 3. OutputDispatcher will be initialized asynchronously below.

        # 4. AudioManager will be initialized asynchronously below.
        # app_state.audio_manager = AudioManager(...) # Removed from here

    except Exception as init_err:  # Keep the overall try/except for LLMClient init
        logger.critical(
            f"CRITICAL ERROR during core component initialization: {init_err}",
            exc_info=True,
        )
        gui_utils.show_error_banner(page, f"核心组件初始化失败: {init_err}")
        # Depending on the error, we might want to return or disable features
        # For now, log and show banner, hoping some parts might still work.

    # --- Define Save Handler Partial (BEFORE creating controls) ---
    # This needs several elements that are defined later, but the partial captures references.
    # Dependencies: page, all_config_controls, config, update_dashboard_info_partial,
    #               update_llm_ui_partial, active_preset_name_label_ctrl
    # We will define the partial here, but some dependencies (like update_llm_ui_partial
    # and active_preset_name_label_ctrl) will be None initially. They get updated later.
    # The partial will use the *latest* values of these variables when it's actually called.
    save_handler_partial = functools.partial(
        save_config_handler,
        page,
        all_config_controls,  # Captures the dict reference
        config,
        None,  # dashboard_update_callback - will be assigned later
        None,  # update_llm_ui_callback - will be assigned later
        None,  # active_preset_name_label_ctrl - will be assigned later
    )

    # --- Create Config Tab Controls ---
    # These are created here because the layout function in gui_config needs them.
    # Functions imported from gui_config are used. Pass the save handler partial.
    all_config_controls.update(
        create_dashscope_controls(initial_config_data, save_handler_partial)
    )
    all_config_controls.update(
        create_audio_controls(initial_config_data, save_handler_partial)
    )
    all_config_controls.update(
        create_llm_controls(initial_config_data, save_handler_partial)
    )
    all_config_controls.update(
        create_vrc_osc_controls(initial_config_data, save_handler_partial)
    )
    all_config_controls.update(
        create_console_output_controls(initial_config_data, save_handler_partial)
    )
    all_config_controls.update(
        create_file_output_controls(initial_config_data, save_handler_partial)
    )
    all_config_controls.update(
        create_logging_controls(initial_config_data, save_handler_partial)
    )

    # REMOVED: Extract key config controls for few-shot examples
    # few_shot_examples_column = ...
    # add_example_button = ...

    # --- Preset controls are now created in gui_presets.py ---
    # REMOVED: manage_presets_button = ...
    # REMOVED: active_preset_name_label = ...

    # --- Create Config Reload Button (Save button removed) ---
    # REMOVED: save_config_button definition
    reload_config_button = ft.ElevatedButton(
        "从文件重载",
        on_click=None,  # Handler assigned later
        icon=ft.icons.REFRESH,
        tooltip="放弃当前更改并从 config.yaml 重新加载",
    )
    open_config_folder_button = ft.ElevatedButton(  # Change to ElevatedButton
        text="打开配置文件夹",  # Set button text
        icon=ft.icons.FOLDER_OPEN_OUTLINED,  # Keep the icon
        tooltip="打开配置文件夹 (包含 config.yaml, presets.yaml 等)",
        on_click=None,  # Handler assigned later
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
    # REMOVED: Definitions of update_status_display, update_output_display
    # REMOVED: Definition of get_control_value
    # REMOVED: Definitions of save_config_handler, reload_config_controls, reload_config_handler
    # REMOVED: Definitions of _create_example_row_internal, add_example_handler
    # REMOVED: restart_application function

    # --- Timer Logic ---
    def _cancel_text_timer(app_state: AppState):
        """Cancels the existing text input timer task."""
        if (
            app_state.text_input_timer_task
            and not app_state.text_input_timer_task.done()
        ):
            app_state.text_input_timer_task.cancel()
            logger.debug("Text input timer cancelled.")
        app_state.text_input_timer_task = None

    async def _timer_expired(
        app_state: AppState,
        page: ft.Page,
        text_input_field: ft.TextField,
        submit_text_button: ft.ElevatedButton,
        text_input_progress: ft.ProgressRing,
        # Need submit_text_handler ref to call it
        submit_handler_func: callable,
    ):
        """Callback executed when the timer expires."""
        logger.info(
            f"Text input timer expired after {app_state.text_input_timer_delay}s."
        )
        app_state.text_input_timer_task = None  # Clear task reference
        # Check if there's actually text to send before submitting
        if text_input_field.value and text_input_field.value.strip():
            logger.info("Timer expired, submitting text...")
            # Call the original submit handler, simulating a button click (event is None)
            # Ensure the submit handler itself cancels any *new* timer if needed
            # (though it shouldn't start one in this flow)
            await submit_handler_func(e=None)  # Pass None for the event
        else:
            logger.info("Timer expired, but text input is empty. Doing nothing.")
            # Ensure UI is in correct state if submit wasn't called
            text_input_field.disabled = False
            submit_text_button.disabled = False
            text_input_progress.visible = False
            # Removed window_exists check - page.update() handles closed pages
            page.update()

    async def _start_text_timer(
        app_state: AppState,
        page: ft.Page,
        text_input_field: ft.TextField,
        submit_text_button: ft.ElevatedButton,
        text_input_progress: ft.ProgressRing,
        submit_handler_func: callable,  # Need submit handler ref
    ):
        """Starts or restarts the text input timer if conditions are met."""
        _cancel_text_timer(app_state)  # Always cancel previous timer first

        if (
            app_state.is_timer_enabled
            and app_state.text_input_timer_delay > 0
            and text_input_field.value
            and text_input_field.value.strip()  # Only start if text exists
        ):
            logger.debug(
                f"Starting text input timer for {app_state.text_input_timer_delay}s..."
            )
            timer_callback = functools.partial(
                _timer_expired,
                app_state,
                page,
                text_input_field,
                submit_text_button,
                text_input_progress,
                submit_handler_func,  # Pass submit handler ref
            )
            # Use asyncio.create_task to run the timer coro
            app_state.text_input_timer_task = asyncio.create_task(
                _run_timer_async(app_state.text_input_timer_delay, timer_callback)
            )

    async def _run_timer_async(delay: float, callback: callable):
        """Helper async function to wait and then call the callback."""
        try:
            await asyncio.sleep(delay)
            logger.debug(f"Timer finished waiting for {delay}s.")
            await callback()  # Await the callback which might be async
        except asyncio.CancelledError:
            logger.debug("Timer task explicitly cancelled.")
            # Don't call callback if cancelled
        except Exception as e:
            logger.error(f"Error during timer execution: {e}", exc_info=True)

    # --- Log Tab Handlers ---
    async def clear_log_handler(e: ft.ControlEvent):
        """Handles clicks on the clear log button."""
        logger.info("Clearing GUI log display.")
        clear_log_display(page, log_output_listview)  # Call function from gui_log

    async def log_level_change_handler(e: ft.ControlEvent):
        """Handles changes in the log level dropdown."""
        selected_level = e.control.value
        logger.info(f"GUI log level filter changed to: {selected_level}")
        set_log_level_filter(selected_level)  # Call function from gui_log
        # The update_log_display function (called periodically) will use the new filter
        # We might force an immediate update here if desired, but periodic is usually fine.
        log_update_callback()  # Force immediate update after level change

    async def _start_recording_internal():
        """Internal logic for starting the process."""
        # Status update now handles button state during processing
        update_status_callback("正在启动...", is_running=None, is_processing=True)
        # page.update() # update_status_callback calls page.update()

        try:
            # --- Create LLMClient (if enabled) ---
            # Create ON DEMAND here to use latest config
            app_state.llm_client = None  # Reset before attempting creation
            if config.get("llm.enabled", False):
                logger.info("LLM is enabled, creating LLMClient instance...")
                try:
                    app_state.llm_client = LLMClient()
                    if not app_state.llm_client.enabled:
                        logger.warning(
                            "LLMClient created but is not enabled (e.g., missing API key). LLM processing will be skipped."
                        )
                        # Keep the instance but it won't process; AudioManager handles None client gracefully
                        # Or set to None if AudioManager requires it:
                        # app_state.llm_client = None
                    else:
                        logger.info(
                            "LLMClient instance created successfully for AudioManager."
                        )
                except Exception as llm_create_err:
                    logger.error(
                        f"Failed to create LLMClient instance: {llm_create_err}",
                        exc_info=True,
                    )
                    gui_utils.show_error_banner(
                        page, f"创建 LLM 客户端时出错: {llm_create_err}"
                    )
                    app_state.llm_client = None  # Ensure it's None on error
            else:
                logger.info("LLM is disabled in config. Skipping LLMClient creation.")
                app_state.llm_client = None

            # --- Create and Start AudioManager ---
            # Always create a NEW instance here to ensure it reads the latest config
            logger.info("Creating new AudioManager instance...")
            try:
                # Ensure necessary components (dispatcher) are available
                if not app_state.output_dispatcher:
                    raise RuntimeError(
                        "OutputDispatcher not available for AudioManager."
                    )
                # Callbacks should already be defined
                # Pass the potentially newly created llm_client instance (or None)
                app_state.audio_manager = AudioManager(
                    llm_client=app_state.llm_client,  # Pass the instance created above
                    output_dispatcher=app_state.output_dispatcher,
                    status_callback=update_status_callback,
                    audio_level_callback=update_audio_level_callback,
                )
                logger.info("New AudioManager instance created successfully.")
            except Exception as am_create_err:
                error_msg = f"创建 AudioManager 实例时出错: {am_create_err}"
                logger.critical(error_msg, exc_info=True)
                update_status_callback(error_msg, is_running=False, is_processing=False)
                gui_utils.show_error_banner(page, error_msg)
                return  # Cannot proceed if AudioManager creation fails

            # Check for Dashscope API key before starting audio, as STT needs it.
            # Access config through the singleton instance
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
                stop_timeout = (
                    10.0  # Seconds to wait for AudioManager.stop() to complete
                )
                logger.info(
                    f"Waiting up to {stop_timeout}s for AudioManager.stop() to complete..."
                )
                await asyncio.wait_for(
                    asyncio.to_thread(app_state.audio_manager.stop),
                    timeout=stop_timeout,
                )
                logger.info("AudioManager stop request completed within timeout.")
                # The final status update ("Stopped", "Error") should come from
                # the AudioManager's status_callback when its threads fully exit.
            except asyncio.TimeoutError:
                logger.warning(
                    f"AudioManager.stop() did not complete within {stop_timeout}s timeout. Proceeding with stop/restart anyway."
                )
                # Force UI update to indicate potential issue but still stopped state
                update_status_callback(
                    "已停止 (超时)", is_running=False, is_processing=False
                )
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

        # --- Clear the AudioManager instance ---
        # Do this AFTER attempting to stop it, regardless of success/failure/timeout.
        # This ensures the next start creates a fresh instance.
        logger.info("Clearing AudioManager instance reference.")
        app_state.audio_manager = None
        logger.info("Clearing LLMClient instance reference (if any).")
        app_state.llm_client = None  # Also clear LLM client created during start

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
        """Handles window events, specifically the close event."""
        if e.data == "close":
            logger.info("Window closing event detected. Initiating cleanup...")

            # Ensure audio processes are stopped
            if app_state.is_running:  # is_running now refers to audio state
                logger.info("Closing: Stopping active audio recording...")
                try:
                    # Use a slightly longer timeout for closing than for restart
                    stop_timeout = 10.0
                    logger.info(
                        f"Closing: Waiting up to {stop_timeout}s for AudioManager.stop()..."
                    )
                    await asyncio.wait_for(
                        asyncio.to_thread(app_state.audio_manager.stop),
                        timeout=stop_timeout,
                    )
                    logger.info("Closing: AudioManager stopped.")
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Closing: AudioManager.stop() timed out after {stop_timeout}s."
                    )
                except Exception as audio_stop_err:
                    logger.error(
                        f"Closing: Error stopping AudioManager: {audio_stop_err}",
                        exc_info=True,
                    )
            else:
                logger.info("Closing: Audio recording not active.")

            # Stop VRCClient if it exists and is running
            if app_state.vrc_client:
                logger.info("Closing: Stopping VRCClient...")
                try:
                    await app_state.vrc_client.stop()
                    logger.info("Closing: VRCClient stopped.")
                except Exception as vrc_stop_err:
                    logger.error(
                        f"Closing: Error stopping VRCClient: {vrc_stop_err}",
                        exc_info=True,
                    )
                app_state.vrc_client = None  # Clear reference
            else:
                logger.info("Closing: VRCClient not active.")

            # Cancel text input timer before closing
            _cancel_text_timer(app_state)
            logger.info("Closing: Text input timer cancelled.")

            # Close the window to exit the application
            logger.info("Cleanup finished. Closing application window...")
            try:
                page.window_close()
                logger.info("Window close requested.")
            except Exception as close_ex:
                logger.error(
                    f"Error requesting window close: {close_ex}", exc_info=True
                )
                logger.info("Attempting sys.exit(0) as fallback.")
                sys.exit(0)  # Force exit if window close fails

    # --- Bind Event Handlers ---
    toggle_button.on_click = toggle_recording  # Dashboard button
    page.on_window_event = on_window_event  # Page event
    clear_log_button.on_click = clear_log_handler  # Log tab button
    log_level_dropdown.on_change = log_level_change_handler  # Log tab dropdown

    # REMOVED: Define a wrapper for create_config_example_row
    # def create_row_wrapper_for_reload(...): ...

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

    # --- Define LLM UI Update Callback (Moved BEFORE use in save/reload/preset handlers) ---
    # This function will be passed to the preset dialog and reload handler
    # It uses the update_llm_config_ui function from gui_config
    # It now needs the active_preset_name_label control passed explicitly
    # We will create this partial *after* creating the preset tab content below
    update_llm_ui_partial = None  # Placeholder

    # Config tab buttons - Use functools.partial to bind arguments to async handlers
    # We will create these partials *after* creating the preset tab content below
    # Flet will automatically run the async handler in its event loop.
    # save_handler_partial is defined earlier now
    # REMOVED: save_config_button.on_click assignment

    reload_handler_partial = None  # Placeholder
    reload_config_button.on_click = None  # Placeholder

    open_folder_handler_partial = None  # Placeholder
    open_config_folder_button.on_click = None  # Placeholder

    # --- Create Preset Tab Content ---
    # This function now returns a dictionary with content and key controls
    # Pass the config instance to initialize with the active preset
    # The update callback will be assigned later after the wrapper is defined.
    preset_tab_elements = create_preset_tab_content(
        page=page,
        config_instance=config,  # Pass the config instance
        update_config_ui_callback=None,  # Assigned later
    )
    preset_tab_layout = preset_tab_elements.get("content")
    # Extract the label control needed for callbacks
    active_preset_name_label_ctrl = preset_tab_elements.get("active_preset_name_label")
    if not active_preset_name_label_ctrl:
        logger.critical(
            "CRITICAL: Active preset name label control not returned from create_preset_tab_content!"
        )
        active_preset_name_label_ctrl = ft.Text(
            "Error: Label Missing", color=ft.colors.RED
        )  # Fallback

    # --- REMOVED: update_text_input_few_shot_display function ---
    # --- REMOVED: update_all_preset_displays wrapper function ---
    # --- REMOVED: update_all_preset_displays_partial partial function ---

    # --- Create Partial for Updating ONLY the Preset Tab Label ---
    # This uses the function from gui_config directly.
    update_preset_tab_label_partial = functools.partial(
        gui_config.update_llm_config_ui,  # Function from gui_config
        page,
        all_config_controls,
        active_preset_name_label_ctrl,  # The label control *in the Preset Tab*
        # The active_preset_name argument is provided when the partial is called
    )
    logger.debug("Created partial for updating Preset Tab label.")

    # --- New Function to Update Text Input Info Display (Moved Here) ---
    def update_text_input_info_display(
        page: ft.Page,
        elements: Dict[str, ft.Control],  # Use the text_input_info_elements dict
        config_instance: "Config",
    ):
        """线程安全地更新文本输入选项卡上的静态配置信息显示"""
        if not page or not elements or not config_instance:
            logger.warning(
                "update_text_input_info_display called with missing arguments."
            )
            return

        try:
            config_data = config_instance.data
            if not config_data:
                logger.warning(
                    "Config instance provided but its data is empty (Text Input Info)."
                )
                return
        except Exception as e:
            logger.error(
                f"Error accessing config_instance.data (Text Input Info): {e}",
                exc_info=True,
            )
            return

        # Extract relevant info (LLM, Preset, VRC, File, Config Path)
        llm_conf = config_data.get("llm", {})
        llm_enabled = llm_conf.get("enabled", False)
        llm_model = llm_conf.get("model", "未知")
        llm_preset = llm_conf.get("active_preset_name", "Default")
        llm_info = f"LLM: {'启用' if llm_enabled else '禁用'}"
        if llm_enabled:
            llm_info += f" ({llm_model})"
        preset_info = f"LLM 预设: {llm_preset}"

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

        # Get control references from the passed elements dict
        llm_label = elements.get("info_llm_label")
        preset_label = elements.get("info_preset_label")
        vrc_label = elements.get("info_vrc_label")
        file_label = elements.get("info_file_label")
        config_path_label = elements.get("info_config_path_label")

        def update_info_ui():
            controls_to_update = []
            if llm_label and isinstance(llm_label, ft.Text):
                llm_label.value = llm_info
                controls_to_update.append(llm_label)
            if preset_label and isinstance(preset_label, ft.Text):
                preset_label.value = preset_info
                controls_to_update.append(preset_label)
            if vrc_label and isinstance(vrc_label, ft.Text):
                vrc_label.value = vrc_info
                controls_to_update.append(vrc_label)
            if file_label and isinstance(file_label, ft.Text):
                file_label.value = file_info
                controls_to_update.append(file_label)
            if config_path_label and isinstance(config_path_label, ft.Text):
                loaded_path = getattr(config_instance, "loaded_config_path", "Unknown")
                config_path_label.value = f"配置文件: {loaded_path}"
                controls_to_update.append(config_path_label)

            try:
                if page and page.controls and controls_to_update:
                    page.update(*controls_to_update)
                elif page and not controls_to_update:
                    logger.debug("No text input info controls found/updated.")
                elif page:
                    logger.warning(
                        "Page has no controls, skipping update in update_text_input_info_display."
                    )
            except Exception as e:
                logger.error(
                    f"Error during page.update in update_text_input_info_display: {e}",
                    exc_info=True,
                )

        try:
            if page and page.controls is not None:
                page.run_thread(update_info_ui)
            elif page:
                logger.warning(
                    "Page object seems invalid, skipping run_thread in update_text_input_info_display."
                )
        except Exception as e:
            logger.error(
                f"Error calling page.run_thread in update_text_input_info_display: {e}",
                exc_info=True,
            )

    # --- Create Partial for Text Input Info Update ---
    # Define this *after* the function it uses is defined
    update_text_input_info_partial = functools.partial(
        update_text_input_info_display,
        page,
        text_input_info_elements,  # Pass the specific elements dict
        config,  # Pass the config instance
    )
    logger.debug("Created partial for updating Text Input info display.")

    # --- Re-create Preset Tab Content, passing the direct partial ---
    # Re-create the preset tab elements, passing the *new wrapper partial* as the callback
    preset_tab_elements = create_preset_tab_content(
        page=page,
        config_instance=config,  # Pass config instance again
        update_config_ui_callback=update_preset_tab_label_partial,  # Pass the direct partial
    )
    preset_tab_layout = preset_tab_elements.get("content")
    # Re-fetch the label control from the potentially re-created elements
    active_preset_name_label_ctrl_check = preset_tab_elements.get(
        "active_preset_name_label"
    )
    if not active_preset_name_label_ctrl_check:
        logger.critical(
            "CRITICAL: Active preset name label control not returned from create_preset_tab_content (2nd attempt)!"
        )
        # Use the fallback created earlier if this fails again
        if not active_preset_name_label_ctrl:
            active_preset_name_label_ctrl = ft.Text(
                "Error: Label Missing", color=ft.colors.RED
            )
    elif not active_preset_name_label_ctrl:
        # If the first attempt failed but second succeeded, use the new one
        active_preset_name_label_ctrl = active_preset_name_label_ctrl_check

    # --- Assign LLM Model Refresh Handler ---
    llm_refresh_button = all_config_controls.get("llm.model_refresh_button")
    if llm_refresh_button and isinstance(llm_refresh_button, ft.IconButton):
        llm_refresh_handler_partial = functools.partial(
            gui_config.fetch_and_update_llm_models_dropdown,  # Use function from gui_config
            page,
            all_config_controls,
        )
        llm_refresh_button.on_click = llm_refresh_handler_partial
        logger.debug("Assigned LLM model refresh handler.")
    else:
        logger.warning(
            "LLM model refresh button not found or invalid, handler not assigned."
        )

    # --- Update Save Handler Partial with Late-Bound Dependencies ---
    # Now that dashboard update, Preset Tab label update, and preset label control exist, update the partial's args
    save_handler_partial.keywords["dashboard_update_callback"] = (
        update_dashboard_info_partial
    )
    # Use the direct partial for updating the Preset Tab label
    save_handler_partial.keywords["update_llm_ui_callback"] = (
        update_preset_tab_label_partial
    )
    save_handler_partial.keywords["active_preset_name_label_ctrl"] = (
        active_preset_name_label_ctrl
    )
    # Add the text input info update callback
    save_handler_partial.keywords["text_input_info_update_callback"] = (
        update_text_input_info_partial
    )
    logger.debug(
        "Updated save_handler_partial with late-bound callbacks (Preset Tab label, Text Input Info) and controls."
    )

    # --- Assign Reload Handler ---
    # REMOVED: save_config_button.on_click assignment

    reload_handler_partial = functools.partial(
        reload_config_handler,
        page,
        all_config_controls,
        config,
        update_dashboard_info_partial,  # Pass the dashboard update callback
        # Pass the direct Preset Tab label update callback and label control
        update_llm_ui_callback=update_preset_tab_label_partial,  # Use direct partial
        active_preset_name_label_ctrl=active_preset_name_label_ctrl,
        # Add the text input info update callback
        text_input_info_update_callback=update_text_input_info_partial,
    )
    reload_config_button.on_click = reload_handler_partial

    # --- Assign Open Config Folder Handler ---
    open_folder_handler_partial = functools.partial(
        open_config_folder_handler,
        page,
        config,  # Pass the config instance
    )
    open_config_folder_button.on_click = open_folder_handler_partial

    # --- REMOVED: Preset Dialog Handlers ---
    # async def open_preset_dialog(e: ft.ControlEvent): ...
    # def close_dialog(dialog_instance: ft.AlertDialog): ...
    # REMOVED: Assignment to manage_presets_button.on_click

    # REMOVED: Ensure add_example_button is valid before assigning handler
    # if add_example_button: ...

    # --- Create Text Input Tab Elements & Handlers ---
    text_input_field = ft.TextField(
        label="在此输入文本 (或等待定时器发送)",  # Updated label
        multiline=True,
        min_lines=3,  # Increased min lines for better multiline visibility
        max_lines=5,  # Increased max lines
        # expand=True, # Keep removed
        border_color=ft.colors.OUTLINE,
        dense=True,
        on_submit=None,  # Assigned later
        on_change=None,  # Assigned later
        text_align=ft.TextAlign.LEFT,
    )
    text_input_progress = ft.ProgressRing(
        visible=False, width=16, height=16, stroke_width=2
    )
    submit_text_button = ft.ElevatedButton(
        "发送",
        icon=ft.icons.SEND,
        tooltip="处理并发送输入的文本 (Enter 键也可发送)",
        on_click=None,  # Assigned later
    )

    # --- Timer UI Handlers (Defined BEFORE use) ---
    async def timer_switch_change(e: ft.ControlEvent):
        """Handles changes to the timer enable switch."""
        app_state.is_timer_enabled = e.control.value
        logger.info(
            f"Text input timer {'enabled' if app_state.is_timer_enabled else 'disabled'}."
        )
        if app_state.is_timer_enabled:
            # Start timer immediately if enabled and text exists
            # Need to ensure _start_text_timer is defined or moved earlier too if not already
            await _start_text_timer(
                app_state,
                page,
                text_input_field,
                submit_text_button,
                text_input_progress,
                submit_text_handler,
            )
        else:
            # Cancel timer if disabled
            _cancel_text_timer(app_state)
        # Removed window_exists check - page.update() handles closed pages
        page.update()  # Update UI if needed

    async def timer_delay_change(e: ft.ControlEvent):
        """Handles changes to the timer delay input."""
        try:
            new_delay = float(e.control.value)
            if new_delay >= 0:  # Allow 0 to effectively disable timer via delay
                app_state.text_input_timer_delay = new_delay
                logger.info(
                    f"Text input timer delay set to: {app_state.text_input_timer_delay}s."
                )
                # Reset the timer with the new delay if it's currently running/should run
                await _start_text_timer(
                    app_state,
                    page,
                    text_input_field,
                    submit_text_button,
                    text_input_progress,
                    submit_text_handler,
                )
            else:
                logger.warning("Timer delay must be non-negative.")
                e.control.value = str(
                    app_state.text_input_timer_delay
                )  # Revert display
                gui_utils.show_error_banner(page, "定时器延迟必须是非负数。")
        except ValueError:
            logger.warning(f"Invalid timer delay input: {e.control.value}")
            e.control.value = str(app_state.text_input_timer_delay)  # Revert display
            gui_utils.show_error_banner(page, "无效的定时器延迟值。")
        # Removed window_exists check - page.update() handles closed pages
        page.update()  # Update UI

    # --- Timer UI Elements ---
    timer_switch = ft.Switch(
        label="启用定时发送",
        value=app_state.is_timer_enabled,
        on_change=timer_switch_change,  # Assigned here
        tooltip="启用后，停止输入指定时间后自动发送",
    )
    timer_delay_input = ft.TextField(
        label="延迟(秒)",
        value=str(app_state.text_input_timer_delay),
        width=80,  # Make input smaller
        dense=True,
        keyboard_type=ft.KeyboardType.NUMBER,
        on_change=timer_delay_change,  # Assigned here
        tooltip="停止输入后等待多少秒发送 (例如 5.0)",
    )

    async def submit_text_handler(e: ft.ControlEvent):
        """
        Handles clicks on the text input submit button or Enter key.
        Also called by the timer expiry function.
        """
        # Always cancel any pending timer on manual submit or timer fire
        _cancel_text_timer(app_state)

        input_text = text_input_field.value
        if not input_text or not input_text.strip():
            # Don't show banner on empty submit, just do nothing.
            # gui_utils.show_error_banner(page, "请输入要发送的文本。")
            logger.debug("Empty text submitted, doing nothing.")
            return

        logger.info(f"Text input submitted: '{input_text[:50]}...'")
        # Disable field and button during processing
        text_input_field.disabled = True
        submit_text_button.disabled = True
        text_input_progress.visible = True
        page.update()  # Show progress

        processed_text = input_text  # Default to original text
        try:
            # 1. Process with LLM (if enabled) - Create temporary instance
            llm_client_instance: Optional[LLMClient] = None
            if config.get("llm.enabled", False):
                logger.info("Text Input: LLM enabled, creating temporary LLMClient...")
                try:
                    llm_client_instance = (
                        LLMClient()
                    )  # Create new instance with current config
                    if not llm_client_instance.enabled:
                        logger.warning(
                            "Text Input: LLMClient created but is not enabled (e.g., missing API key). Skipping LLM."
                        )
                        llm_client_instance = None  # Treat as disabled
                    else:
                        logger.info("Text Input: Temporary LLMClient created.")
                except Exception as llm_create_err:
                    logger.error(
                        f"Text Input: Failed to create LLMClient: {llm_create_err}",
                        exc_info=True,
                    )
                    gui_utils.show_error_banner(
                        page, f"创建 LLM 客户端时出错: {llm_create_err}"
                    )
                    llm_client_instance = None  # Ensure None on error

            if llm_client_instance:  # Check if instance was created and is enabled
                logger.debug("Text Input: Processing text with temporary LLMClient...")
                llm_result = await llm_client_instance.process_text(input_text)
                if llm_result is not None:
                    processed_text = llm_result
                    logger.info(
                        f"Text Input: LLM processed text: '{processed_text[:50]}...'"
                    )
                else:
                    logger.warning(
                        "Text Input: LLM processing returned None, using original text."
                    )
                    # Optionally show a warning banner?
                    # gui_utils.show_banner(page, "LLM 处理失败，使用原始文本。", ...)
            else:
                logger.debug(
                    "Text Input: LLM processing skipped (disabled or client creation failed)."
                )

            # 2. Dispatch the result (original or processed)
            if app_state.output_dispatcher:
                logger.info(
                    f"Dispatching text input result: '{processed_text[:50]}...'"
                )  # Log before dispatch
                await app_state.output_dispatcher.dispatch(processed_text)
                # REMOVED: update_output_callback(...) call
                logger.info("Text input dispatched successfully.")
                text_input_field.value = ""  # Clear input on successful dispatch
                # DO NOT restart timer here. Timer restarts on text_input_change.
            else:
                logger.error(
                    "OutputDispatcher not available, cannot dispatch text input."
                )
                gui_utils.show_error_banner(page, "错误：无法分发文本。")

        except Exception as ex:
            error_msg = f"处理文本输入时出错: {ex}"
            logger.error(error_msg, exc_info=True)
            gui_utils.show_error_banner(page, error_msg)
        finally:
            # Ensure UI is reset regardless of success or error
            text_input_field.disabled = False
            submit_text_button.disabled = False
            text_input_progress.visible = False
            # Optionally clear field only on success? Currently clears before dispatch attempt.
            # text_input_field.value = "" # Clearing moved to after successful dispatch
            # Removed check for page.window_exists() as it's not a valid attribute
            # Flet handles updates to closed pages gracefully.
            page.update()

    async def text_input_change(e: ft.ControlEvent):
        """Handles changes in the text input field to reset the timer."""
        # Don't await here, just schedule the timer start/reset
        # Use asyncio.create_task to avoid blocking the UI thread if _start_text_timer takes time
        asyncio.create_task(
            _start_text_timer(
                app_state,
                page,
                text_input_field,
                submit_text_button,
                text_input_progress,
                submit_text_handler,
            )
        )
        # No page.update() needed here, typing updates the field itself.

    # Assign handlers now that they are defined
    text_input_field.on_submit = submit_text_handler  # Assign handler for Enter key
    text_input_field.on_change = (
        text_input_change  # Assign handler for typing (timer reset)
    )
    submit_text_button.on_click = submit_text_handler  # Clicking the button

    # --- Create Tab Layouts ---
    dashboard_tab_layout = create_dashboard_tab_content(
        dashboard_elements
    )  # Pass created elements dict

    config_tab_layout = create_config_tab_content(
        # REMOVED: save_button argument
        reload_button=reload_config_button,
        open_folder_button=open_config_folder_button,  # Pass the new button
        all_controls=all_config_controls,  # Pass controls dict
    )

    log_tab_layout = create_log_tab_content(log_elements)  # Create log tab layout

    # --- Text Input Tab Layout (Revised with Timer) ---
    text_input_tab_content = ft.Column(
        [
            ft.Divider(height=10, color=ft.colors.TRANSPARENT),  # Add spacer at the top
            # Removed the descriptive text, label is clearer now
            # ft.Text(
            #     "手动输入文本并发送...",
            #     size=12,
            #     text_align=ft.TextAlign.CENTER,
            # ),
            text_input_field,  # The multi-line text field
            ft.Row(  # Row for Send button and progress indicator
                [submit_text_button, text_input_progress],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=10,
            ),
            ft.Divider(height=10, color=ft.colors.TRANSPARENT),  # Spacer
            ft.Row(  # Row for Timer controls
                [
                    timer_switch,
                    timer_delay_input,
                ],
                alignment=ft.MainAxisAlignment.CENTER,  # Center timer controls
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=10,
            ),
            # --- REMOVED Few-Shot Display Section ---
            # ft.Divider(height=10),
            # ft.Text("当前预设 Few-Shot 示例:", ...),
            # ft.Container(content=text_input_few_shot_display, ...),
            ft.Divider(height=10),  # Keep a divider before the new info section
            # --- New Info Display Section (Placeholder) ---
            ft.Column(  # Column to hold the new info rows
                controls=[
                    _create_info_row(
                        ft.icons.TEXT_SNIPPET_OUTLINED,
                        text_input_info_elements["info_llm_label"],
                    ),
                    _create_info_row(
                        ft.icons.EDIT_NOTE_OUTLINED,
                        text_input_info_elements["info_preset_label"],
                    ),
                    _create_info_row(
                        ft.icons.SEND_AND_ARCHIVE_OUTLINED,
                        text_input_info_elements["info_vrc_label"],
                    ),
                    _create_info_row(
                        ft.icons.SAVE_ALT_OUTLINED,
                        text_input_info_elements["info_file_label"],
                    ),
                    _create_info_row(
                        ft.icons.FOLDER_OPEN_OUTLINED,
                        text_input_info_elements["info_config_path_label"],
                    ),
                ],
                spacing=5,
                alignment=ft.MainAxisAlignment.START,
                horizontal_alignment=ft.CrossAxisAlignment.START,
                # Add key for easy access later? Not strictly needed if elements dict is used.
                # key="text_input_info_column",
            ),
        ],
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,  # Keep outer alignment
        spacing=10,
        expand=True,  # Keep outer column expanding
    )

    # --- Create Tabs ---
    tabs_control = ft.Tabs(
        [
            ft.Tab(
                text=" 语音 ",  # Padded with spaces
                icon=ft.icons.MIC,  # Changed icon from DASHBOARD
                content=dashboard_tab_layout,  # Content remains the same layout
            ),
            ft.Tab(  # Moved Text Input tab up
                text="文本输入",
                icon=ft.icons.TEXT_FIELDS,
                content=text_input_tab_content,
            ),
            ft.Tab(  # Add new Preset Tab
                text=" 预设 ",  # Padded with spaces
                icon=ft.icons.EDIT_NOTE,  # Use preset icon
                content=preset_tab_layout,  # Use content created above
            ),
            ft.Tab(
                text=" 配置 ", icon=ft.icons.SETTINGS, content=config_tab_layout
            ),  # Config tab, Padded
            ft.Tab(
                text=" 日志 ",  # Log tab, Padded
                icon=ft.icons.LIST_ALT_ROUNDED,
                content=log_tab_layout,
            ),
        ],
        expand=True,  # Tabs control itself should expand within its container
    )

    # --- Container for Tabs with Dynamic Padding (Moved Here) ---
    tabs_container = ft.Container(
        expand=True,  # Make container fill the space
        content=tabs_control,  # Assign content immediately
        padding=ft.padding.symmetric(horizontal=50),  # Initial padding, will be updated
        alignment=ft.alignment.top_center,  # Center content if it doesn't expand fully
    )

    # --- Page Resize Handler (Moved Here) ---
    async def on_page_resize(e=None):  # Allow calling without event
        if page.width and tabs_container:
            # Reduce padding percentage from 0.1 (10%) to 0.05 (5%) and min padding from 20 to 10
            horizontal_padding = max(10, page.width * 0.05)
            tabs_container.padding = ft.padding.symmetric(horizontal=horizontal_padding)
            logger.debug(
                f"Page resized to {page.width}px width. Setting tabs container padding to {horizontal_padding}px."
            )
            try:
                if page.controls:  # Check if page is still valid
                    tabs_container.update()
            except Exception as resize_update_err:
                logger.warning(
                    f"Error updating tabs container on resize: {resize_update_err}"
                )
        elif not page.width:
            logger.debug("Page width not available during resize event.")

    page.on_resize = on_page_resize  # Assign the handler

    # --- Add Container (with Tabs inside) to Page ---
    page.add(tabs_container)  # Add the container to the page

    # --- Periodic Log Queue Check ---
    # Schedule a task to periodically check the log queue and update the UI
    async def periodic_log_update():
        while True:
            try:
                # Attempt to update the log display
                log_update_callback()  # Call the partial function
            except Exception as log_update_err:
                # If any error occurs during update (e.g., page closed), log it and stop the loop.
                logger.warning(
                    f"Error during periodic log update (likely page closed), stopping task: {log_update_err}"
                )
                break  # Exit the loop
            # If update succeeds, wait and continue
            await asyncio.sleep(0.5)  # Check every 500ms

    page.run_task(periodic_log_update)

    # --- Initial Population of LLM Active Preset Label (Preset Tab) ---
    logger.debug("Initial population of LLM active preset label (Preset Tab).")
    try:
        # Get active preset name from initial config data
        initial_active_preset = initial_config_data.get("llm", {}).get(
            "active_preset_name", "Default"
        )
        logger.info(f"Initial active preset from config: '{initial_active_preset}'")

        # Call the partial function directly to update the Preset Tab label
        # The partial already has page, controls, and the label control bound.
        # We just need to provide the preset name.
        # Run it in the background as it involves UI updates.
        async def initial_preset_label_update_task():
            try:
                # The partial itself is not async, but the underlying function might update UI
                # Let's call it directly. If it needs async, wrap it.
                # update_preset_tab_label_partial is bound to gui_config.update_llm_config_ui
                # which updates UI controls. It should be called within the Flet event loop context.
                # Calling it directly here might be okay if Flet handles it, or use page.run_task.
                update_preset_tab_label_partial(initial_active_preset)
                logger.info(
                    f"Initial Preset Tab label updated for preset '{initial_active_preset}'."
                )
            except Exception as label_update_err:
                logger.error(
                    f"Error during initial preset label update task: {label_update_err}",
                    exc_info=True,
                )

        # Schedule the update task
        page.run_task(initial_preset_label_update_task)
        # Also set the initial value for the dropdown in the Preset Tab (this part remains)
        preset_select_dd_ctrl = preset_tab_elements.get("preset_select_dd")
        if preset_select_dd_ctrl and isinstance(preset_select_dd_ctrl, ft.Dropdown):
            # Ensure the active preset exists in the options before setting
            if any(
                opt.key == initial_active_preset
                for opt in preset_select_dd_ctrl.options
            ):
                preset_select_dd_ctrl.value = initial_active_preset
                logger.debug(
                    f"Set initial value of Preset Tab dropdown to '{initial_active_preset}'."
                )
            else:
                logger.warning(
                    f"Initial active preset '{initial_active_preset}' not found in Preset Tab dropdown options. Leaving dropdown unselected."
                )
                preset_select_dd_ctrl.value = None  # Explicitly set to None
        else:
            logger.warning(
                "Could not find preset dropdown control in Preset Tab elements to set initial value."
            )

    except Exception as llm_init_err:
        logger.error(
            f"Error during initial LLM config/preset UI population: {llm_init_err}",
            exc_info=True,
        )
        gui_utils.show_error_banner(page, "初始化 LLM/预设 UI 时出错")
    try:
        # Get active preset name from initial config data
        initial_active_preset = initial_config_data.get("llm", {}).get(
            "active_preset_name", "Default"
        )
        logger.info(f"Initial active preset from config: '{initial_active_preset}'")

        # Call the partial function directly to update the Preset Tab label
        # The partial already has page, controls, and the label control bound.
        # We just need to provide the preset name.
        # Run it in the background as it involves UI updates.
        async def initial_preset_label_update_task():
            try:
                # The partial itself is not async, but the underlying function might update UI
                # Let's call it directly. If it needs async, wrap it.
                # update_preset_tab_label_partial is bound to gui_config.update_llm_config_ui
                # which updates UI controls. It should be called within the Flet event loop context.
                # Calling it directly here might be okay if Flet handles it, or use page.run_task.
                update_preset_tab_label_partial(initial_active_preset)
                logger.info(
                    f"Initial Preset Tab label updated for preset '{initial_active_preset}'."
                )
            except Exception as label_update_err:
                logger.error(
                    f"Error during initial preset label update task: {label_update_err}",
                    exc_info=True,
                )

        # Schedule the update task
        page.run_task(initial_preset_label_update_task)
        # Also set the initial value for the dropdown in the Preset Tab (this part remains)
        preset_select_dd_ctrl = preset_tab_elements.get("preset_select_dd")
        if preset_select_dd_ctrl and isinstance(preset_select_dd_ctrl, ft.Dropdown):
            # Ensure the active preset exists in the options before setting
            if any(
                opt.key == initial_active_preset
                for opt in preset_select_dd_ctrl.options
            ):
                preset_select_dd_ctrl.value = initial_active_preset
                logger.debug(
                    f"Set initial value of Preset Tab dropdown to '{initial_active_preset}'."
                )
            else:
                logger.warning(
                    f"Initial active preset '{initial_active_preset}' not found in Preset Tab dropdown options. Leaving dropdown unselected."
                )
                preset_select_dd_ctrl.value = None  # Explicitly set to None
        else:
            logger.warning(
                "Could not find preset dropdown control in Preset Tab elements to set initial value."
            )

    except Exception as llm_init_err:
        logger.error(
            f"Error during initial LLM config/preset UI population: {llm_init_err}",
            exc_info=True,
        )
        gui_utils.show_error_banner(page, "初始化 LLM/预设 UI 时出错")

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
        logger.debug("Scheduled initial dashboard info update.")
    except Exception as e:
        logger.error(f"Error scheduling initial dashboard update: {e}", exc_info=True)

    # --- Initial Text Input Info Population ---
    logger.debug("Initial population of text input info display.")
    try:
        # Call the partial for the text input tab info
        update_text_input_info_partial()
        logger.debug("Scheduled initial text input info update.")
    except Exception as e:
        logger.error(
            f"Error scheduling initial text input info update: {e}", exc_info=True
        )

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
            if (
                app_state.output_dispatcher
            ):  # Check if dispatcher exists before assigning None
                app_state.output_dispatcher.vrc_client_instance = None

        # --- Initialize OutputDispatcher (now that VRCClient is potentially ready) ---
        try:
            logger.info("Initializing OutputDispatcher asynchronously...")
            app_state.output_dispatcher = OutputDispatcher(
                vrc_client_instance=app_state.vrc_client,  # Pass the actual client instance (or None)
                # REMOVED: gui_output_callback=update_output_callback,
            )
            logger.info("OutputDispatcher initialized.")
        except Exception as disp_err:
            logger.critical(
                f"CRITICAL ERROR initializing OutputDispatcher: {disp_err}",
                exc_info=True,
            )
            gui_utils.show_error_banner(
                page, f"OutputDispatcher 初始化失败: {disp_err}"
            )
            # Cannot proceed without dispatcher for audio/text input
            return  # Stop further async initialization

            # --- Initialize AudioManager (now that Dispatcher is ready) ---
            # AudioManager is now created synchronously in _start_recording_internal
            # No need to initialize it here anymore.
            # try:
            #     logger.info("Initializing AudioManager asynchronously...")
            #     app_state.audio_manager = AudioManager(
            #         llm_client=app_state.llm_client, # LLM client is also created in start
            #         output_dispatcher=app_state.output_dispatcher,
            #         status_callback=update_status_callback,
            #         audio_level_callback=update_audio_level_callback,
            #     )
            #     logger.info("AudioManager initialized.")
            # except Exception as am_err:
            #     logger.critical(f"CRITICAL ERROR initializing AudioManager: {am_err}", exc_info=True)
            #     gui_utils.show_error_banner(page, f"AudioManager 初始化失败: {am_err}")
            # Audio input will not work, but text input might still function
            # ) # Comment out or remove the parenthesis
            logger.info("AudioManager initialized.")
        except Exception as am_err:
            logger.critical(
                f"CRITICAL ERROR initializing AudioManager: {am_err}", exc_info=True
            )
            gui_utils.show_error_banner(page, f"AudioManager 初始化失败: {am_err}")
            # Audio input will not work, but text input might still function
        # --- End AudioManager Initialization Removal ---

        logger.info("Async component initialization finished (VRCClient, OutputDispatcher).")

    # Schedule the async initialization task to run (for VRCClient, OutputDispatcher)
    page.run_task(initialize_async_components)

    # --- Apply Initial Padding ---
    # Call resize handler once manually after layout is added to set initial padding
    page.run_task(on_page_resize)

    # --- Final Page Update ---
    page.update()


# 注意：此文件不再包含 if __name__ == "__main__": ft.app(...)
# 这将移至 main.py
