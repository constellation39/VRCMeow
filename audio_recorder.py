import threading
import threading
import queue  # Use standard queue for thread-safe communication
from typing import Optional, TYPE_CHECKING, Union, Callable, Any, List, Dict

import numpy as np
import sounddevice as sd

# 直接从 config 模块导入 config 实例
from config import config

# 获取该模块的 logger 实例
from logger_config import get_logger

# Import Dashscope base recognizer types for type hinting
from dashscope.audio.asr import TranslationRecognizerRealtime, Recognition
# Import specific error type
from dashscope.common.error import InvalidParameter

# Local STT implementation imports
from stt_gummy import create_gummy_recognizer
from stt_paraformer import create_paraformer_recognizer


logger = get_logger(__name__)


# --- Audio Device Helper ---
def get_input_devices() -> List[Dict[str, Any]]:
    """Returns a list of available audio input devices."""
    devices = []
    try:
        device_list = sd.query_devices()
        hostapis = sd.query_hostapis()
        default_input_idx = sd.default.device[0]  # Get default input device index

        for i, device in enumerate(device_list):
            # Check if it's an input device (has input channels)
            if device.get("max_input_channels", 0) > 0:
                # Try to get host API info
                try:
                    hostapi_info = hostapis[device.get("hostapi", 0)]
                    hostapi_name = hostapi_info.get("name", "Unknown API")
                except IndexError:
                    hostapi_name = (
                        "Unknown API"  # Handle case where hostapi index is invalid
                    )

                # --- Filter by Host API (Keep only MME) ---
                if hostapi_name == "MME":
                    # Use original device name, maybe add (MME) for clarity?
                    # For now, just use the name sounddevice provides.
                    device_name = device.get("name", "Unknown Device")
                    # Add (MME) suffix for clarity in the dropdown
                    display_name = f"{device_name} (MME)"

                    is_default = i == default_input_idx
                    devices.append(
                        {
                            "id": i,  # Store index for potential use with sounddevice
                            "name": display_name,  # User-friendly name with MME suffix
                            "is_default": is_default,
                            # Store the original name without suffix for potential matching in AudioManager
                            "raw_name": device.get("name"),
                        }
                    )
                else:
                    # Log skipped devices for debugging if needed
                    # logger.debug(f"Skipping device '{device.get('name')}' with host API '{hostapi_name}'")
                    pass  # Skip devices that are not MME

        logger.debug(f"Found MME input devices: {devices}")
    except Exception as e:
        logger.error(f"Error querying audio devices: {e}", exc_info=True)
        # Return a fallback device entry if query fails
        devices.append(
            {
                "id": -1,
                "name": "Error querying devices",
                "is_default": False,
                "raw_name": None,
            }
        )
    return devices


# --- Component Imports for Type Hinting (Keep these) ---
# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    try:
        from llm_client import LLMClient
    except ImportError:
        LLMClient = None  # type: ignore
    try:
        from output_dispatcher import OutputDispatcher
    except ImportError:
        OutputDispatcher = None  # type: ignore
    try:
        from osc_client import VRCClient
    except ImportError:
        VRCClient = None  # type: ignore


# Actual imports needed at runtime (handled elsewhere, these are just for hints)


# --- AudioManager Class ---
class AudioManager:
    """
    Manages audio input, STT processing, and communication with the GUI.
    Runs audio stream and STT task in separate threads.
    """

    def __init__(
        self,
        llm_client: Optional["LLMClient"],
        output_dispatcher: "OutputDispatcher",
        status_callback: Optional[Callable[[str, Optional[bool], bool], None]] = None,
        audio_level_callback: Optional[
            Callable[[float], None]
        ] = None,  # Add audio level callback
    ):
        self.llm_client = llm_client
        self.output_dispatcher = output_dispatcher
        self.audio_level_callback = audio_level_callback  # Store audio level callback
        # Store the raw status callback
        self._raw_status_callback = status_callback
        # Wrapper for the callback to include state (initialized later if needed)
        self.status_callback: Optional[Callable[[str, Optional[bool], bool], None]] = (
            status_callback  # Assign callback directly
        )

        self._audio_queue = queue.Queue()  # Thread-safe queue
        self._stop_event = (
            threading.Event()
        )  # Use threading.Event for cross-thread signaling
        self._audio_thread: Optional[threading.Thread] = None
        self._stt_thread: Optional[threading.Thread] = None
        # Remove asyncio related attributes
        # self._stt_task: Optional[asyncio.Task] = None
        # self._stt_loop: Optional[asyncio.AbstractEventLoop] = None

        # Load necessary config values during initialization
        self.sample_rate = config.get("audio.sample_rate")  # Get initial value
        self.channels = config.get("audio.channels", 1)  # Use get for robustness
        self.dtype = config.get("audio.dtype", "int16")
        self.debug_echo_mode = config.get("audio.debug_echo_mode", False)
        self.device = config.get("audio.device", "Default")  # Load selected device

        # --- Get STT Model and Sample Rate from Config ---
        selected_model_name = config.get("dashscope.stt.selected_model")
        stt_models_config = config.get("dashscope.stt.models", {})
        model_info = stt_models_config.get(selected_model_name)

        if not selected_model_name or not model_info:
            # Fallback or error if selected model is invalid or not found
            fallback_model = "gummy-realtime-v1" # Or choose another default
            logger.error(
                f"Selected STT model '{selected_model_name}' not found or invalid in config. Falling back to '{fallback_model}'."
            )
            selected_model_name = fallback_model
            model_info = stt_models_config.get(selected_model_name, {}) # Try getting fallback info

        self.stt_model = selected_model_name # Store the final selected model name
        self.sample_rate = model_info.get("sample_rate")

        # Validate if sample rate was found for the selected (or fallback) model
        if self.sample_rate is None:
            default_sr = 16000 # Define a hardcoded default if config is broken
            logger.critical(
                f"Sample rate not defined for STT model '{self.stt_model}' in config! Using default: {default_sr} Hz. Audio stream might fail!"
            )
            self.sample_rate = default_sr
        else:
            logger.info(f"Using STT Model: '{self.stt_model}' with Sample Rate: {self.sample_rate} Hz")

    # REMOVED: _determine_sample_rate method and its body. Sample rate is now determined from config.

    # Remove the old _update_status signature that didn't take kwargs # This comment is still relevant
    # def _update_status(self, message: str): ... # REMOVED

    # Keep the correctly defined overload signature
    def _update_status(
        self,
        message: str,
        *,
        is_running: Optional[bool] = None,
        is_processing: bool = False,
    ):
        """Helper to call the status callback with running and processing state."""
        logger.info(
            f"AudioManager Status: {message} (Running: {is_running}, Processing: {is_processing})"
        )  # Log status internally
        # Use the wrapped status_callback
        if self.status_callback:
            try:
                # Pass the new arguments to the callback
                self.status_callback(
                    message, is_running=is_running, is_processing=is_processing
                )
            except Exception as e:
                logger.error(f"Error calling status callback: {e}", exc_info=True)

    # Overload signature for clarity and type hinting (optional but good practice)
    def _update_status(
        self,
        message: str,
        *,
        is_running: Optional[bool] = None,
        is_processing: bool = False,
    ):
        """Helper to call the status callback with running and processing state."""
        logger.info(
            f"AudioManager Status: {message} (Running: {is_running}, Processing: {is_processing})"
        )  # Log status internally
        if self.status_callback:
            try:
                # Pass the new arguments to the callback
                self.status_callback(
                    message, is_running=is_running, is_processing=is_processing
                )
            except Exception as e:
                logger.error(f"Error calling status callback: {e}", exc_info=True)

    # --- STT Processing Logic (Now synchronous, runs in its own thread) ---
    # Rename and remove async def
    def _run_stt_processor(self):
        """Target function for the STT processing thread. Handles recognizer lifecycle and data feeding."""
        # No longer an async task, but the main logic for the STT thread.
        # Model name (self.stt_model) and sample rate (self.sample_rate) are now set in __init__

        logger.info("STT Processor Thread Started.")
        if self.status_callback:
            self._update_status(
                "STT 线程启动中...", is_processing=True
            )  # Indicate processing during startup
        logger.info(
            f"STT processing loop (Dashscope, model: {self.stt_model}) starting in thread {threading.current_thread().ident}..."
        )
        recognizer: Optional[Union[TranslationRecognizerRealtime, Recognition]] = None
        # No event loop needed here
        #     from output_dispatcher import OutputDispatcher # Not typically needed here
        # except ImportError:
        #     OutputDispatcher = None

        #     from output_dispatcher import OutputDispatcher # Not typically needed here
        # except ImportError:
        #     OutputDispatcher = None
        engine_type = "Unknown"

        # --- Reconnect Parameters ---
        max_retries = 5
        initial_retry_delay = 1.0  # Initial retry delay (seconds)
        max_retry_delay = 30.0  # 最大重试延迟 (秒)
        retry_count = 0
        current_delay = initial_retry_delay

        # API Key check happens in main.py before AudioManager is created

        # --- Outer Reconnect Loop ---
        while not self._stop_event.is_set():
            recognizer = None  # Reset recognizer before each connection attempt
            try:
                # --- Check config and model compatibility (re-check each loop) ---
                # Use instance variables set in __init__
                current_model = self.stt_model # Get model name from instance var
                stt_models_config = config.get("dashscope.stt.models", {}) # Reload models dict in case config changed
                model_info = stt_models_config.get(current_model, {}) # Get current model's info
                model_type = model_info.get("type") # 获取模型类型

                target_language = config.get("dashscope.stt.translation_target_language")
                # Determine translation support based on config
                model_supports_translation = model_info.get("supports_translation", False)
                enable_translation = bool(target_language) and model_supports_translation

                # --- Validate Model Type ---
                if not model_type or model_type not in ["gummy", "paraformer"]:
                    error_msg = f"配置中为模型 '{current_model}' 指定的类型 '{model_type}' 无效或缺失。请在 config.yaml 中设置 'type' 为 'gummy' 或 'paraformer'。"
                    logger.error(error_msg)
                    if self.status_callback:
                        self._update_status(
                            f"错误: {error_msg}", is_running=False, is_processing=False
                        )
                    self._stop_event.set()  # Signal stop
                    break  # Exit reconnect loop

                # Check if translation is requested but not supported by the selected model
                if bool(target_language) and not model_supports_translation:
                     logger.warning(
                        f"配置文件中请求了翻译 (目标语言: {target_language}), "
                        f"但所选模型 '{current_model}' 不支持翻译。将禁用翻译。"
                    )
                     enable_translation = False # Ensure translation is off

                # --- Select and create recognizer based on type from config ---
                # Indicate processing (connecting) state
                if self.status_callback:
                    self._update_status(
                        f"连接 STT (模型: {current_model}, 类型: {model_type}, 尝试 {retry_count + 1}/{max_retries})...",
                        is_processing=True,
                    )
                logger.info(
                    f"Attempting to connect STT service (Model: {current_model}, Type: {model_type}, Attempt {retry_count + 1}/{max_retries})..."
                )

                # Use model_type read from config
                if model_type == "gummy":
                    if create_gummy_recognizer is None:
                        raise RuntimeError("Gummy STT module (stt_gummy.py) failed to load.")
                    engine_type = "Gummy" # Keep track of engine type for logging
                    logger.info(f"Creating Gummy Recognizer (translation enabled: {enable_translation})...")
                    recognizer = create_gummy_recognizer(
                        sample_rate=self.sample_rate,
                        llm_client=self.llm_client,
                        output_dispatcher=self.output_dispatcher,
                        # target_language is read from config inside create_gummy_recognizer
                    )
                elif model_type == "paraformer":
                    if create_paraformer_recognizer is None:
                        raise RuntimeError("Paraformer STT module (stt_paraformer.py) failed to load.")
                    engine_type = "Paraformer" # Keep track of engine type
                    logger.info("Creating Paraformer Recognizer...")
                    recognizer = create_paraformer_recognizer(
                        sample_rate=self.sample_rate,
                        llm_client=self.llm_client,
                        output_dispatcher=self.output_dispatcher,
                    )
                else:
                    # This case should be caught by the validation above, but handle defensively
                    raise RuntimeError(f"未知的模型类型 '{model_type}' 在配置中为模型 '{current_model}' 指定。")

                # Check if recognizer creation succeeded (it should have if no exception)
                if not recognizer:
                     # This path might be less likely now, but keep for robustness
                     raise RuntimeError(f"未能为模型 '{current_model}' (类型: {model_type}) 创建识别器实例。")

                # --- Start Recognizer ---
                recognizer.start()
                logger.info(
                    f"Dashscope {engine_type} Recognizer connected and started."
                )
                # Update status to indicate running (connected), not processing anymore until data flows
                if self.status_callback:
                    self._update_status(
                        f"STT 已连接 (引擎: {engine_type})",
                        is_running=True,
                        is_processing=False,
                    )
                retry_count = 0  # Reset retries on success
                current_delay = initial_retry_delay  # Reset delay

                # --- Inner Audio Processing Loop ---
                while not self._stop_event.is_set():
                    try:
                        # Get audio data from the thread-safe queue
                        # Use timeout to allow checking _stop_event periodically
                        try:
                            audio_data = self._audio_queue.get(timeout=0.1)
                        except queue.Empty:
                            # Check stop event if queue is empty
                            if self._stop_event.is_set():
                                break
                            # No need for double check
                            continue  # Continue waiting for data

                        # DEBUG: Log that we got data and are sending it
                        # logger.debug(f"STT Processor: Got audio data block, size: {len(audio_data)}")

                        # Send audio frame (can raise errors on connection issues)
                        recognizer.send_audio_frame(
                            audio_data.tobytes()
                        )  # audio_data is np.ndarray here

                        # DEBUG: Log successful send
                        # logger.debug("STT Processor: Sent audio frame successfully.")

                        self._audio_queue.task_done()  # Mark task as done for the queue
                    except Exception as send_error:
                        # 捕获发送音频帧时的错误，假设是连接问题
                        logger.error(
                            f"Error sending audio frame to STT service: {send_error}",
                            exc_info=True,
                        )
                        # Indicate error, but potentially still 'running' while trying to reconnect
                        if self.status_callback:
                            self._update_status(
                                f"STT 发送错误: {send_error}",
                                is_running=True,
                                is_processing=True,
                            )  # Show processing for reconnect attempt
                        logger.info("假设连接丢失，尝试重新连接...")
                        try:
                            # Attempt to stop the failed recognizer
                            recognizer.stop()
                            logger.info(
                                f"Stopped failed Dashscope {engine_type} Recognizer instance."
                            )
                        except InvalidParameter as ip_err:
                            # If the error is specifically that it's already stopped, just warn.
                            if str(ip_err) == "Speech recognition has stopped.":
                                logger.warning(f"Attempted to stop an already stopped recognizer: {ip_err}")
                            else:
                                # Log other InvalidParameter errors as errors
                                logger.error(f"Error stopping failed recognizer (InvalidParameter): {ip_err}", exc_info=True)
                        except Exception as stop_err:
                            # Catch any other unexpected errors during stop
                            logger.error(
                                f"Unexpected error stopping failed recognizer: {stop_err}",
                                exc_info=True,
                            )
                        recognizer = None  # Clear reference
                        break  # Break inner loop to trigger outer reconnect loop

                # If inner loop exited, check stop event before reconnecting
                if self._stop_event.is_set():
                    logger.info(
                        "Stop signal received during STT processing, exiting outer loop."
                    )
                    break  # Exit outer loop

            except Exception as connect_error:
                # --- Handle connection or startup errors ---
                logger.error(
                    f"Error connecting or starting Dashscope {engine_type} Recognizer: {connect_error}",
                    exc_info=True,
                )
                # Indicate connection error, not running, but processing (retrying)
                if self.status_callback:
                    self._update_status(
                        f"STT 连接错误: {connect_error}",
                        is_running=False,
                        is_processing=True,
                    )
                if recognizer:  # If recognizer was created but start failed
                    try:
                        recognizer.stop()
                    except Exception as stop_err:
                        logger.error(
                            f"Error stopping failed recognizer on startup: {stop_err}",
                            exc_info=True,
                        )
                    recognizer = None

                retry_count += 1
                if retry_count >= max_retries:
                    # Note: error_msg was defined within the IF block in the previous version,
                    # but here it seems to be referenced before definition if the loop gets here.
                    # Define error message here to ensure it's always available
                    final_error_msg = f"STT 在 {max_retries} 次重试后连接失败 (模型: '{current_model}')。正在停止 STT 处理。" # Use current_model
                    logger.critical(final_error_msg)
                    # Indicate final error state: not running, not processing
                    if self.status_callback:
                        self._update_status(
                            f"错误: {final_error_msg}",
                            is_running=False,
                            is_processing=False,
                        )
                    self._stop_event.set()  # Signal stop
                    break  # Exit reconnect loop

                wait_msg = f"将在 {current_delay:.1f} 秒后重试 STT 连接..."
                logger.info(wait_msg)
                # Indicate waiting state (not running, still processing/retrying)
                if self.status_callback:
                    self._update_status(wait_msg, is_running=False, is_processing=True)
                # Use threading Event's wait method, which is blocking but interruptible
                stopped_during_wait = self._stop_event.wait(timeout=current_delay)
                if stopped_during_wait:
                    logger.info(
                        "Stop signal received while waiting to retry STT connection."
                    )
                    break  # Exit reconnect loop

                # If wait timed out, continue with retry
                current_delay = min(
                    current_delay * 2, max_retry_delay
                )  # Exponential backoff

        # --- Final Cleanup after loop exit ---
        logger.info(f"STT processing loop (Dashscope {engine_type}) stopping...")
        # Indicate stopping state (was running, now processing stop)
        if self.status_callback:
            self._update_status(
                f"STT 任务停止中 (引擎: {engine_type})",
                is_running=True,
                is_processing=True,
            )
        if recognizer:  # Check if a recognizer was active when the loop exited
            try:
                logger.info(
                    f"Stopping final Dashscope {engine_type} Recognizer instance..."
                )
                recognizer.stop()  # This might block briefly
                logger.info(
                    f"Final Dashscope {engine_type} Recognizer instance stopped."
                )
                # Don't update status here yet, wait for the whole stop process
                # self._update_status("STT Recognizer Stopped.") # Removed intermediate update
            except Exception as e:
                logger.error(
                    f"Error stopping final recognizer instance: {e}", exc_info=True
                )
                # Indicate error during stop, final state will be 'Stopped' or 'Stopped (with issues)' later
                if self.status_callback:
                    self._update_status(
                        f"停止 STT 时出错: {e}", is_running=False, is_processing=False
                    )
        else:
            logger.info("没有活动的 Recognizer 实例需要停止。")

        # Drain the queue (use the thread-safe queue - this matches the user's current code context)
        logger.info("Draining remaining audio queue...")
        drained_count = 0
        # Use the correct queue object and exception type based on user's current code context
        while True:
            try:
                # Assuming self._audio_queue based on the AudioManager context
                self._audio_queue.get_nowait()
                self._audio_queue.task_done()
                drained_count += 1
            except queue.Empty:  # Use standard queue.Empty
                break
        if drained_count > 0:
            logger.info(f"清空了 {drained_count} 个音频队列项目。")

        logger.info(f"STT processing loop (Dashscope {engine_type}) fully stopped.")
        # Final status update for STT thread stop is handled by the main stop() method
        # self._update_status("STT Thread Stopped.") # Removed

        # --- Audio Stream Handling --- # Adjusted comment

        # This method belongs inside the AudioManager class based on the refactoring

    def _audio_callback(  # Correct indentation for method
        self,
        indata: np.ndarray,
        outdata: np.ndarray,  # Add outdata back for Stream
        frames: int,
        time: Any,
        status: sd.CallbackFlags,
    ):
        """Synchronous callback for sounddevice stream (handles input and output)."""
        if status:
            logger.warning(f"Audio callback status: {status}")
            # Potentially update status for critical flags?
            # self._update_status(f"Audio Status: {status}")

        # Echo mode (using instance variable) - Restore functionality
        # DEBUG: 在回调中记录 debug_echo_mode 的状态
        # logger.debug(f"音频回调: debug_echo_mode = {self.debug_echo_mode}") # 取消注释以进行调试
        if self.debug_echo_mode:
            # Copy input directly to output for echo effect
            # logger.debug(f"音频回调: 正在回显 {indata.shape} 到 {outdata.shape}") # 可选：记录形状
            outdata[:] = indata
        else:
            # Fill output buffer with silence when echo is off
            outdata.fill(0)

        # Put audio data into the thread-safe queue
        # This callback runs in the sounddevice thread.
        # Use put_nowait for thread-safe queue access.
        try:
            # Ensure correct dtype if necessary (sounddevice usually handles this)
            if indata.dtype != np.dtype(self.dtype):
                logger.warning(
                    f"Incoming audio dtype {indata.dtype} != expected {self.dtype}. Converting."
                )
                indata = indata.astype(self.dtype)
            # Put a copy into the queue
            self._audio_queue.put_nowait(indata.copy())

            # --- Calculate and send audio level ---
            if self.audio_level_callback:
                try:
                    # Calculate RMS (Root Mean Square) - requires float conversion
                    # Ensure indata is float32 for calculation
                    float_data = indata.astype(np.float32)
                    rms = np.sqrt(np.mean(float_data**2))

                    # Normalize RMS to 0.0 - 1.0 range
                    # This requires tuning based on expected input levels and dtype.
                    # For int16, max amplitude is 32767. A value around 3000-5000 might represent loud speech.
                    # Let's use a simple scaling factor, clamping at 1.0.
                    # Adjust the divisor (e.g., 5000) based on testing. Lower value = more sensitive meter.
                    scaling_factor = 5000.0
                    normalized_level = min(1.0, rms / scaling_factor)

                    # Call the callback with the normalized level
                    self.audio_level_callback(normalized_level)
                except Exception as level_err:
                    # Log error but don't stop audio processing for level calculation failure
                    logger.error(
                        f"Error calculating/sending audio level: {level_err}",
                        exc_info=False,
                    )  # Avoid excessive logging

        except queue.Full:
            logger.warning("Audio queue is full. Dropping audio frame.")
            # Consider adding a status update here if it happens frequently
            # self._update_status("Warning: Audio queue full, dropping data")
            pass  # Continue processing

    def _run_audio_stream(self):
        """Target function for the audio processing thread."""
        stream_device_index: Optional[int] = None  # Use index for reliability
        try:
            # --- Determine Device Index to Use ---
            configured_device_name = self.device  # Name from config (e.g., "Mic (MME)")
            if configured_device_name and configured_device_name != "Default":
                logger.info(
                    f"Attempting to find index for configured device: '{configured_device_name}'"
                )
                found_device = False
                # Use the same logic as get_input_devices to find the matching index
                available_devices = (
                    get_input_devices()
                )  # Get list of dicts {id, name, is_default, raw_name}
                for device_info in available_devices:
                    if device_info.get("name") == configured_device_name:
                        stream_device_index = device_info.get("id")
                        if stream_device_index is not None and stream_device_index >= 0:
                            logger.info(
                                f"Found matching device index: {stream_device_index} for '{configured_device_name}'"
                            )
                            found_device = True
                            break  # Stop searching once found
                        else:
                            logger.warning(
                                f"Found matching device '{configured_device_name}' but its index '{stream_device_index}' is invalid."
                            )

                if not found_device:
                    logger.warning(
                        f"Configured audio device '{configured_device_name}' not found among available MME devices. Falling back to default."
                    )
                    stream_device_index = None  # Fallback to default explicitly
            else:
                logger.info("Using default audio input device (index: None).")
                stream_device_index = None  # Use default

            # --- Log Device Info ---
            try:
                # Query the device that will actually be used (default if index is None)
                actual_input_device_info = sd.query_devices(
                    device=stream_device_index, kind="input"
                )
                input_name = (
                    actual_input_device_info.get("name", "Unknown")
                    if actual_input_device_info
                    else "Unknown (Default)"
                )
                logger.info(
                    f"Using Input Device: {input_name} (Index: {stream_device_index})"
                )

                # Log default output device for echo mode reference
                default_output_info = sd.query_devices(kind="output")
                output_name = (
                    default_output_info.get("name", "Unknown")
                    if default_output_info
                    else "Unknown (Default)"
                )
                logger.info(f"Default Output Device (for echo): {output_name}")

            except Exception as dev_err:
                logger.warning(f"Could not query specific audio devices: {dev_err}")

            # --- Log Configuration ---
            logger.info("Audio Stream Thread Configuration:")
            # Use the input_name determined earlier, which reflects the actual device used (including default)
            logger.info(f"  Device: {input_name} (Index: {stream_device_index})")
            logger.info(f"  Sample Rate: {self.sample_rate} Hz (from model '{self.stt_model}')") # Clarify origin
            logger.info(f"  Channels: {self.channels}")
            logger.info(f"  Dtype: {self.dtype}")
            logger.info(f"  Debug Echo: {self.debug_echo_mode}")
            # Indicate processing during startup
            if self.status_callback:
                self._update_status("音频流启动中...", is_processing=True)

            # Use sounddevice Stream context manager
            with sd.Stream(
                device=(
                    stream_device_index,
                    None,
                ),  # Specify input device index, default output
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                callback=self._audio_callback,
                blocksize=int(
                    self.sample_rate * 0.1
                ),  # Optional: Process in 100ms chunks
            ) as stream:
                actual_rate = getattr(
                    stream, "samplerate", "N/A"
                )  # Get actual rate if available
                logger.info(
                    f"Audio stream active. Actual Sample Rate: {actual_rate} Hz."
                )
                # Indicate running state, not processing (unless STT is connecting etc.)
                # The overall 'running' state depends on STT too. Set here tentatively.
                if self.status_callback:
                    self._update_status(
                        "音频流运行中", is_running=True, is_processing=False
                    )
                # Keep the stream running until stop_event is set
                self._stop_event.wait()  # Block here until stop() is called

        except sd.PortAudioError as e:
            error_msg = f"PortAudio Error: {e}. Check devices and permissions."
            logger.error(error_msg, exc_info=True)
            logger.error("Ensure microphone/speakers are connected and not in use.")
            logger.error(
                f"Check support for {self.sample_rate} Hz (required by model '{self.stt_model}'), {self.channels} channels, {self.dtype}."
            )
            # Indicate fatal error state
            if self.status_callback:
                self._update_status(
                    f"错误: {error_msg}", is_running=False, is_processing=False
                )
            self._stop_event.set()  # Signal other threads to stop
        except ValueError as e:
            error_msg = f"Audio Parameter Error: {e}. Check config."
            logger.error(error_msg, exc_info=True)
            logger.error(
                f"Verify sample rate ({self.sample_rate} for model '{self.stt_model}'), channels ({self.channels}), dtype ({self.dtype})."
            )
            # Indicate fatal error state
            if self.status_callback:
                self._update_status(
                    f"错误: {error_msg}", is_running=False, is_processing=False
                )
            self._stop_event.set()
        except Exception as e:
            error_msg = f"Unknown error in audio stream thread: {e}"
            logger.error(error_msg, exc_info=True)
            # Indicate fatal error state
            if self.status_callback:
                self._update_status(
                    f"错误: {error_msg}", is_running=False, is_processing=False
                )
            self._stop_event.set()  # Signal stop on unexpected errors
        finally:
            logger.info("Audio stream thread finishing.")
            # Ensure stop event is set if this thread exits unexpectedly
            self._stop_event.set()
            # Status update upon stopping is handled in the stop() method generally

    # Removed _run_stt_processor (old async one) - the synchronous logic is now directly above

    def start(self):
        """Starts the audio stream and STT processing in background threads."""
        if self._audio_thread or self._stt_thread:
            logger.warning("AudioManager already running or not fully stopped.")
            return

        logger.info("Starting AudioManager...")
        # Indicate processing during overall startup
        if self.status_callback:
            self._update_status("正在启动...", is_processing=True)
        self._stop_event.clear()  # Reset stop event for a new run

        # Start the STT processor thread
        self._stt_thread = threading.Thread(
            target=self._run_stt_processor,  # Target the synchronous method
            name="STTProcessorThread",
            daemon=True,
        )
        self._stt_thread.start()

        # Start the audio stream thread
        self._audio_thread = threading.Thread(
            target=self._run_audio_stream, name="AudioStreamThread", daemon=True
        )
        self._audio_thread.start()

        logger.info("AudioManager threads started.")
        # Status like "Running" will be set by the threads themselves

    def stop(self):
        """Signals the audio stream and STT processing to stop and waits for threads."""
        if not self._audio_thread and not self._stt_thread:
            logger.info("AudioManager is not running.")
            return

        logger.info("Stopping AudioManager...")
        # Indicate processing during overall stop sequence
        # Keep is_running=True initially as things are still shutting down
        if self.status_callback:
            self._update_status("正在停止...", is_running=True, is_processing=True)

        # Signal stop event - this will be checked by loops and sd.Stream wait
        self._stop_event.set()

        # No need to signal an event loop anymore

        # --- Graceful Shutdown ---
        stt_stopped = False
        if self._stt_thread and self._stt_thread.is_alive():
            # Give the STT task/loop time to shut down Dashscope connection gracefully
            # The _stt_processor_task should handle recognizer.stop()
            logger.info("Waiting for STT thread to finish...")
            self._stt_thread.join(timeout=7.0)  # Increased timeout for network ops
            if self._stt_thread.is_alive():
                logger.warning("STT thread did not finish gracefully within timeout.")
                # If thread is stuck, loop might need forceful stop (handled in _run_stt_processor finally)
            else:
                logger.info("STT thread finished.")
                stt_stopped = True
        else:
            logger.info("STT thread was not running or already finished.")
            stt_stopped = True  # Consider it stopped if it wasn't running

        audio_stopped = False
        if self._audio_thread and self._audio_thread.is_alive():
            # The audio thread should exit quickly once stop_event is set
            logger.info("Waiting for audio thread to finish...")
            self._audio_thread.join(timeout=2.0)
            if self._audio_thread.is_alive():
                logger.warning("Audio thread did not finish within timeout.")
            else:
                logger.info("Audio thread finished.")
                audio_stopped = True
        else:
            logger.info("Audio thread was not running or already finished.")
            audio_stopped = True  # Consider it stopped

        # Clean up references
        self._audio_thread = None
        self._stt_thread = None
        # No asyncio refs to clean

        if audio_stopped and stt_stopped:
            logger.info("AudioManager stopped successfully.")
            # Final status: Stopped, not running, not processing
            if self.status_callback: self._update_status("已停止", is_running=False, is_processing=False)
        else:
            logger.warning("AudioManager stopped with potential issues (threads did not join).")
            # Final status: Stopped (with issues), not running, not processing
            if self.status_callback: self._update_status("已停止 (有潜在问题)", is_running=False, is_processing=False)


# --- Remove the old standalone start function ---
# async def start_audio_processing(...): ...
# The logic is now inside the AudioManager class.
