import asyncio
import threading
import queue  # Use standard queue for thread-safe communication
from typing import Optional, TYPE_CHECKING, Union, Callable, Any

import numpy as np
import sounddevice as sd

# 直接从 config 模块导入 config 实例
from config import config

# 获取该模块的 logger 实例
from logger_config import get_logger

# Import Dashscope base recognizer types for type hinting
from dashscope.audio.asr import TranslationRecognizerRealtime, Recognition

# Local STT implementation imports
from stt_gummy import create_gummy_recognizer
from stt_paraformer import create_paraformer_recognizer


logger = get_logger(__name__)


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
        status_callback: Optional[
            Callable[[str], None]
        ] = None,  # Callback for status updates
    ):
        self.llm_client = llm_client
        self.output_dispatcher = output_dispatcher
        self.status_callback = status_callback

        self._audio_queue = queue.Queue()  # Thread-safe queue
        self._stop_event = (
            threading.Event()
        )  # Use threading.Event for cross-thread signaling
        self._audio_thread: Optional[threading.Thread] = None
        self._stt_thread: Optional[threading.Thread] = None
        self._stt_task: Optional[asyncio.Task] = (
            None  # To hold the asyncio task reference
        )
        self._stt_loop: Optional[asyncio.AbstractEventLoop] = (
            None  # Loop for the STT thread
        )

        # Load necessary config values during initialization
        self.sample_rate = config.get("audio.sample_rate")  # Get initial value
        self.channels = config["audio.channels"]
        self.dtype = config["audio.dtype"]
        self.debug_echo_mode = config["audio.debug_echo_mode"]
        self.stt_model = config["stt.model"]

        # Dynamically determine sample rate if not configured
        if self.sample_rate is None:
            self.sample_rate = self._determine_sample_rate()

    def _determine_sample_rate(self) -> int:
        """Queries device info to determine sample rate if not set in config."""
        try:
            device_info = sd.query_devices(kind="input")
            if device_info and "default_samplerate" in device_info:
                rate = int(device_info["default_samplerate"])
                logger.info(f"Using default input device sample rate: {rate} Hz")
                return rate
            else:
                logger.warning(
                    "Could not determine default sample rate, falling back to 16000 Hz."
                )
                return 16000
        except Exception as e:
            logger.error(
                f"Error querying audio devices for sample rate: {e}", exc_info=True
            )
            logger.warning("Falling back to 16000 Hz sample rate due to error.")
            return 16000

    def _update_status(self, message: str):
        """Helper to call the status callback if it exists."""
        logger.info(f"AudioManager Status: {message}")  # Log status internally
        if self.status_callback:
            try:
                # Assuming the callback needs to be run in the main (GUI) thread eventually.
                # For now, just call it directly. GUI integration might need thread-safe calls.
                self.status_callback(message)
            except Exception as e:
                logger.error(f"Error calling status callback: {e}", exc_info=True)

    # --- STT Processing Logic (Now an async method) ---
    async def _stt_processor_task(self):
        """Async task for STT processing, run in a dedicated thread's event loop."""
        # Config values are accessed via self now
        model = self.stt_model  # Use pre-loaded config
        # sample_rate is guaranteed by start_audio_processing but unused here
        # sample_rate = config["audio.sample_rate"]
        # channels = config["audio.channels"] # Unused
        target_language = config.get(
            "stt.translation_target_language"
        )  # Re-read in case it changed via reload

        logger.info(
            f"STT processing task (Dashscope, model: {model}) starting in thread..."
        )
        self._update_status(f"STT Task Starting (Model: {model})")
        recognizer: Optional[Union[TranslationRecognizerRealtime, Recognition]] = None
        # main_loop = asyncio.get_running_loop() # Use self._stt_loop which is set when the thread starts
        # except ImportError:
        #      LLMClient = None
        # try:
        #     from output_dispatcher import OutputDispatcher # Not typically needed here
        # except ImportError:
        #     OutputDispatcher = None

        #     from output_dispatcher import OutputDispatcher # Not typically needed here
        # except ImportError:
        #     OutputDispatcher = None
        engine_type = "Unknown"

        # --- Reconnect Parameters ---
        max_retries = 5
        initial_retry_delay = 1.0  # 初始重试延迟 (秒)
        max_retry_delay = 30.0  # 最大重试延迟 (秒)
        retry_count = 0
        current_delay = initial_retry_delay

        # API Key check happens in main.py before AudioManager is created

        # --- Determine translation based on potentially reloaded config ---
        target_language = config.get(
            "stt.translation_target_language"
        )  # Re-check config
        enable_translation = bool(target_language)

        # --- Select API based on potentially reloaded config ---
        # Re-check model compatibility each loop iteration
        model = config["stt.model"]  # Re-read model from config
        self.stt_model = model  # Update instance variable if changed
        is_gummy_model = model.startswith("gummy-")
        is_paraformer_model = model.startswith("paraformer-")

        if not is_gummy_model and not is_paraformer_model:
            error_msg = (
                f"Unsupported Dashscope model: {model}. Use 'gummy-' or 'paraformer-'."
            )
            logger.error(error_msg)
            self._update_status(f"Error: {error_msg}")
            self._stop_event.set()  # Signal stop to all threads
            return

        # --- Outer Reconnect Loop ---
        while not self._stop_event.is_set():
            recognizer = None  # 每次尝试连接前重置
            try:
                # --- Check config and model compatibility (re-check each loop) ---
                model = config["stt.model"]  # Re-read model
                self.stt_model = model  # Update instance var
                target_language = config.get(
                    "stt.translation_target_language"
                )  # Re-read target lang
                enable_translation = bool(target_language)
                is_gummy_model = model.startswith("gummy-")
                is_paraformer_model = model.startswith("paraformer-")

                if not is_gummy_model and not is_paraformer_model:
                    error_msg = f"Unsupported Dashscope model: {model}. Use 'gummy-' or 'paraformer-'."
                    logger.error(error_msg)
                    self._update_status(f"Error: {error_msg}")
                    self._stop_event.set()  # Fatal error, stop everything
                    break  # Exit reconnect loop

                if enable_translation and is_paraformer_model:
                    logger.warning(
                        f"Model '{model}' (Paraformer) does not support translation. "
                        f"'translation_target_language' will be ignored."
                    )
                    enable_translation = False  # Disable translation for this attempt

                # --- Select and create recognizer ---
                self._update_status(
                    f"Connecting STT (Model: {model}, Attempt {retry_count + 1}/{max_retries})..."
                )
                logger.info(
                    f"Attempting to connect STT service (Model: {model}, Attempt {retry_count + 1}/{max_retries})..."
                )
                if is_gummy_model:
                    if create_gummy_recognizer is None:
                        raise RuntimeError("Gummy STT module failed to load.")
                    engine_type = "Gummy"
                    recognizer = create_gummy_recognizer(
                        main_loop=self._stt_loop,
                        sample_rate=self.sample_rate, # Pass the determined sample rate
                        llm_client=self.llm_client,
                        output_dispatcher=self.output_dispatcher,
                    )
                elif is_paraformer_model:
                    if create_paraformer_recognizer is None:
                        raise RuntimeError("Paraformer STT module failed to load.")
                    engine_type = "Paraformer"
                    recognizer = create_paraformer_recognizer(
                        main_loop=self._stt_loop,
                        sample_rate=self.sample_rate, # Pass the determined sample rate
                        llm_client=self.llm_client,
                        output_dispatcher=self.output_dispatcher,
                    )

                if not recognizer:
                    raise RuntimeError(
                        f"Failed to create recognizer instance for model '{model}'."
                    )

                # --- Start Recognizer ---
                recognizer.start()
                logger.info(
                    f"Dashscope {engine_type} Recognizer connected and started."
                )
                self._update_status(f"STT Connected (Engine: {engine_type})")
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
                            continue  # Continue waiting for data

                        # Send audio frame (can raise errors on connection issues)
                        recognizer.send_audio_frame(
                            audio_data.tobytes()
                        )  # audio_data is np.ndarray here

                        self._audio_queue.task_done()  # Mark task as done for the queue
                    except asyncio.TimeoutError:  # This shouldn't happen with queue.get
                        # Should not be reached with queue.get, but kept for safety
                        continue
                    except queue.Full:  # This shouldn't happen with queue.get
                        logger.warning(
                            "Audio queue unexpectedly full in STT processor."
                        )
                        continue

                        # 发送音频帧 - 可能在此处发生连接错误
                        recognizer.send_audio_frame(audio_data.tobytes())

                        audio_queue.task_done()
                    except asyncio.TimeoutError:
                        # 队列为空，继续检查停止信号
                        continue
                    except asyncio.QueueFull:
                        logger.warning("STT 处理器中的音频队列意外已满。")
                        continue
                    except Exception as send_error:
                        # 捕获发送音频帧时的错误，假设是连接问题
                        logger.error(
                            f"Error sending audio frame to STT service: {send_error}",
                            exc_info=True,
                        )
                        self._update_status(f"STT Send Error: {send_error}")
                        logger.info("Assuming connection lost, attempting reconnect...")
                        try:
                            # Attempt to stop the failed recognizer
                            recognizer.stop()
                            logger.info(
                                f"Stopped failed Dashscope {engine_type} Recognizer instance."
                            )
                        except Exception as stop_err:
                            logger.error(
                                f"Error stopping failed recognizer: {stop_err}",
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
                self._update_status(f"STT Connect Error: {connect_error}")
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
                    # Assuming the user's provided actual code snipped had `error_msg` defined earlier
                    # or intended to define it here. Re-creating based on the previous REPLACE logic.
                    error_msg = f"STT connection failed after {max_retries} retries. Stopping STT processing."
                    logger.critical(error_msg)
                    self._update_status(f"Error: {error_msg}")
                    self._stop_event.set()  # Signal stop
                    break  # Exit reconnect loop

                wait_msg = f"Retrying STT connection in {current_delay:.1f} seconds..."
                logger.info(wait_msg)
                self._update_status(wait_msg)
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
        logger.info(f"STT processing task (Dashscope {engine_type}) stopping...")
        self._update_status(f"STT Task Stopping (Engine: {engine_type})")
        if recognizer:  # Check if a recognizer was active when the loop exited
            try:
                logger.info(
                    f"Stopping final Dashscope {engine_type} Recognizer instance..."
                )
                recognizer.stop()  # This might block briefly
                logger.info(
                    f"Final Dashscope {engine_type} Recognizer instance stopped."
                )
                self._update_status("STT Recognizer Stopped.")
            except Exception as e:
                logger.error(
                    f"Error stopping final recognizer instance: {e}", exc_info=True
                )
                self._update_status(f"Error stopping STT: {e}")
        else:
            logger.info("No active Recognizer instance to stop.")

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
            logger.info(f"Drained {drained_count} items from audio queue.")

        logger.info(f"STT processing task (Dashscope {engine_type}) fully stopped.")
        self._update_status("STT Task Stopped.")

        # --- Audio Stream Handling (Now methods within AudioManager) ---

        # This method belongs inside the AudioManager class based on the refactoring

    def _audio_callback(  # Correct indentation for method
        self,  # Add self for instance method
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        time: Any,  # Use Any for time if type isn't strictly defined
        status: sd.CallbackFlags,
    ):
        """Synchronous callback for sounddevice stream."""
        if status:
            logger.warning(f"Audio callback status: {status}")
            # Potentially update status for critical flags?
            # self._update_status(f"Audio Status: {status}")

        # Echo mode (using instance variable)
        if self.debug_echo_mode:
            outdata[:] = indata
        else:
            outdata.fill(0)  # Output silence in normal mode

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
        except queue.Full:
            logger.warning("Audio queue is full. Dropping audio frame.")
            # Consider adding a status update here if it happens frequently
            # self._update_status("Warning: Audio queue full, dropping data")
            pass  # Continue processing

    def _run_audio_stream(self):
        """Target function for the audio processing thread."""
        try:
            # Log configuration being used
            logger.info("Audio Stream Thread Started. Config:")
            logger.info(f"  Sample Rate: {self.sample_rate} Hz")
            logger.info(f"  Channels: {self.channels}")
            logger.info(f"  Dtype: {self.dtype}")
            logger.info(f"  Debug Echo: {self.debug_echo_mode}")
            self._update_status("Audio Stream Starting...")

            # Use sounddevice Stream context manager
            with sd.Stream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                callback=self._audio_callback,  # Use the instance method
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
                self._update_status("Audio Stream Running")
                # Keep the stream running until stop_event is set
                self._stop_event.wait()  # Block here until stop() is called

        except sd.PortAudioError as e:
            error_msg = f"PortAudio Error: {e}. Check devices and permissions."
            logger.error(error_msg, exc_info=True)
            logger.error("Ensure microphone/speakers are connected and not in use.")
            logger.error(
                f"Check support for {self.sample_rate} Hz, {self.channels} channels, {self.dtype}."
            )
            self._update_status(f"Error: {error_msg}")
            self._stop_event.set()  # Signal other threads to stop
        except ValueError as e:
            error_msg = f"Audio Parameter Error: {e}. Check config."
            logger.error(error_msg, exc_info=True)
            logger.error(
                f"Verify sample rate ({self.sample_rate}), channels ({self.channels}), dtype ({self.dtype})."
            )
            self._update_status(f"Error: {error_msg}")
            self._stop_event.set()
        except Exception as e:
            error_msg = f"Unknown error in audio stream thread: {e}"
            logger.error(error_msg, exc_info=True)
            self._update_status(f"Error: {error_msg}")
            self._stop_event.set()  # Signal stop on unexpected errors
        finally:
            logger.info("Audio stream thread finishing.")
            # Ensure stop event is set if this thread exits unexpectedly
            self._stop_event.set()
            # Status update upon stopping is handled in the stop() method generally

    def _run_stt_processor(self):
        """Target function for the STT processing thread."""
        logger.info("STT Processor Thread Started.")
        self._update_status("STT Thread Starting...")
        try:
            # Create a new event loop for this thread
            self._stt_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._stt_loop)

            # Schedule the async task within this loop
            self._stt_task = self._stt_loop.create_task(self._stt_processor_task())

            # Run the event loop until the task completes or stop is signaled
            self._stt_loop.run_until_complete(self._stt_task)

        except Exception as e:
            error_msg = f"Error in STT processor thread: {e}"
            logger.error(error_msg, exc_info=True)
            self._update_status(f"Error: {error_msg}")
        finally:
            logger.info("STT processor thread finishing.")
            if self._stt_loop and self._stt_loop.is_running():
                self._stt_loop.stop()  # Stop the loop if it's still running
            if self._stt_loop:
                # Cancel any remaining tasks in the loop before closing
                for task in asyncio.all_tasks(self._stt_loop):
                    if not task.done():
                        task.cancel()
                # Run loop briefly to allow cancellations to process
                try:
                    # Gather cancelled tasks to suppress CancelledError propagation
                    cancelled_tasks = [
                        task
                        for task in asyncio.all_tasks(self._stt_loop)
                        if task.cancelled()
                    ]
                    if cancelled_tasks:
                        self._stt_loop.run_until_complete(
                            asyncio.gather(*cancelled_tasks, return_exceptions=True)
                        )
                except Exception as loop_cancel_err:
                    logger.error(
                        f"Error during STT loop task cancellation: {loop_cancel_err}",
                        exc_info=True,
                    )

                # Close the loop
                try:
                    self._stt_loop.close()
                    logger.info("STT event loop closed.")
                except Exception as loop_close_err:
                    logger.error(
                        f"Error closing STT event loop: {loop_close_err}", exc_info=True
                    )

            self._stt_loop = None
            self._stt_task = None
            # Ensure stop event is set if this thread exits unexpectedly
            self._stop_event.set()
            # Status update upon stopping is handled in the stop() method

    def start(self):
        """Starts the audio stream and STT processing in background threads."""
        if self._audio_thread or self._stt_thread:
            logger.warning("AudioManager already running or not fully stopped.")
            return

        logger.info("Starting AudioManager...")
        self._update_status("Starting...")
        self._stop_event.clear()  # Reset stop event for a new run

        # Start the STT processor thread first (it needs the loop ready)
        self._stt_thread = threading.Thread(
            target=self._run_stt_processor, name="STTProcessorThread", daemon=True
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
        self._update_status("Stopping...")

        # Signal stop event - this will be checked by loops and sd.Stream wait
        self._stop_event.set()

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
        # self._stt_task = None # Cleaned up in _run_stt_processor finally
        # self._stt_loop = None # Cleaned up in _run_stt_processor finally

        if audio_stopped and stt_stopped:
             logger.info("AudioManager stopped successfully.")
             self._update_status("Stopped")
        else:
             logger.warning("AudioManager stopped with potential issues (threads did not join).")
             self._update_status("Stopped (with issues)")


# --- Remove the old standalone start function ---
# async def start_audio_processing(...): ...
# The logic is now inside the AudioManager class.
