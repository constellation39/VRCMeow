import asyncio
import threading  # Import threading
from typing import Optional
from dashscope.audio.asr import RecognitionCallback, Recognition, RecognitionResult
import time  # Import time for LLM timeout and typing status
from logger_config import get_logger

# 直接从 config 模块导入 config 实例
from config import config

# Import LLMClient and OutputDispatcher for type hinting and usage
from llm_client import LLMClient
from output_dispatcher import OutputDispatcher
from osc_client import VRCClient  # <-- Add this import


# --- Callback for Paraformer API (Recognition only) ---
class ParaformerCallback(RecognitionCallback):
    """处理 Dashscope Paraformer API (仅识别) 事件的回调类。"""

    def __init__(
        self,
        # loop: asyncio.AbstractEventLoop, # Remove loop parameter
        llm_client: Optional[LLMClient],
        output_dispatcher: Optional[OutputDispatcher],
    ):
        super().__init__()
        if not output_dispatcher:
            raise ValueError("OutputDispatcher is required for ParaformerCallback")

        # self.loop = loop # Remove loop storage
        self.llm_client = llm_client
        self.output_dispatcher = output_dispatcher
        # --- 直接从 config 实例获取配置 ---
        self.intermediate_behavior = config.get(
            "stt.intermediate_result_behavior", "ignore"
        ).lower()
        self.logger = get_logger(f"{__name__}.ParaformerCallback")
        self.logger.debug("ParaformerCallback 初始化完成。")  # Changed to DEBUG
        self.logger.debug(
            f"  - 中间结果处理: {self.intermediate_behavior}"
        )  # Changed to DEBUG

        # --- VRC Client for Intermediate Messages (similar to Gummy) ---
        outputs_config = config.get("outputs", {})
        vrc_osc_config = outputs_config.get("vrc_osc", {})
        self.vrc_client_for_intermediate: Optional[VRCClient] = (
            None  # Import VRCClient if not already
        )
        self.vrc_osc_intermediate_enabled = (
            vrc_osc_config.get("enabled", True)
            and self.intermediate_behavior != "ignore"
        )
        if self.vrc_osc_intermediate_enabled:
            # We need the VRCClient instance from the OutputDispatcher
            self.vrc_client_for_intermediate = (
                self.output_dispatcher.vrc_client
            )  # May be None
            if not self.vrc_client_for_intermediate:
                self.logger.warning(
                    "Intermediate VRC OSC messages enabled in config, but no VRCClient available in OutputDispatcher."
                )
                self.vrc_osc_intermediate_enabled = (
                    False  # Disable if client is missing
                )

        self._last_typing_send_time = 0  # Track typing status send time
        self._typing_interval = (
            1.0  # Minimum interval between typing status updates (seconds)
        )
        self.logger.debug(
            f"  - VRC 中间 OSC: {'启用' if self.vrc_osc_intermediate_enabled else '禁用'}"
        )

    def _dispatch_in_background(self, text: str):
        """
        Handles LLM processing (if enabled) and final dispatch in a separate thread.
        Uses asyncio.run() to manage async operations within the thread.
        """
        thread_id = threading.current_thread().ident
        self.logger.info(
            f"[Thread-{thread_id}] STT_PARA_BG: Starting background dispatch for text: '{text[:50]}...'"
        )
        final_text_to_dispatch = text

        # --- LLM Processing (if enabled) ---
        if self.llm_client and self.llm_client.enabled:
            self.logger.info(
                f"[Thread-{thread_id}] STT_PARA_BG: Sending to LLM for processing: '{text[:50]}...'"
            )
            try:
                processed_text = asyncio.run(
                    asyncio.wait_for(
                        self.llm_client.process_text(text),
                        timeout=config.get("llm.request_timeout", 10.0),
                    )
                )
                if processed_text:
                    final_text_to_dispatch = processed_text
                    self.logger.info(
                        f"[Thread-{thread_id}] STT_PARA_BG: LLM processing successful: '{final_text_to_dispatch[:50]}...'"
                    )
                else:
                    self.logger.warning(
                        f"[Thread-{thread_id}] STT_PARA_BG: LLM returned empty result, dispatching original."
                    )
            except asyncio.TimeoutError:
                self.logger.error(
                    f"[Thread-{thread_id}] STT_PARA_BG: LLM processing timed out, dispatching original."
                )
            except Exception as e:
                self.logger.error(
                    f"[Thread-{thread_id}] STT_PARA_BG: LLM processing error: {e}",
                    exc_info=True,
                )
        else:
            self.logger.debug(
                f"[Thread-{thread_id}] STT_PARA_BG: LLM processing disabled."
            )

        # --- Final Dispatch ---
        if self.output_dispatcher:
            self.logger.info(
                f"[Thread-{thread_id}] STT_PARA_BG: Calling dispatcher for: '{final_text_to_dispatch[:50]}...'"
            )
            try:
                asyncio.run(self.output_dispatcher.dispatch(final_text_to_dispatch))
                self.logger.info(
                    f"[Thread-{thread_id}] STT_PARA_BG: Dispatch finished successfully."
                )
            except Exception as e:
                self.logger.error(
                    f"[Thread-{thread_id}] STT_PARA_BG: Error during dispatch: {e}",
                    exc_info=True,
                )
        else:
            self.logger.error(
                f"[Thread-{thread_id}] STT_PARA_BG: OutputDispatcher missing, cannot dispatch."
            )

        self.logger.info(
            f"[Thread-{thread_id}] STT_PARA_BG: Background dispatch thread finished."
        )

    # Removed _dispatch_intermediate_in_background method
    # Intermediate messages are now handled by _send_osc_intermediate if enabled

    def on_open(self) -> None:
        self.logger.info("Dashscope Paraformer 连接已打开。")  # Keep as INFO

    def on_close(self) -> None:
        self.logger.info("Dashscope Paraformer 连接已关闭。")  # Keep as INFO

    def on_complete(self) -> None:
        # This method is called when the recognizer finishes processing.
        # It might correspond to receiving a 'task-finished' type event internally.
        self.logger.info(
            "Dashscope Paraformer task completed (on_complete received)."
        )  # Keep as INFO, updated message

    def on_error(self, message) -> None:
        # Paraformer error structure might contain request_id and message
        error_msg = (
            f"Msg={message.message}" if hasattr(message, "message") else str(message)
        )
        request_id = (
            f"ID={message.request_id}" if hasattr(message, "request_id") else ""
        )
        # Attempt to extract details similar to the 'task-failed' event structure
        error_code = getattr(
            message, "error_code", "UNKNOWN_CODE"
        )  # Check if SDK provides error_code
        error_message = getattr(
            message, "error_message", str(message)
        )  # Check for error_message, fallback to str()
        request_id = getattr(
            message, "request_id", "UNKNOWN_ID"
        )  # Check for request_id

        self.logger.error(
            f"Dashscope Paraformer 错误 (task-failed): ID={request_id}, Code={error_code}, Message={error_message}"
        )
        # Consider triggering stop_event here if needed
        # self.loop.call_soon_threadsafe(stop_event.set) # Requires passing stop_event

    def on_event(self, result: RecognitionResult) -> None:
        # Process Paraformer's RecognitionResult
        sentence_data = result.get_sentence()  # Renamed for clarity
        request_id = result.get_request_id()
        usage = result.get_usage(sentence_data)  # Get usage info
        # Log the thread ID where the callback is executed
        self.logger.debug(
            f"ParaformerCallback.on_event executing in Thread ID: {threading.current_thread().ident}"
        )
        self.logger.debug(
            f"Dashscope Paraformer 事件: ID={request_id}, Usage={usage}, Sentence={sentence_data}"
        )

        text_to_process = None
        log_prefix = ""
        is_final = False

        # Determine if the result is final based on 'end_time'.
        # According to the documentation for the 'result-generated' event,
        # if payload.output.sentence.end_time is not null, it's a final result for that sentence.
        if (
            sentence_data
            and "end_time" in sentence_data
            and sentence_data["end_time"] is not None # Check if end_time exists and is not null
        ):
            is_final = True

        # --- Extract Text ---
        if sentence_data and "text" in sentence_data and sentence_data["text"]:
            text_to_process = sentence_data["text"]
        else:
            # No text in this event, nothing to do
            self.logger.debug("Paraformer 事件无有效文本。")
            return

        # --- Handle Final Result ---
        if is_final:
            log_prefix = "最终识别"
            self.logger.info(f"{log_prefix}: {text_to_process}")
            # INFO: Log final text before potential LLM processing
            self.logger.info(f"STT_PARA: Final text received: '{text_to_process}'")

            # --- Log LLM processing intent ---
            if self.llm_client and self.llm_client.enabled:
                self.logger.info(
                    "STT_PARA: LLM processing is enabled. Will attempt in background."
                )
            else:
                self.logger.info("STT_PARA: LLM processing is disabled.")

            # --- Start Background Thread for LLM/Dispatch ---
            self.logger.info(
                f"STT_PARA: Preparing background thread for final dispatch of '{text_to_process[:50]}...'"
            )
            # --- Log confirmation of background process initiation ---
            self.logger.info(
                "STT_PARA: Initiating background dispatch process..."
            )  # Added log  # noqa: F541

            dispatch_thread = threading.Thread(
                target=self._dispatch_in_background,
                args=(text_to_process,),
                name=f"ParaformerDispatchThread-{text_to_process[:10]}",
                daemon=True,
            )
            dispatch_thread.start()
            self.logger.info(
                f"STT_PARA: Started background dispatch thread: {dispatch_thread.name}"
            )

        # --- Handle Intermediate Result ---
        else:  # Not final (Intermediate result from Paraformer)
            # Log the intermediate text regardless of behavior setting
            self.logger.info(
                f"STT_PARA: Intermediate text received: '{text_to_process}'"
            )  # <-- Added log line

            if self.intermediate_behavior == "show_typing":
                # Send "Typing..." status periodically via dispatcher
                current_time = time.monotonic()
                if current_time - self._last_typing_send_time >= self._typing_interval:
                    log_prefix = "中间状态 (Typing...)"
                    # Send "Typing..." only via VRC OSC if enabled
                    if (
                        self.vrc_osc_intermediate_enabled
                        and self.vrc_client_for_intermediate
                    ):
                        self.logger.debug(
                            f"STT_PARA: Preparing background thread for intermediate VRC OSC: '{log_prefix}'"
                        )
                        osc_thread = threading.Thread(
                            target=self._send_osc_intermediate,
                            args=("Typing...",),
                            name="ParaformerOscIntermediateThread-Typing",
                            daemon=True,
                        )
                        osc_thread.start()
                        self._last_typing_send_time = (
                            current_time  # Update last send time
                        )
                    else:
                        self.logger.debug(
                            f"Intermediate result '{log_prefix}' generated but VRC OSC sending disabled/unavailable."
                        )
                # else: # Suppress frequent typing updates
                #    self.logger.debug("Typing... 状态更新已抑制 (过于频繁)")

            elif self.intermediate_behavior == "show_partial":
                # Paraformer doesn't really give partials like Gummy.
                # We could send the latest non-final sentence, but it might be confusing.
                # Paraformer doesn't provide granular partial results like Gummy.
                # We will send the latest non-final sentence as the partial result via VRC OSC if enabled.
                log_prefix = "中间结果 (部分)"
                self.logger.debug(f"STT_PARA: {log_prefix}: {text_to_process}")
                # Send partial text only via VRC OSC if enabled
                if (
                    self.vrc_osc_intermediate_enabled
                    and self.vrc_client_for_intermediate
                ):
                    self.logger.debug(
                        f"STT_PARA: Preparing background thread for intermediate VRC OSC: '{text_to_process[:20]}...'"
                    )
                    osc_thread = threading.Thread(
                        target=self._send_osc_intermediate,
                        args=(text_to_process,),
                        name="ParaformerOscIntermediateThread-Partial",
                        daemon=True,
                    )
                    osc_thread.start()
                else:
                    self.logger.debug(
                        f"Intermediate result '{log_prefix}' generated but VRC OSC sending disabled/unavailable."
                    )
                # Note: We don't use _last_typing_send_time here as we dispatch every partial result if enabled.

            # If behavior is "ignore", do nothing for intermediate results.

    def _send_osc_intermediate(self, text: str):
        """Sends intermediate OSC message using asyncio.run in a thread."""
        thread_id = threading.current_thread().ident
        if self.vrc_client_for_intermediate:
            self.logger.debug(
                f"[Thread-{thread_id}] STT_PARA_OSC_BG: Sending intermediate OSC: {text}"
            )
            try:
                # Ensure VRCClient is imported at the top
                asyncio.run(self.vrc_client_for_intermediate.send_chatbox(text))
                self.logger.debug(
                    f"[Thread-{thread_id}] STT_PARA_OSC_BG: Intermediate OSC sent successfully."
                )
            except Exception as e:
                self.logger.error(
                    f"[Thread-{thread_id}] STT_PARA_OSC_BG: Error sending intermediate OSC: {e}",
                    exc_info=True,
                )
        else:
            # This case should ideally be prevented by the vrc_osc_intermediate_enabled check before calling
            self.logger.error(
                f"[Thread-{thread_id}] STT_PARA_OSC_BG: VRC client missing for intermediate OSC."
            )

    # Remove _process_with_llm_and_dispatch method


def create_paraformer_recognizer(
    # Remove main_loop parameter
    sample_rate: int,
    llm_client: Optional[LLMClient],
    output_dispatcher: Optional[OutputDispatcher],
) -> Recognition:
    """创建并配置 Dashscope Paraformer 实时识别器。"""
    # Check if necessary components are available
    if not output_dispatcher:
        logger = get_logger(__name__)
        logger.error(
            "OutputDispatcher 未提供给 create_paraformer_recognizer，无法分发结果。"
        )
        raise ValueError(
            "OutputDispatcher is required to create the Paraformer recognizer."
        )

    logger = get_logger(__name__)  # Use logger from this module
    logger.debug("使用 Paraformer API (仅识别)")  # Changed to DEBUG

    # --- 直接从 config 实例获取配置 ---
    # --- Directly access config for elements NOT passed as arguments ---
    # Use nested keys with .get() for safety
    api_key = config.get("dashscope.api_key")
    if not api_key:
        # Log a critical error or raise if API key is essential
        logger.error(
            "Dashscope API Key not found in configuration (dashscope.api_key). Cannot create recognizer."
        )
        raise ValueError("Missing Dashscope API Key in configuration.")

    model = config.get(
        "dashscope.stt.model", "paraformer-realtime-v1"
    )  # Use nested key and provide default
    # sample_rate is now passed as an argument
    channels = config.get("audio.channels", 1)  # Use get() for robustness

    # Create the callback instance, passing dispatcher and llm client
    callback = ParaformerCallback(
        # Remove loop argument
        llm_client=llm_client,
        output_dispatcher=output_dispatcher,
    )

    # Prepare parameters for the recognizer
    recognizer_params = {
        "model": model,
        "format": "pcm",
        "sample_rate": sample_rate,
        "channels": channels,
        "api_key": api_key,
        "callback": callback,
        # Add Paraformer-specific parameters if needed, e.g.:
        # "semantic_punctuation_enabled": False,
    }

    # Initialize the recognizer
    recognizer = Recognition(**recognizer_params)

    # Log initialization details at DEBUG level
    logger.debug(
        f"Dashscope Paraformer Recognizer (模型: {model}) 初始化完成。"
    )  # Changed to DEBUG
    logger.debug(f"  - 采样率: {sample_rate}, 声道: {channels}")  # Changed to DEBUG
    logger.debug("  - 翻译: 不支持")  # Changed to DEBUG

    return recognizer
