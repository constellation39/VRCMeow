import asyncio
import threading  # <-- Added import
from typing import Optional
from dashscope.audio.asr import (
    TranslationRecognizerCallback,
    TranslationRecognizerRealtime,
    TranscriptionResult,
    TranslationResult,
)
from logger_config import get_logger

# 直接从 config 模块导入 config 实例
from config import config


# Import VRCClient for type hinting and usage in callback
from llm_client import LLMClient
from output_dispatcher import OutputDispatcher
from osc_client import VRCClient  # Import VRCClient


# --- Callback for Gummy API (Translation/Transcription) ---
class GummyCallback(TranslationRecognizerCallback):
    """处理 Dashscope Gummy API (翻译/转录) 事件的回调类。"""

    def __init__(
        self,
        # loop: asyncio.AbstractEventLoop, # Remove loop parameter
        llm_client: Optional[LLMClient],
        output_dispatcher: Optional[OutputDispatcher],
    ):
        super().__init__()
        if not output_dispatcher:
            raise ValueError("OutputDispatcher is required for GummyCallback")

        # self.loop = loop # Remove loop storage
        self.llm_client = llm_client
        self.output_dispatcher = output_dispatcher  # Keep for final dispatch

        # --- Directly access config ---
        stt_config = config.get("stt", {})
        outputs_config = config.get("outputs", {})
        vrc_osc_config = outputs_config.get("vrc_osc", {})

        self.target_language = stt_config.get("translation_target_language")
        self.enable_translation = bool(self.target_language)
        self.intermediate_behavior = stt_config.get(
            "intermediate_result_behavior", "ignore"
        ).lower()
        self.logger = get_logger(
            f"{__name__}.GummyCallback"
        )  # Initialize logger earlier

        # --- VRC Client for Intermediate Messages ---
        # Store VRC client directly if OSC output is enabled for potentially faster intermediate updates
        self.vrc_client_for_intermediate: Optional[VRCClient] = None
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
                # Now self.logger exists
                self.logger.warning(
                    "Intermediate VRC OSC messages enabled in config, but no VRCClient available in OutputDispatcher."
                )
                self.vrc_osc_intermediate_enabled = (
                    False  # Disable if client is missing
                )

        # Log initialization details after all relevant attributes are set
        self.logger.debug(
            f"GummyCallback initialized. Translation: {'Enabled' if self.enable_translation else 'Disabled'} ({self.target_language or 'N/A'}), "
            f"Intermediate Behavior: {self.intermediate_behavior}, "
            f"VRC Intermediate OSC: {'Enabled' if self.vrc_osc_intermediate_enabled else 'Disabled'}"
        )

    # Remove _log_dispatch_result method

    def _dispatch_in_background(self, text: str):
        """
        Handles LLM processing (if enabled) and final dispatch in a separate thread.
        Uses asyncio.run() to manage async operations within the thread.
        """
        thread_id = threading.current_thread().ident
        self.logger.info(
            f"[Thread-{thread_id}] STT_GUMMY_BG: Starting background dispatch for text: '{text[:50]}...'"
        )
        final_text_to_dispatch = text

        # --- LLM Processing (if enabled) ---
        if self.llm_client and self.llm_client.enabled:
            self.logger.info(
                f"[Thread-{thread_id}] STT_GUMMY_BG: Sending to LLM for processing: '{text[:50]}...'"
            )
            try:
                # Use asyncio.run() to execute the async LLM call
                processed_text = asyncio.run(
                    asyncio.wait_for(
                        self.llm_client.process_text(text),
                        timeout=config.get("llm.request_timeout", 10.0),
                    )
                )
                if processed_text:
                    final_text_to_dispatch = processed_text
                    self.logger.info(
                        f"[Thread-{thread_id}] STT_GUMMY_BG: LLM processing successful: '{final_text_to_dispatch[:50]}...'"
                    )
                else:
                    self.logger.warning(
                        f"[Thread-{thread_id}] STT_GUMMY_BG: LLM returned empty result, dispatching original."
                    )
            except asyncio.TimeoutError:
                self.logger.error(
                    f"[Thread-{thread_id}] STT_GUMMY_BG: LLM processing timed out, dispatching original."
                )
            except Exception as e:
                self.logger.error(
                    f"[Thread-{thread_id}] STT_GUMMY_BG: LLM processing error: {e}",
                    exc_info=True,
                )
        else:
            self.logger.debug(
                f"[Thread-{thread_id}] STT_GUMMY_BG: LLM processing disabled."
            )

        # --- Final Dispatch ---
        if self.output_dispatcher:
            self.logger.info(
                f"[Thread-{thread_id}] STT_GUMMY_BG: Calling dispatcher for: '{final_text_to_dispatch[:50]}...'"
            )
            try:
                # Use asyncio.run() to execute the async dispatch call
                asyncio.run(self.output_dispatcher.dispatch(final_text_to_dispatch))
                self.logger.info(
                    f"[Thread-{thread_id}] STT_GUMMY_BG: Dispatch finished successfully."
                )
            except Exception as e:
                self.logger.error(
                    f"[Thread-{thread_id}] STT_GUMMY_BG: Error during dispatch: {e}",
                    exc_info=True,
                )
        else:
            self.logger.error(
                f"[Thread-{thread_id}] STT_GUMMY_BG: OutputDispatcher missing, cannot dispatch."
            )

        self.logger.info(
            f"[Thread-{thread_id}] STT_GUMMY_BG: Background dispatch thread finished."
        )

    def on_open(self) -> None:
        self.logger.info("Dashscope Gummy 连接已打开。")  # Keep as INFO

    def on_close(self) -> None:
        self.logger.info("Dashscope Gummy 连接已关闭。")

    def on_error(self, message: str) -> None:
        self.logger.error(f"Dashscope Gummy 错误: Msg={message}")

    def on_event(
        self,
        request_id,
        transcription_result: Optional[TranscriptionResult],
        translation_result: Optional[TranslationResult],
        usage,
    ) -> None:
        # Log the thread ID where the callback is executed
        self.logger.debug(
            f"GummyCallback.on_event executing in Thread ID: {threading.current_thread().ident}"
        )
        self.logger.debug(f"Dashscope Gummy 事件: ID={request_id}, Usage={usage}")
        text_to_send = None
        log_prefix = ""
        is_final = False

        # 确定结果是否为最终状态
        # 优先检查翻译结果的状态，如果翻译启用且有结果
        if (
            self.enable_translation
            and translation_result
            and translation_result.is_sentence_end
        ):
            is_final = True
        # 如果翻译未启用或无翻译结果，检查转录结果的状态
        elif transcription_result and transcription_result.is_sentence_end:
            is_final = True

        # --- 处理最终结果 ---
        if is_final:
            # 提取最终文本 (优先翻译)
            if self.enable_translation and self.target_language and translation_result:
                try:
                    target_translation = translation_result.get_translation(
                        self.target_language
                    )
                    if target_translation and target_translation.text:
                        text_to_send = f"{target_translation.text}"
                        log_prefix = f"最终翻译 ({self.target_language})"
                    else:  # 翻译结果为空，尝试回退到转录
                        self.logger.debug("最终翻译结果文本为空，尝试使用转录。")
                except KeyError:
                    self.logger.warning(
                        f"最终结果中未找到目标语言 '{self.target_language}' 的翻译，尝试使用转录。"
                    )
                except Exception as e:
                    self.logger.error(f"处理最终翻译结果时出错: {e}", exc_info=True)

            # 如果没有翻译文本，使用转录文本
            if (
                text_to_send is None
                and transcription_result
                and transcription_result.text
            ):
                text_to_send = f"{transcription_result.text}"

            if text_to_send:
                # INFO: Log final text before potential LLM processing
                self.logger.info(f"STT_GUMMY: Final text received: '{text_to_send}'")
            else:
                self.logger.debug(
                    "最终结果文本为空，不发送。"
                )  # Keep this as debug for empty results

        # --- 处理中间结果 ---
        else:
            # --- Extract potential intermediate text first ---
            intermediate_text = None
            # Try translation first if enabled
            if (
                self.enable_translation
                and self.target_language
                and translation_result
            ):
                try:
                    target_translation = translation_result.get_translation(
                        self.target_language
                    )
                    if target_translation and target_translation.text:
                        intermediate_text = f"{target_translation.text}"
                        log_prefix = f"部分翻译 ({self.target_language})"
                except KeyError:
                    pass # Ignore intermediate key errors
                except Exception as e:
                    self.logger.error(f"处理部分翻译结果时出错: {e}", exc_info=True)

            # Fallback to transcription if no translation or translation disabled
            if (
                intermediate_text is None
                and transcription_result
                and transcription_result.text
            ):
                intermediate_text = f"{transcription_result.text}"
                log_prefix = "部分转录" # Update log prefix if using transcription

            # --- Log the extracted intermediate text (if any) ---
            if intermediate_text:
                self.logger.info(f"STT_GUMMY: Intermediate text received: '{intermediate_text}'") # <-- Added log line
            else:
                self.logger.debug("STT_GUMMY: Intermediate event received, but no text extracted.")

            # --- Now handle behavior based on config ---
            if self.intermediate_behavior == "show_typing":
                text_to_send = "Typing..."  # 固定消息
                self.logger.debug("发送 'Typing...' 状态")  # 使用 debug 级别避免刷屏
            elif self.intermediate_behavior == "show_partial":
                # Use the previously extracted intermediate_text
                text_to_send = intermediate_text # Assign the extracted text
                if text_to_send:
                    # Log prefix was already set during extraction
                    self.logger.debug(f"STT_GUMMY: Using intermediate text for 'show_partial': '{text_to_send}'") # Use debug level for behavior log
                else:
                    self.logger.debug("STT_GUMMY: 'show_partial' enabled, but no intermediate text available.")

            # 如果是 "ignore" 或部分文本为空，text_to_send 保持为 None

        # --- 发送 OSC 消息 ---
        # --- Dispatch Final Result (via OutputDispatcher) ---
        if is_final and text_to_send and self.output_dispatcher:
            # This block handles the final result after STT/Translation

            # Create and start a background thread for LLM (if enabled) and dispatching
            self.logger.info(
                f"STT_GUMMY: Preparing background thread for final dispatch of '{text_to_send[:50]}...'"
            )
            dispatch_thread = threading.Thread(
                target=self._dispatch_in_background,
                args=(text_to_send,),
                name=f"GummyDispatchThread-{text_to_send[:10]}",  # Give thread a name
                daemon=True,  # Ensure thread exits if main program exits
            )
            dispatch_thread.start()
            self.logger.info(
                f"STT_GUMMY: Started background dispatch thread: {dispatch_thread.name}"
            )

        # --- Handle Intermediate Results ---
        elif (  # Combined intermediate handling
            not is_final and text_to_send
        ):
            # Intermediate result generated ('Typing...' or partial text)
            if self.vrc_osc_intermediate_enabled and self.vrc_client_for_intermediate:
                # Send directly via VRC client in a background thread
                self.logger.debug(
                    f"STT_GUMMY: Preparing background thread for intermediate VRC OSC: {text_to_send}"
                )
                osc_thread = threading.Thread(
                    target=self._send_osc_intermediate,
                    args=(text_to_send,),
                    name="GummyOscIntermediateThread",
                    daemon=True,
                )
                osc_thread.start()
            else:
                # Intermediate result exists but VRC OSC intermediate sending is disabled or unavailable
                self.logger.debug(
                    f"Intermediate result generated but VRC OSC sending disabled/unavailable: {text_to_send}"
                )

    def _send_osc_intermediate(self, text: str):
        """Sends intermediate OSC message using asyncio.run in a thread."""
        thread_id = threading.current_thread().ident
        if self.vrc_client_for_intermediate:
            self.logger.debug(
                f"[Thread-{thread_id}] STT_GUMMY_OSC_BG: Sending intermediate OSC: {text}"
            )
            try:
                asyncio.run(self.vrc_client_for_intermediate.send_chatbox(text))
                self.logger.debug(
                    f"[Thread-{thread_id}] STT_GUMMY_OSC_BG: Intermediate OSC sent successfully."
                )
            except Exception as e:
                self.logger.error(
                    f"[Thread-{thread_id}] STT_GUMMY_OSC_BG: Error sending intermediate OSC: {e}",
                    exc_info=True,
                )
        else:
            self.logger.error(
                f"[Thread-{thread_id}] STT_GUMMY_OSC_BG: VRC client missing for intermediate OSC."
            )

    # Remove _process_with_llm_and_dispatch method


def create_gummy_recognizer(
    # Remove main_loop parameter
    sample_rate: int,
    llm_client: Optional[LLMClient],
    output_dispatcher: Optional[OutputDispatcher],
) -> TranslationRecognizerRealtime:
    """创建并配置 Dashscope Gummy 实时识别器。"""
    # Check if necessary components are available
    if not output_dispatcher:
        # Log an error or raise? For now, log and proceed without dispatching.
        logger = get_logger(__name__)
        # Raising an error is safer to prevent unexpected behavior
        logger.error(
            "OutputDispatcher 未提供给 create_gummy_recognizer，无法分发最终结果。"
        )
        raise ValueError("OutputDispatcher is required to create the Gummy recognizer.")

    logger = get_logger(__name__)  # Use logger from this module
    logger.debug("使用 Gummy API (支持翻译)")  # Changed to DEBUG

    # --- Directly access config for elements NOT passed as arguments ---
    # Use nested keys with .get() for safety
    api_key = config.get("dashscope.api_key")
    if not api_key:
        # Log a critical error or raise if API key is essential
        logger.error("Dashscope API Key not found in configuration (dashscope.api_key). Cannot create recognizer.")
        raise ValueError("Missing Dashscope API Key in configuration.")

    model = config.get("dashscope.stt.model", "gummy-realtime-v1") # Use nested key and provide default
    # sample_rate is now passed as an argument
    channels = config.get("audio.channels", 1) # Use get() for robustness
    target_language = config.get("dashscope.stt.translation_target_language") # Use nested key
    enable_translation = bool(target_language)

    # Create the callback instance, passing all necessary clients/dispatchers
    callback = GummyCallback(
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
        "transcription_enabled": True,  # Gummy requires explicit enabling
        "translation_enabled": enable_translation,  # 传递派生出的布尔值
    }

    # Add target language only if translation is enabled
    if enable_translation:  # target_language 必然存在，因为 enable_translation 由此派生
        recognizer_params["translation_target_languages"] = [target_language]

    # Initialize the recognizer
    recognizer = TranslationRecognizerRealtime(**recognizer_params)

    # Log initialization details at DEBUG level
    logger.debug(
        f"Dashscope Gummy Recognizer (模型: {model}) 初始化完成。"
    )  # Changed to DEBUG
    logger.debug(f"  - 采样率: {sample_rate}, 声道: {channels}")  # Changed to DEBUG
    if enable_translation:
        translation_log = f"启用, 目标: {target_language}"
    else:
        translation_log = "禁用 (未设置 translation_target_language)"
    logger.debug(f"  - 翻译: {translation_log}")  # Changed to DEBUG

    return recognizer
