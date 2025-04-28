import asyncio
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


# Import LLMClient and OutputDispatcher for type hinting and usage


# --- Callback for Gummy API (Translation/Transcription) ---
class GummyCallback(TranslationRecognizerCallback):
    """处理 Dashscope Gummy API (翻译/转录) 事件的回调类。"""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        llm_client: Optional[LLMClient],
        output_dispatcher: Optional[OutputDispatcher],
    ):
        super().__init__()
        if not output_dispatcher:
            raise ValueError("OutputDispatcher is required for GummyCallback")

        self.loop = loop
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
                log_prefix = "最终转录"

            if text_to_send:
                self.logger.info(f"{log_prefix}: {text_to_send}")
                # INFO: Log final text before potential LLM processing
                self.logger.info(f"STT_GUMMY: Final text received: '{text_to_send}'")
            else:
                self.logger.debug("最终结果文本为空，不发送。") # Keep this as debug for empty results

        # --- 处理中间结果 ---
        else:
            if self.intermediate_behavior == "show_typing":
                text_to_send = "Typing..."  # 固定消息
                log_prefix = "中间状态"
                self.logger.debug("发送 'Typing...' 状态")  # 使用 debug 级别避免刷屏
            elif self.intermediate_behavior == "show_partial":
                # 提取部分文本 (优先翻译)
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
                            text_to_send = f"{target_translation.text}"
                            log_prefix = f"部分翻译 ({self.target_language})"
                    except KeyError:
                        pass  # 忽略中间结果的 KeyErrors
                    except Exception as e:
                        self.logger.error(f"处理部分翻译结果时出错: {e}", exc_info=True)

                # 如果没有部分翻译文本，使用部分转录文本
                if (
                    text_to_send is None
                    and transcription_result
                    and transcription_result.text
                ):
                    text_to_send = f"{transcription_result.text}"
                    log_prefix = "部分转录"

                if text_to_send:
                    self.logger.debug(
                        f"{log_prefix}: {text_to_send}"
                    )  # 使用 debug 级别避免刷屏
                # else: # 部分结果为空时无需记录
                #    self.logger.debug("部分结果文本为空，不发送。")

            # 如果是 "ignore" 或部分文本为空，text_to_send 保持为 None

        # --- 发送 OSC 消息 ---
        # --- Dispatch Final Result (via OutputDispatcher) ---
        if is_final and text_to_send and self.output_dispatcher:
            # This block handles the final result after STT/Translation

            # Schedule the processing and dispatching asynchronously
            if self.loop.is_running():
                if self.llm_client and self.llm_client.enabled:
                     # Schedule the helper function that handles LLM and dispatch
                     target_coro = self._process_with_llm_and_dispatch(text_to_send)
                     self.logger.info(f"STT_GUMMY: PRE-SCHEDULE LLM processing & dispatch for: '{text_to_send[:50]}...'")
                     future = asyncio.run_coroutine_threadsafe(target_coro, self.loop)
                     self.logger.info(f"STT_GUMMY: POST-SCHEDULE LLM processing & dispatch (Future: {future})")
                elif self.output_dispatcher:
                     # LLM disabled, schedule dispatch directly
                     target_coro = self.output_dispatcher.dispatch(text_to_send)
                     self.logger.info(f"STT_GUMMY: PRE-SCHEDULE direct dispatch for: '{text_to_send[:50]}...'")
                     future = asyncio.run_coroutine_threadsafe(target_coro, self.loop)
                     self.logger.info(f"STT_GUMMY: POST-SCHEDULE direct dispatch (Future: {future})")
                else:
                     self.logger.error("STT_GUMMY: Cannot dispatch final text - OutputDispatcher missing.")

                self.logger.info( # Log that scheduling happened
                    f"STT_GUMMY: Scheduled final processing/dispatch for '{text_to_send[:50]}...'"
                )
            else:
                self.logger.warning("事件循环未运行，无法调度最终结果处理。")

        # --- Handle Intermediate Results (Directly to VRC OSC if configured) ---
        elif (
            not is_final
            and text_to_send
            and self.vrc_osc_intermediate_enabled
            and self.vrc_client_for_intermediate
        ):
            # Send 'Typing...' or partial text directly via VRC client, bypassing the full dispatcher
            # This avoids logging intermediate messages to file/console.
            if self.loop.is_running():
                # Use the stored VRC client instance directly
                asyncio.run_coroutine_threadsafe(
                    self.vrc_client_for_intermediate.send_chatbox(text_to_send),
                    self.loop,
                )
                # Use debug level for potentially frequent intermediate messages
                self.logger.debug(f"Sent intermediate VRC OSC message: {text_to_send}")
            else:
                self.logger.warning(
                    "Event loop not running, cannot send intermediate VRC OSC message."
                )
        elif not is_final and text_to_send:
            # Intermediate result exists but VRC OSC intermediate sending is disabled or unavailable
            self.logger.debug(
                f"Intermediate result generated but not sent via VRC OSC: {text_to_send}"
            )

    async def _process_with_llm_and_dispatch(self, original_text: str):
        """Processes text with LLM (if enabled) and then dispatches."""
        final_text_to_dispatch = original_text
        if self.llm_client and self.llm_client.enabled:
            self.logger.info(f"STT_GUMMY: Sending to LLM for processing: '{original_text[:50]}...'")
            try:
                # Await the LLM processing directly here
                processed_text = await asyncio.wait_for(
                    self.llm_client.process_text(original_text),
                    timeout=config.get("llm.request_timeout", 10.0) # Use configured timeout
                )
                if processed_text:
                    final_text_to_dispatch = processed_text
                    self.logger.info(
                        f"LLM 处理完成: '{final_text_to_dispatch[:50]}...'"
                    )
                else:
                    # LLMClient handles logging errors/empty results internally
                    self.logger.warning("LLM 处理失败或返回空，将分发原始文本。")
            except asyncio.TimeoutError:
                self.logger.error("LLM 处理超时，将分发原始文本。")
            except Exception as e:
                self.logger.error(
                    f"LLM 处理过程中发生错误: {e}", exc_info=True
                )
        else:
             self.logger.debug("LLM processing disabled, dispatching original text.")

        # Dispatch the final text (original or processed) using the dispatcher
        if self.loop.is_running() and self.output_dispatcher:
            # INFO: Log just before awaiting the actual dispatch call within the helper coroutine
            self.logger.info(f"STT_GUMMY_HELPER: About to await output_dispatcher.dispatch for: '{final_text_to_dispatch}'")
            # Directly await the dispatch coroutine as we are already in an async context managed by the loop
            await self.output_dispatcher.dispatch(final_text_to_dispatch)
            self.logger.info(
                f"STT_GUMMY_HELPER: Finished awaiting output_dispatcher.dispatch for '{final_text_to_dispatch[:50]}...'"
            )
        else:
             # This case might happen if the loop stops between LLM processing and dispatching
             self.logger.warning("事件循环未运行或分发器不可用，无法分发最终文本。")


def create_gummy_recognizer(
    main_loop: asyncio.AbstractEventLoop,
    sample_rate: int, # Add sample_rate argument
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
    api_key = config["dashscope_api_key"]
    model = config["stt.model"]
    # sample_rate is now passed as an argument
    channels = config["audio.channels"]
    target_language = config.get("stt.translation_target_language")
    enable_translation = bool(target_language)

    # Create the callback instance, passing all necessary clients/dispatchers
    callback = GummyCallback(
        loop=main_loop, llm_client=llm_client, output_dispatcher=output_dispatcher
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
