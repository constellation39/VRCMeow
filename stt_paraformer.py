import asyncio
from typing import Optional
from dashscope.audio.asr import RecognitionCallback, Recognition, RecognitionResult
import time  # Import time for LLM timeout
from logger_config import get_logger

# 直接从 config 模块导入 config 实例
from config import config

# Import LLMClient and OutputDispatcher for type hinting and usage
from llm_client import LLMClient

from output_dispatcher import OutputDispatcher


# --- Callback for Paraformer API (Recognition only) ---
class ParaformerCallback(RecognitionCallback):
    """处理 Dashscope Paraformer API (仅识别) 事件的回调类。"""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        llm_client: Optional[LLMClient],
        output_dispatcher: Optional[OutputDispatcher],
    ):
        super().__init__()
        if not output_dispatcher:
            raise ValueError("OutputDispatcher is required for ParaformerCallback")

        self.loop = loop
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
        self._last_typing_send_time = 0  # Track typing status send time
        self._typing_interval = (
            1.0  # Minimum interval between typing status updates (seconds)
        )

    def on_open(self) -> None:
        self.logger.info("Dashscope Paraformer 连接已打开。")  # Keep as INFO

    def on_close(self) -> None:
        self.logger.info("Dashscope Paraformer 连接已关闭。")  # Keep as INFO

    def on_complete(self) -> None:
        # Paraformer specific event - might be useful for final cleanup if needed
        self.logger.info(
            "Dashscope Paraformer 识别完成 (on_complete)。"
        )  # Keep as INFO

    def on_error(self, message) -> None:
        # Paraformer error structure might contain request_id and message
        error_msg = (
            f"Msg={message.message}" if hasattr(message, "message") else str(message)
        )
        request_id = (
            f"ID={message.request_id}" if hasattr(message, "request_id") else ""
        )
        self.logger.error(f"Dashscope Paraformer 错误: {request_id} {error_msg}")
        # Consider triggering stop_event here if needed
        # self.loop.call_soon_threadsafe(stop_event.set) # Requires passing stop_event

    def on_event(self, result: RecognitionResult) -> None:
        # Process Paraformer's RecognitionResult
        sentence_data = result.get_sentence()  # Renamed for clarity
        request_id = result.get_request_id()
        usage = result.get_usage(sentence_data)  # Get usage info
        self.logger.debug(
            f"Dashscope Paraformer 事件: ID={request_id}, Usage={usage}, Sentence={sentence_data}"
        )

        text_to_process = None
        log_prefix = ""
        is_final = False

        # Determine if the result is final based on 'status'
        if (
            sentence_data
            and "status" in sentence_data
            and sentence_data["status"] == "sentence_end"
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

            # --- LLM Processing (if enabled) ---
            final_text_to_dispatch = text_to_process  # Default to original text
            if self.llm_client and self.llm_client.enabled:
                self.logger.debug(f"尝试 LLM 处理: '{text_to_process[:50]}...'")
                llm_future = asyncio.run_coroutine_threadsafe(
                    self.llm_client.process_text(text_to_process), self.loop
                )
                try:
                    # Wait for LLM result with a timeout
                    processed_text = llm_future.result(
                        timeout=config.get("llm.request_timeout", 10.0)
                    )  # Use config timeout
                    if processed_text:
                        final_text_to_dispatch = processed_text
                        self.logger.info(
                            f"LLM 处理完成: '{final_text_to_dispatch[:50]}...'"
                        )
                    else:
                        self.logger.warning("LLM 处理失败或返回空，将分发原始文本。")
                except asyncio.TimeoutError:
                    self.logger.error("LLM 处理超时，将分发原始文本。")
                except Exception as e:
                    self.logger.error(
                        f"等待 LLM 处理结果时发生错误: {e}", exc_info=True
                    )

            # --- Dispatch Final Result ---
            if self.loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self.output_dispatcher.dispatch(final_text_to_dispatch), self.loop
                )
                self.logger.debug(
                    f"已调度最终文本 '{final_text_to_dispatch[:50]}...' 进行分发"
                )
            else:
                self.logger.warning("事件循环未运行，无法调度最终结果分发。")

        # --- Handle Intermediate Result ---
        else:  # Not final
            if self.intermediate_behavior == "show_typing":
                # Send "Typing..." status periodically via dispatcher
                current_time = time.monotonic()
                if current_time - self._last_typing_send_time >= self._typing_interval:
                    log_prefix = "中间状态"
                    # Dispatch "Typing..." - OutputDispatcher decides how to handle this
                    if self.loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            self.output_dispatcher.dispatch(
                                "Typing..."
                            ),  # Send fixed message
                            self.loop,
                        )
                        self.logger.debug("已调度 'Typing...' 状态进行分发")
                        self._last_typing_send_time = (
                            current_time  # Update last send time
                        )
                    else:
                        self.logger.warning(
                            "事件循环未运行，无法调度 'Typing...' 状态分发。"
                        )
                # else: # Suppress frequent typing updates
                #    self.logger.debug("Typing... 状态更新已抑制 (过于频繁)")

            elif self.intermediate_behavior == "show_partial":
                # Paraformer doesn't really give partials like Gummy.
                # We could send the latest non-final sentence, but it might be confusing.
                # Paraformer doesn't provide granular partial results like Gummy.
                # We will send the latest non-final sentence as the partial result.
                log_prefix = "中间结果 (部分)"  # Indicate source
                self.logger.debug(f"{log_prefix}: {text_to_process}")
                # Dispatch the partial text
                if self.loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        self.output_dispatcher.dispatch(
                            text_to_process
                        ),  # Send actual partial text
                        self.loop,
                    )
                    self.logger.debug(
                        f"已调度 '{log_prefix}' 文本 '{text_to_process[:50]}...' 进行分发"
                    )
                else:
                    self.logger.warning(
                        f"事件循环未运行，无法调度 '{log_prefix}' 文本分发。"
                    )
                # Note: We don't use _last_typing_send_time here as we dispatch every partial result.

            # If behavior is "ignore", do nothing for intermediate results.


def create_paraformer_recognizer(
    main_loop: asyncio.AbstractEventLoop,
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
    # 注意：config 实例现在是从模块顶部导入的
    api_key = config["dashscope_api_key"]
    model = config["stt.model"]
    # 确保从 config 获取最新的 sample_rate (可能被 audio_recorder 更新)
    sample_rate = config["audio.sample_rate"]
    channels = config["audio.channels"]

    # Create the callback instance, passing dispatcher and llm client
    callback = ParaformerCallback(
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
