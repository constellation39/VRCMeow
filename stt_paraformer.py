import asyncio
from typing import Optional
from dashscope.audio.asr import (
    RecognitionCallback,
    Recognition,
    RecognitionResult
)
from logger_config import get_logger
# 直接从 config 模块导入 config 实例
from config import config

# Import VRCClient for type hinting and usage in callback
try:
    from osc_client import VRCClient
except ImportError:
    VRCClient = None  # Define as None if import fails


# --- Callback for Paraformer API (Recognition only) ---
class ParaformerCallback(RecognitionCallback):
    """处理 Dashscope Paraformer API (仅识别) 事件的回调类。"""

    def __init__(self,
                 loop: asyncio.AbstractEventLoop,
                 vrc_client: Optional[VRCClient]):
        super().__init__()
        self.loop = loop
        self.vrc_client = vrc_client
        self.logger = get_logger(f"{__name__}.ParaformerCallback")
        self.logger.info("ParaformerCallback 初始化完成。")

    def on_open(self) -> None:
        self.logger.info("Dashscope Paraformer 连接已打开。")

    def on_close(self) -> None:
        self.logger.info("Dashscope Paraformer 连接已关闭。")

    def on_complete(self) -> None:
        # Paraformer specific event
        self.logger.info("Dashscope Paraformer 识别完成。")

    def on_error(self, message) -> None:
        # Paraformer error structure might contain request_id and message
        error_msg = f"Msg={message.message}" if hasattr(message, 'message') else str(message)
        request_id = f"ID={message.request_id}" if hasattr(message, 'request_id') else ""
        self.logger.error(f"Dashscope Paraformer 错误: {request_id} {error_msg}")
        # Consider triggering stop_event here if needed
        # self.loop.call_soon_threadsafe(stop_event.set) # Requires passing stop_event

    def on_event(self, result: RecognitionResult) -> None:
        # Process Paraformer's RecognitionResult
        sentence = result.get_sentence()
        request_id = result.get_request_id()
        usage = result.get_usage(sentence)  # Get usage info
        self.logger.debug(f"Dashscope Paraformer 事件: ID={request_id}, Usage={usage}")

        text_to_send = None
        if sentence and 'text' in sentence:
            text_to_send = sentence['text']
            # Log the recognized text
            self.logger.info(f"识别: {text_to_send}")

        # Send OSC message (similar to GummyCallback)
        if text_to_send:
            if self.vrc_client:
                if self.loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        self.vrc_client.send_chatbox(text_to_send),
                        self.loop
                    )
                    self.logger.debug(f"已调度 OSC 消息: {text_to_send}")
                else:
                    self.logger.warning("事件循环未运行，无法发送 OSC 消息。")
            else:
                self.logger.debug("VRC 客户端不可用，跳过 OSC 发送。")


def create_paraformer_recognizer(
        main_loop: asyncio.AbstractEventLoop,
        vrc_client: Optional[VRCClient]
) -> Recognition:
    """创建并配置 Dashscope Paraformer 实时识别器。"""
    logger = get_logger(__name__)  # Use logger from this module
    logger.info("使用 Paraformer API (仅识别)")

    # --- 直接从 config 实例获取配置 ---
    # 注意：config 实例现在是从模块顶部导入的
    api_key = config['dashscope_api_key']
    model = config['stt.model']
    # 确保从 config 获取最新的 sample_rate (可能被 audio_recorder 更新)
    sample_rate = config['audio.sample_rate']
    channels = config['audio.channels']

    # Create the callback instance
    callback = ParaformerCallback(loop=main_loop, vrc_client=vrc_client)

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

    # Log initialization details
    logger.info(f"Dashscope Paraformer Recognizer (模型: {model}) 初始化完成。")
    logger.info(f"  - 采样率: {sample_rate}, 声道: {channels}")
    logger.info("  - 翻译: 不支持")

    return recognizer
