import asyncio
import datetime
# Directly import the config instance
from typing import Optional, Callable  # 导入 Callable 和 Awaitable

import aiofiles  # Use aiofiles for async file operations

# Directly import the config instance
from config import config
from logger_config import get_logger
import logging  # 导入标准 logging 库
# Import VRCClient for type hinting, handle potential ImportError if needed
from osc_client import VRCClient

logger = get_logger(__name__)

# --- 临时诊断代码 ---
# 确保此 logger 实例配置了 handler 并设置了低级别，以便输出日志
if not logger.handlers and not logging.getLogger().hasHandlers(): # 检查自身和根 logger 是否都没有 handlers
    print(f"DIAGNOSTIC [{__name__}]: Logger '{logger.name}' and root logger have no handlers. Adding StreamHandler to '{logger.name}'.")
    _handler = logging.StreamHandler(sys.stdout) # 创建一个流处理器（输出到标准输出）
    _formatter = logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s') # 改进格式
    _handler.setFormatter(_formatter) # 设置格式
    logger.addHandler(_handler) # 添加处理器
    logger.setLevel(logging.DEBUG) # 设置级别为 DEBUG
    logger.propagate = False # 阻止日志消息传递给父 logger，避免重复输出
    logger.warning(f"DIAGNOSTIC [{__name__}]: Temporary StreamHandler added and level set to DEBUG for '{logger.name}'.") # 使用 warning 以便更容易看到
elif not logger.handlers and logger.parent and logger.parent.handlers:
     print(f"DIAGNOSTIC [{__name__}]: Logger '{logger.name}' has no handlers, but parent '{logger.parent.name}' does. Ensuring level is DEBUG.")
     logger.setLevel(logging.DEBUG) # 确保子 logger 级别足够低以允许消息传递
     logger.warning(f"DIAGNOSTIC [{__name__}]: Ensured logger level is DEBUG for '{logger.name}' to allow propagation.")
else:
    # 如果已有 handlers，或者根 logger 有 handlers，确保此 logger 的级别足够低
    current_level = logger.getEffectiveLevel() # 获取实际生效的级别
    print(f"DIAGNOSTIC [{__name__}]: Logger '{logger.name}' already has handlers or root has handlers. Effective level: {logging.getLevelName(current_level)}. Setting level to DEBUG.")
    logger.setLevel(logging.DEBUG)
    logger.warning(f"DIAGNOSTIC [{__name__}]: Ensured logger level is DEBUG for '{logger.name}'.")
# --- 临时诊断代码结束 ---


class OutputDispatcher:
    """Handles dispatching final text results to configured outputs."""

    def __init__(
        self,
        vrc_client_instance: Optional[VRCClient] = None,
        gui_output_callback: Optional[Callable[[str], None]] = None # 添加 GUI 回调参数
    ) -> None:
        self.vrc_client = vrc_client_instance
        self.gui_output_callback = gui_output_callback # 存储回调
        self.outputs_config = config.get("outputs", {})
        self.loop = asyncio.get_running_loop()

        # --- Validate File Output Config ---
        self.file_output_enabled = self.outputs_config.get("file", {}).get(
            "enabled", False
        )
        self.file_path = self.outputs_config.get("file", {}).get(
            "path", "output_log.txt"
        )
        self.file_format = self.outputs_config.get("file", {}).get(
            "format", "{timestamp} - {text}"
        )

        if self.file_output_enabled:
            logger.info(f"File output enabled. Appending results to: {self.file_path}")
            # Optional: Could create the directory if it doesn't exist here,
            # but aiofiles.open handles file creation.

        # --- Console Output Config ---
        self.console_output_enabled = self.outputs_config.get("console", {}).get(
            "enabled", True
        )
        self.console_prefix = self.outputs_config.get("console", {}).get(
            "prefix", "[Final Text]"
        )

        if self.console_output_enabled:
            logger.info("Console output enabled.")

        # --- VRC OSC Output Config ---
        self.vrc_osc_enabled = self.outputs_config.get("vrc_osc", {}).get(
            "enabled", True
        )
        self.vrc_osc_format = self.outputs_config.get("vrc_osc", {}).get(
            "format", "{text}"
        )  # Default to just text

        if self.vrc_osc_enabled and not self.vrc_client:
            logger.warning(
                "OutputDispatcher: VRC OSC output is enabled, but VRCClient instance was not provided. OSC output will be skipped."
            )
            self.vrc_osc_enabled = False  # Disable if client missing
        elif self.vrc_osc_enabled:
            logger.info(f"VRC OSC output enabled. Format: '{self.vrc_osc_format}'")

    async def dispatch(self, text: str):
        """
        Sends the final text to all enabled output destinations.

        Args:
            text: The final text string to dispatch (original or LLM-processed).
        """
        # INFO: Log entry point of dispatch to confirm execution start
        logger.info(f"OUTPUT_DISP: Entering dispatch method with text: '{text}'")

        if not text:
            logger.debug("OUTPUT_DISP: Received empty text, nothing to dispatch.")
            return

        # INFO: Log when dispatch starts for a non-empty text (now redundant with above, but keep for flow)
        logger.info(f"OUTPUT_DISP: Starting dispatch for text: '{text}'")

        # Determine and log enabled outputs
        enabled_outputs = []
        if self.console_output_enabled:
            enabled_outputs.append("Console")
        if self.file_output_enabled:
            enabled_outputs.append(f"File ({self.file_path})")
        if self.vrc_osc_enabled:
            enabled_outputs.append("VRC OSC")
        if self.gui_output_callback:
            enabled_outputs.append("GUI")

        if enabled_outputs:
            logger.info(f"OUTPUT_DISP: Enabled outputs for this dispatch: {', '.join(enabled_outputs)}")
        else:
            logger.warning("OUTPUT_DISP: No outputs enabled for dispatch.")


        dispatch_tasks = []

        # 1. Console Output
        if self.console_output_enabled:
            # Console print is generally fast enough, run directly
            # Use print for direct console output, logger for internal app logging
            print(f"{self.console_prefix} {text}")

        # 2. File Output (Async)
        if self.file_output_enabled:
            dispatch_tasks.append(asyncio.create_task(self._write_to_file(text)))

        # 3. VRC OSC Output (Async)
        if self.vrc_osc_enabled and self.vrc_client:
            # Format the text according to the configured format string
            try:
                formatted_vrc_text = self.vrc_osc_format.format(text=text)
            except KeyError as e:
                logger.warning(
                    f"OutputDispatcher: Invalid key '{e}' in vrc_osc format string '{self.vrc_osc_format}'. Sending raw text."
                )
                formatted_vrc_text = text  # Fallback to raw text
            except Exception as e:
                logger.error(
                    f"OutputDispatcher: Error formatting VRC OSC text: {e}. Sending raw text.",
                    exc_info=True,
                )
                formatted_vrc_text = text  # Fallback to raw text

            # VRCClient's send_chatbox should be async
            # INFO: Log before calling VRCClient.send_chatbox
            logger.info(f"OUTPUT_DISP: Calling VRCClient.send_chatbox with: '{formatted_vrc_text}'")
            dispatch_tasks.append(
                asyncio.create_task(self.vrc_client.send_chatbox(formatted_vrc_text))
            )

        # 4. GUI Output (Direct Call - Callback handles thread safety)
        if self.gui_output_callback:
            try:
                # 直接调用回调，假设回调是线程安全的 (例如，使用 page.run_thread_safe)
                self.gui_output_callback(text)
                logger.debug("Dispatched text to GUI.")
            except Exception as gui_err:
                 logger.error(f"OutputDispatcher: Error calling GUI output callback: {gui_err}", exc_info=True)
                 # 不将此添加到 dispatch_tasks，因为回调应快速返回

        # Wait for background async tasks like file writing or OSC sending to complete
        if dispatch_tasks:
            try:
                # Gather results, logging any exceptions that occurred in tasks
                # Wait for all tasks and capture exceptions
                results = await asyncio.gather(*dispatch_tasks, return_exceptions=True)
                for task, result in zip(dispatch_tasks, results):
                    if isinstance(result, Exception):
                        # Log the exception from the failed task
                        # Attempt to get a meaningful name from the task object (coro repr)
                        task_name = (
                            task.get_coro().__qualname__
                            if hasattr(task, "get_coro")
                            else "Unknown Task"
                        )
                        logger.error(
                            f"OutputDispatcher: Error in dispatch task '{task_name}': {result}",
                            exc_info=result,
                        )  # Pass exception for traceback
            except Exception as e:
                # This catches errors in the gather itself, though less common
                logger.error(
                    f"OutputDispatcher: Unexpected error during asyncio.gather for dispatch tasks: {e}",
                    exc_info=True,
                )

    async def _write_to_file(self, text: str):
        """Appends text asynchronously to the configured log file."""
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted_line = (
                self.file_format.format(timestamp=timestamp, text=text) + "\n"
            )

            async with aiofiles.open(self.file_path, mode="a", encoding="utf-8") as f:
                await f.write(formatted_line)
            logger.debug(f"Appended text to file: {self.file_path}")
        except Exception as e:
            logger.error(
                f"OutputDispatcher: Failed to write to file {self.file_path}: {e}",
                exc_info=True,
            )
            raise  # Re-raise to allow gather to catch it


# Example Usage
async def _test_dispatcher():
    from logger_config import setup_logging

    setup_logging()

    # Mock VRCClient for testing
    class MockVRCClient:
        async def send_chatbox(self, content):
            print(f"[Mock VRC Client] Received: {content}")
            await asyncio.sleep(0.1)  # Simulate async work
            # Simulate an error occasionally
            # if "error" in content.lower():
            #    raise ConnectionRefusedError("Simulated OSC connection error")

    # Ensure you have a config.yaml with outputs configured
    # Example: enable console, file, and vrc_osc
    mock_client = MockVRCClient()
    dispatcher = OutputDispatcher(vrc_client_instance=mock_client)

    print("\nTesting Dispatcher...")
    await dispatcher.dispatch("This is the first test message.")
    await dispatcher.dispatch("你好，这是第二条消息。")
    # await dispatcher.dispatch("This message might cause an error.") # Uncomment to test error handling
    print("Dispatcher test complete.")


if __name__ == '__main__':
    print("Running OutputDispatcher test...")
    asyncio.run(_test_dispatcher())
