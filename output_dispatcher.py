import asyncio
import datetime
from typing import Optional
import aiofiles  # Use aiofiles for async file operations

# Directly import the config instance
from config import config
from logger_config import get_logger

# Import VRCClient for type hinting, handle potential ImportError if needed
from osc_client import VRCClient

logger = get_logger(__name__)


class OutputDispatcher:
    """Handles dispatching final text results to configured outputs."""

    def __init__(self, vrc_client_instance: Optional[VRCClient] = None) -> None:
        self.vrc_client = vrc_client_instance
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
        if not text:
            logger.debug("OutputDispatcher: Received empty text, nothing to dispatch.")
            return

        logger.debug(f"Dispatching text: '{text[:50]}...'")

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
            dispatch_tasks.append(
                asyncio.create_task(self.vrc_client.send_chatbox(formatted_vrc_text))
            )

        # Wait for async tasks like file writing or OSC sending to complete
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
