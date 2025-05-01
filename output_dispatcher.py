import asyncio
import datetime

# Directly import the config instance
from typing import Optional, Callable  # 导入 Callable 和 Awaitable

import aiofiles  # Use aiofiles for async file operations

# Directly import the config instance
from config import config
from logger_config import get_logger

# Import VRCClient for type hinting, handle potential ImportError if needed
from osc_client import VRCClient

logger = get_logger(__name__)


class OutputDispatcher:
    """Handles dispatching final text results to configured outputs."""

    def __init__(
        self,
        vrc_client_instance: Optional[VRCClient] = None,
        # REMOVED: gui_output_callback parameter
    ) -> None:
        logger.info("OutputDispatcher initializing...")
        self.vrc_client = vrc_client_instance
        # REMOVED: self.gui_output_callback = gui_output_callback
        self.outputs_config = config.get("outputs", {})

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
            # Log a warning if enabled but client missing *at init time*, but DON'T disable it permanently.
            logger.warning(
                "OutputDispatcher: VRC OSC output is enabled in config, but VRCClient instance was not provided during initialization. Dispatch will depend on client being assigned later."
            )
            # REMOVED: self.vrc_osc_enabled = False
        elif self.vrc_osc_enabled:
            # Log that it's enabled based on config, client check happens at dispatch time.
            logger.info(f"VRC OSC output enabled per config. Format: '{self.vrc_osc_format}'")

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
        # REMOVED: GUI output check

        if enabled_outputs:
            logger.info(
                f"OUTPUT_DISP: Enabled outputs for this dispatch: {', '.join(enabled_outputs)}"
            )
        else:
            logger.warning("OUTPUT_DISP: No outputs enabled for dispatch.")
            return  # Exit early if nothing to do

        # 1. Console Output (Synchronous)
        if self.console_output_enabled:
            # Console print is generally fast enough, run directly
            # Use print for direct console output, logger for internal app logging
            print(f"{self.console_prefix} {text}")

        # 2. File Output (Async - Await directly)
        if self.file_output_enabled:
            try:
                # DEBUG: Log before awaiting file write
                logger.debug(f"OUTPUT_DISP: Awaiting _write_to_file for text: '{text}'")
                await self._write_to_file(text)
                # DEBUG: Log after successful file write
                logger.debug("OUTPUT_DISP: Finished awaiting _write_to_file")
            except Exception as file_err:
                # Log error directly here, no need to gather exceptions later
                logger.error(
                    f"OutputDispatcher: Error during file write task: {file_err}",
                    exc_info=file_err,
                )

        # 3. VRC OSC Output (Async - Await directly)
        # DEBUG: Check if VRC OSC dispatch will be attempted
        logger.debug(f"OUTPUT_DISP: Checking VRC OSC condition. Enabled: {self.vrc_osc_enabled}, Client valid: {bool(self.vrc_client)}")
        if self.vrc_osc_enabled and self.vrc_client:
            logger.info("OUTPUT_DISP: VRC OSC condition met. Proceeding with formatting and sending.") # Log entry into VRC block
            formatted_vrc_text = text  # Default to raw text
            try:
                # DEBUG: Log the format string being used
                logger.debug(f"OUTPUT_DISP: Formatting VRC text using format: '{self.vrc_osc_format}'")
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
                formatted_vrc_text = text # Ensure fallback is assigned on error

            try:
                # INFO: Log before calling VRCClient.send_chatbox
                logger.info(
                    f"OUTPUT_DISP: Attempting to call VRCClient.send_chatbox with formatted text: '{formatted_vrc_text}'"
                )
                await self.vrc_client.send_chatbox(formatted_vrc_text)
                # DEBUG: Log after successful send
                logger.debug("OUTPUT_DISP: Finished awaiting VRCClient.send_chatbox")
            except Exception as osc_err:
                # Log error directly here
                logger.error(
                    f"OutputDispatcher: Error during VRC OSC send task: {osc_err}",
                    exc_info=osc_err,
                )

        # REMOVED: GUI Output section

        # No more asyncio.gather needed as operations are awaited directly above.
        logger.info(f"OUTPUT_DISP: Exiting dispatch method for text: '{text}'")

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
