import datetime

# Directly import the config instance
from typing import Optional  # 导入 Callable 和 Awaitable

import pathlib # Import pathlib

# Use aiofiles for async file operations
import aiofiles

# Directly import the config instance and APP_DIR
try:
    from config import config, APP_DIR
except ImportError:
    # Fallback or error handling if config module/APP_DIR isn't available
    # Get logger instance first (assuming logger_config might partially work)
    logger = get_logger(__name__) # Use get_logger from logger_config
    logger.critical("Could not import config or APP_DIR. Output file path cannot be reliably determined.")
    # Define a fallback APP_DIR or handle error
    APP_DIR = pathlib.Path.cwd() # Fallback to CWD
    logger.warning(f"Falling back to CWD for output file path resolution: {APP_DIR}")
    # Attempt to import just config if APP_DIR failed
    try:
        from config import config
    except ImportError:
        config = None # Indicate config unavailable
        logger.error("Config instance unavailable for OutputDispatcher.")


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
        self.outputs_config = config.get("outputs", {}) if config else {} # Handle config being None

        # --- Validate File Output Config ---
        file_config = self.outputs_config.get("file", {})
        self.file_output_enabled = file_config.get("enabled", False)
        self.file_path_str = file_config.get("path", "vrcmeow_output.log") # Default filename
        self.file_format = file_config.get("format", "{timestamp} - {text}")
        self.resolved_file_path: Optional[pathlib.Path] = None # Store resolved Path object

        if self.file_output_enabled:
            # Resolve the file path: if relative, make it relative to APP_DIR
            file_path_obj = pathlib.Path(self.file_path_str)
            if not file_path_obj.is_absolute():
                 # Ensure APP_DIR is available before resolving
                if 'APP_DIR' not in globals():
                     logger.error("APP_DIR not available, cannot resolve relative output file path. Disabling file output.")
                     self.file_output_enabled = False
                else:
                    self.resolved_file_path = APP_DIR / file_path_obj
                    logger.debug(f"Resolved relative output file path to: {self.resolved_file_path}")
            else:
                self.resolved_file_path = file_path_obj
                logger.debug(f"Using absolute output file path: {self.resolved_file_path}")

            # Ensure the target directory exists before first write attempt (only if path resolved)
            if self.resolved_file_path:
                try:
                    self.resolved_file_path.parent.mkdir(parents=True, exist_ok=True)
                    logger.info(f"File output enabled. Appending results to: {self.resolved_file_path}")
                except OSError as e:
                    logger.error(f"Failed to create directory for output file {self.resolved_file_path}: {e}. Disabling file output.", exc_info=True)
                    self.file_output_enabled = False # Disable if directory fails
                except Exception as e: # Catch other potential errors with the path
                     logger.error(f"Invalid output file path configured: {self.resolved_file_path}. Disabling file output. Error: {e}", exc_info=True)
                     self.file_output_enabled = False # Disable on invalid path
            else:
                 # If path didn't resolve (e.g., APP_DIR missing), disable file output
                 self.file_output_enabled = False

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
            logger.info(
                f"VRC OSC output enabled per config. Format: '{self.vrc_osc_format}'"
            )

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
        # Check enabled status AND if resolved_file_path is valid
        if self.file_output_enabled and self.resolved_file_path:
            try:
                # DEBUG: Log before awaiting file write
                logger.debug(f"OUTPUT_DISP: Awaiting _write_to_file for text: '{text}' to path: {self.resolved_file_path}")
                await self._write_to_file(text)
                # DEBUG: Log after successful file write
                logger.debug(f"OUTPUT_DISP: Finished awaiting _write_to_file to {self.resolved_file_path}")
            except Exception as file_err:
                # Log error directly here, no need to gather exceptions later
                logger.error(
                    f"OutputDispatcher: Error during file write task: {file_err}",
                    exc_info=file_err,
                )

        # 3. VRC OSC Output (Async - Await directly)
        # DEBUG: Check if VRC OSC dispatch will be attempted
        logger.debug(
            f"OUTPUT_DISP: Checking VRC OSC condition. Enabled: {self.vrc_osc_enabled}, Client valid: {bool(self.vrc_client)}"
        )
        if self.vrc_osc_enabled and self.vrc_client:
            logger.info(
                "OUTPUT_DISP: VRC OSC condition met. Proceeding with formatting and sending."
            )  # Log entry into VRC block
            formatted_vrc_text = text  # Default to raw text
            try:
                # DEBUG: Log the format string being used
                logger.debug(
                    f"OUTPUT_DISP: Formatting VRC text using format: '{self.vrc_osc_format}'"
                )
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
                formatted_vrc_text = text  # Ensure fallback is assigned on error

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
            # Ensure resolved_file_path is not None before using it
            if not self.resolved_file_path:
                 logger.error("OutputDispatcher: Cannot write to file, resolved path is missing.")
                 return # Or raise an error

            # Ensure directory exists again just before writing (optional, but safe)
            try:
                self.resolved_file_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                 logger.error(f"OutputDispatcher: Failed to ensure directory exists for {self.resolved_file_path}: {e}", exc_info=True)
                 raise # Re-raise directory creation error

            async with aiofiles.open(self.resolved_file_path, mode="a", encoding="utf-8") as f:
                await f.write(formatted_line)
            logger.debug(f"Appended text to file: {self.resolved_file_path}")
        except Exception as e:
            logger.error(
                f"OutputDispatcher: Failed to write to file {self.resolved_file_path}: {e}",
                exc_info=True,
            )
            raise  # Re-raise to allow calling code to handle
