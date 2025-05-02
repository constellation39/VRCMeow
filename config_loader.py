import logging
from typing import Any, Dict

# Attempt to import from the new config module
try:
    # Import the data access function
    # We no longer import 'config' here as it's unused in this compatibility layer
    from config import get_config_data as _get_config_data

    # Optionally, configure a logger for this compatibility layer if needed
    logger = logging.getLogger(__name__)
    logger.info("config_loader.py: Successfully imported from the new 'config' module.")
except ImportError as e:
    # Log a critical error if the new module cannot be imported
    logger = logging.getLogger(__name__)  # Get a logger instance anyway
    logger.critical(
        f"config_loader.py: FATAL - Failed to import from 'config' module: {e}",
        exc_info=True,
    )
    logger.critical(
        "Configuration system failed. Application will likely crash or malfunction."
    )

    # Define a dummy function to prevent import errors elsewhere, but the app is broken
    def get_config() -> Dict[str, Any]:
        """Dummy function returning an empty dict due to critical import failure."""
        print(
            "CRITICAL ERROR: Configuration system failed to load."
        )  # Also print to stderr/stdout
        return {}

    # Depending on the desired failure mode, you might want to:
    # raise RuntimeError("Configuration system failed to initialize") from e
    # or sys.exit(1)
else:
    # If import succeeded, define the compatibility function
    def get_config() -> Dict[str, Any]:
        """
        Provides the application configuration dictionary.

        [DEPRECATED] Prefer importing 'config' instance or 'get_config_data'
                     directly from the 'config' module for new code.
                     This function provides backward compatibility.
        """
        # Return the data dictionary from the singleton config instance in config.py
        return _get_config_data()
