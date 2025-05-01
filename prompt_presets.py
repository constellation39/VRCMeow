import json
import os
import pathlib
import logging
from typing import Dict, Any, List, Optional, Tuple

# Directly import the config instance and default config for fallback values
# Need to handle potential circular import or delayed import if config uses this module
try:
    from config import config, _DEFAULT_CONFIG
except ImportError:
    # Fallback if config cannot be imported directly (e.g., during initial setup)
    # This might happen if config.py itself tries to import this module too early.
    # In this case, get_default_preset_values might rely on a hardcoded default.
    config = None
    _DEFAULT_CONFIG = { # Hardcoded minimal default for fallback
        "llm": {
            "system_prompt": "You are a helpful assistant.",
            "few_shot_examples": []
        }
    }
    logging.warning("Could not import 'config' module in prompt_presets. Using fallback defaults.")


logger = logging.getLogger(__name__)

# Determine paths relative to the Current Working Directory (CWD)
CWD = pathlib.Path.cwd()
PRESETS_FILENAME = "prompt_presets.json"
PRESETS_PATH = CWD / PRESETS_FILENAME

# --- Preset Data Structure ---
# {
#   "preset_name": {
#     "system_prompt": "...",
#     "few_shot_examples": [{"user": "...", "assistant": "..."}, ...]
#   },
#   ...
# }
# ---

def load_presets() -> Dict[str, Dict[str, Any]]:
    """Loads presets from the JSON file."""
    if not PRESETS_PATH.exists():
        logger.info(f"Presets file '{PRESETS_PATH}' not found. Returning empty presets.")
        return {}
    try:
        with open(PRESETS_PATH, "r", encoding="utf-8") as f:
            presets_data = json.load(f)
            if not isinstance(presets_data, dict):
                logger.warning(f"Presets file '{PRESETS_PATH}' does not contain a valid JSON dictionary. Returning empty presets.")
                return {}
            # Basic validation of structure could be added here if needed
            logger.info(f"Successfully loaded presets from '{PRESETS_PATH}'.")
            return presets_data
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from presets file '{PRESETS_PATH}'. Returning empty presets.", exc_info=True)
        return {}
    except Exception as e:
        logger.error(f"Error loading presets file '{PRESETS_PATH}': {e}", exc_info=True)
        return {}

def save_presets(presets_data: Dict[str, Dict[str, Any]]) -> bool:
    """Saves the presets dictionary to the JSON file."""
    try:
        # Ensure the directory exists
        PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PRESETS_PATH, "w", encoding="utf-8") as f:
            json.dump(presets_data, f, ensure_ascii=False, indent=2) # Use indent for readability
        logger.info(f"Presets successfully saved to '{PRESETS_PATH}'.")
        return True
    except Exception as e:
        logger.error(f"Failed to save presets to '{PRESETS_PATH}': {e}", exc_info=True)
        return False

def get_preset(preset_name: str) -> Optional[Dict[str, Any]]:
    """Loads all presets and returns the specified one, or None if not found."""
    presets = load_presets()
    preset_data = presets.get(preset_name)
    if preset_data:
        logger.debug(f"Retrieved preset '{preset_name}'.")
        # Ensure expected keys exist, provide defaults if missing
        if "system_prompt" not in preset_data:
            preset_data["system_prompt"] = "" # Default empty string
            logger.warning(f"Preset '{preset_name}' missing 'system_prompt', using empty string.")
        if "few_shot_examples" not in preset_data:
            preset_data["few_shot_examples"] = [] # Default empty list
            logger.warning(f"Preset '{preset_name}' missing 'few_shot_examples', using empty list.")
        # Validate few_shot_examples structure
        if not isinstance(preset_data["few_shot_examples"], list):
             logger.warning(f"Preset '{preset_name}' has invalid 'few_shot_examples' (not a list). Using empty list.")
             preset_data["few_shot_examples"] = []
        else:
            valid_examples = []
            for i, example in enumerate(preset_data["few_shot_examples"]):
                if isinstance(example, dict) and "user" in example and "assistant" in example:
                    valid_examples.append(example)
                else:
                    logger.warning(f"Invalid structure for few_shot_example at index {i} in preset '{preset_name}'. Skipping.")
            preset_data["few_shot_examples"] = valid_examples

        return preset_data
    else:
        logger.warning(f"Preset '{preset_name}' not found in loaded presets.")
        return None

def save_preset(preset_name: str, system_prompt: str, few_shot_examples: List[Dict[str, str]]) -> bool:
    """Saves or updates a single preset."""
    if not preset_name or not preset_name.strip():
        logger.error("Cannot save preset with an empty name.")
        return False

    presets = load_presets()
    presets[preset_name.strip()] = { # Ensure name is stripped
        "system_prompt": system_prompt,
        "few_shot_examples": few_shot_examples,
    }
    logger.info(f"Saving preset '{preset_name.strip()}'...")
    return save_presets(presets)

def delete_preset(preset_name: str) -> bool:
    """Deletes a single preset."""
    if preset_name == "Default":
        logger.error("Cannot delete the 'Default' preset.")
        return False

    presets = load_presets()
    if preset_name in presets:
        del presets[preset_name]
        logger.info(f"Deleting preset '{preset_name}'...")
        return save_presets(presets)
    else:
        logger.warning(f"Preset '{preset_name}' not found, cannot delete.")
        return False

def get_default_preset_values() -> Tuple[str, List[Dict[str, str]]]:
    """Gets the default system prompt and few-shot examples from _DEFAULT_CONFIG."""
    # Use the imported _DEFAULT_CONFIG (might be fallback)
    default_sys_prompt = _DEFAULT_CONFIG.get("llm", {}).get(
        "system_prompt", "You are a helpful assistant."
    )
    default_examples = _DEFAULT_CONFIG.get("llm", {}).get(
        "few_shot_examples", []
    )
    # Ensure examples are a list of dicts (basic check)
    if not isinstance(default_examples, list) or not all(isinstance(ex, dict) for ex in default_examples):
        logger.warning("Default few_shot_examples in _DEFAULT_CONFIG is invalid, using empty list.")
        default_examples = []

    return default_sys_prompt, default_examples


def ensure_default_preset() -> None:
    """Checks if the 'Default' preset exists, creates it from _DEFAULT_CONFIG if not."""
    presets = load_presets()
    if "Default" not in presets:
        logger.info("Preset 'Default' not found. Creating it from default configuration...")
        default_prompt, default_examples = get_default_preset_values()
        presets["Default"] = {
            "system_prompt": default_prompt,
            "few_shot_examples": default_examples,
        }
        if save_presets(presets):
            logger.info("Successfully created and saved 'Default' preset.")
        else:
            logger.error("Failed to save the newly created 'Default' preset.")

# --- Initial Check ---
# Call ensure_default_preset() from gui.py after config load and logging setup,
# not directly on module import, to avoid potential initialization order issues.
