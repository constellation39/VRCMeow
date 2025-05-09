import yaml
import os
import logging
from typing import Dict, Any, Optional

# Use standard logging; configuration (level etc.) is handled by logger_config later
logger = logging.getLogger(__name__)

# Set a basic handler and level for the config module's logger initially.
# This ensures messages during config loading are visible even before
# the main application logging is fully configured.
# This basic configuration will be overridden/replaced by logger_config.setup_logging().
if not logger.hasHandlers():
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO) # Default to INFO for initial config loading messages
    # logger.propagate = False # Optional: Prevent messages duplicating if root logger is configured


DEFAULT_CONFIG_PATH = "config.yaml"

# --- Default Config (kept private for clarity) ---
_DEFAULT_CONFIG: Dict[str, Any] = {
    "dashscope_api_key": "",  # Strongly recommend setting via environment variable (DASHSCOPE_API_KEY)
    "stt": {
        # If set, enables translation (only for supported models like Gummy)
        "translation_target_language": None,  # e.g., "en", "ja", "ko"
        # Available models: "gummy-realtime-v1" (supports translation), "paraformer-realtime-v2", "paraformer-realtime-v1" (recognition only)
        "model": "gummy-realtime-v1",
        # How to handle intermediate (non-final) STT results
        # "ignore": Ignore intermediate results.
        # "show_typing": Send a fixed "Typing..." message (VRC OSC only).
        # "show_partial": Send the incomplete recognized text (VRC OSC only).
        "intermediate_result_behavior": "ignore",
    },
    "audio": {
        "sample_rate": None,  # None means use device default (e.g., 16000 for Gummy)
        "channels": 1,        # (e.g., 1 for Gummy)
        "dtype": "int16",     # (e.g., "int16" for Gummy)
        "debug_echo_mode": False,
    },
    "llm": {
        "enabled": False,
        "api_key": "", # Strongly recommend setting via environment variable (OPENAI_API_KEY)
        "base_url": None, # Optional: e.g., for local LLMs or proxies like "http://localhost:11434/v1"
        "model": "gpt-3.5-turbo", # Or any OpenAI compatible model name
        # "system_prompt_path": "prompts/default_system_prompt.txt", # REMOVED: Path is no longer used
        "system_prompt": "You are a helpful assistant.", # Default prompt if not set in config.yaml
        "temperature": 0.7, # Controls randomness (0.0 to 2.0)
        "max_tokens": 150, # Max response length
        # Few-shot examples: List of {"user": "...", "assistant": "..."} dictionaries
        "few_shot_examples": [],
    },
    "outputs": {
        "vrc_osc": {
            "enabled": True, # Enable/disable sending to VRChat OSC
            "address": "127.0.0.1",
            "port": 9000,
            "message_interval": 1.333, # Minimum interval between messages (seconds)
            "format": "{text}", # Format string for VRChat output ({text})
        },
        "console": {
            "enabled": True, # Print final results to console
            "prefix": "[Final Text]" # Optional prefix for console output
        },
        "file": {
            "enabled": False, # Append final results to a file
            "path": "output_log.txt", # Path to the output file
            "format": "{timestamp} - {text}", # Format string ({timestamp}, {text})
        }
        # Add more output types here later (e.g., websocket, http_post)
    },
    "logging": {
        "level": "INFO", # DEBUG, INFO, WARNING, ERROR, CRITICAL
    },
}

def _recursive_update(d: Dict, u: Dict) -> Dict:
    """Recursively update dict d with values from dict u."""
    for k, v in u.items():
        if isinstance(v, dict):
            # Ensure the key exists in d and is a dict before recursing
            d_k = d.get(k)
            if isinstance(d_k, dict):
                d[k] = _recursive_update(d_k, v)
            else: # If d[k] is not a dict (or doesn't exist), replace it
                d[k] = v
        else:
            d[k] = v
    return d

class Config:
    """Singleton class to load and hold application configuration."""
    _instance: Optional['Config'] = None
    _config_data: Dict[str, Any] = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            # Load config only once when the first instance is created
            cls._instance._load_config()
        return cls._instance

    def _load_config(self, config_path: str = DEFAULT_CONFIG_PATH) -> None:
        """Loads configuration from file and environment variables."""
        # Start with a deep copy of defaults to avoid modifying the original
        # Use recursive copy for nested dicts
        config = {}
        for k, v in _DEFAULT_CONFIG.items():
            if isinstance(v, dict):
                config[k] = v.copy() # Shallow copy for top-level dict is okay for defaults
            else:
                config[k] = v

        # 1. Load from file
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
                if file_config and isinstance(file_config, dict):
                    config = _recursive_update(config, file_config)
                    logger.info(f"Loaded configuration from {config_path}.")
                elif file_config: # Loaded something, but not a dict
                     logger.warning(f"Config file {config_path} does not contain a valid YAML dictionary. Using defaults.")
                else: # File is empty
                    logger.info(f"Config file {config_path} is empty. Using default configuration.")
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default configuration.")
            logger.info(f"Consider copying config.example.yaml to {config_path}.")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file {config_path}: {e}. Using default config.", exc_info=True)
        except Exception as e:
            logger.error(f"Unknown error loading config file {config_path}: {e}. Using default config.", exc_info=True)

        # 2. Environment variable override (API Key)
        env_api_key = os.getenv('DASHSCOPE_API_KEY')
        if env_api_key:
            # Ensure 'dashscope_api_key' exists at the top level
            config['dashscope_api_key'] = env_api_key
            logger.info("Overridden Dashscope API Key using DASHSCOPE_API_KEY environment variable.")
        elif not config.get('dashscope_api_key'):
            # Check if the key exists and is empty or if it doesn't exist at all
             logger.warning("Dashscope API Key not found in config file or DASHSCOPE_API_KEY environment variable.")

        # 2b. Environment variable override (LLM API Key)
        env_llm_api_key = os.getenv('OPENAI_API_KEY')
        if env_llm_api_key:
             # Ensure 'llm' dict exists before setting the key
            if "llm" not in config:
                config["llm"] = {}
            if not isinstance(config.get("llm"), dict): # Check if existing llm is dict
                 logger.warning("LLM config section is not a dictionary, cannot override API key. Check config.yaml structure.")
            else:
                 config['llm']['api_key'] = env_llm_api_key
                 logger.info("Overridden LLM API Key using OPENAI_API_KEY environment variable.")
        elif config.get("llm", {}).get("enabled") and not config.get("llm", {}).get("api_key"):
            # Check if LLM is enabled but the key is missing/empty
            if config.get("llm", {}).get("enabled") and not config.get("llm", {}).get("api_key"):
                logger.warning("LLM processing is enabled but API Key not found in config file or OPENAI_API_KEY environment variable.")


        # 3. Ensure LLM System Prompt exists (using config value or default)
        try:
            # Ensure llm structure exists
            if "llm" not in config or not isinstance(config.get("llm"), dict):
                config["llm"] = {} # Initialize if missing or wrong type

            # Get the default prompt from the _DEFAULT_CONFIG dictionary
            default_prompt = _DEFAULT_CONFIG.get("llm", {}).get("system_prompt", "You are a helpful assistant.")

            # Ensure 'system_prompt' key exists in the loaded config's llm section,
            # otherwise set it to the default.
            if "system_prompt" not in config.get("llm", {}):
                config["llm"]["system_prompt"] = default_prompt
                logger.info("LLM system prompt not found in config, using default.")
            else:
                # If it exists, log that we are using the one from the config file
                # (or the one potentially loaded from the file if that logic were still here)
                logger.debug("Using LLM system prompt from configuration.")

        except Exception as e:
             logger.error(f"Error processing LLM system prompt configuration: {e}. Using default prompt.", exc_info=True)
             # Ensure a safe fallback if structure was bad
             if "llm" not in config or not isinstance(config.get("llm"), dict):
                 config["llm"] = {}
             config["llm"]["system_prompt"] = _DEFAULT_CONFIG.get("llm", {}).get("system_prompt", "You are a helpful assistant.")


        # 4. Validate and transform (Log level)
        try:
            # Ensure logging dict structure exists before accessing nested keys
            if "logging" not in config or not isinstance(config.get("logging"), dict):
                 config["logging"] = {} # Initialize if missing or wrong type

            log_level_str = config.get("logging", {}).get("level", "INFO").upper()
            log_level = getattr(logging, log_level_str, logging.INFO)
            if not isinstance(log_level, int):
                logger.warning(f"Invalid log level '{log_level_str}', defaulting to INFO.")
                log_level = logging.INFO

            # Store both string and int level for convenience
            config["logging"]["level"] = logging.getLevelName(log_level) # Ensure string matches level
            config["logging"]["level_int"] = log_level
        except Exception as e:
            logger.error(f"Error processing logging configuration: {e}. Using default log level INFO.", exc_info=True)
            # Ensure defaults are set if error occurred
            if "logging" not in config:
                config["logging"] = {}
            config["logging"]["level"] = "INFO"
            config["logging"]["level_int"] = logging.INFO


        self._config_data = config
        logger.debug(f"Final loaded configuration: {self._config_data}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Access configuration values using dot notation or direct key access.
        Returns a default value if the key is not found.
        e.g., config.get('stt.model') or config.get('dashscope_api_key')
        """
        if '.' in key:
            keys = key.split('.')
            value = self._config_data
            try:
                for k in keys:
                    if isinstance(value, dict):
                        value = value[k]
                    else:
                        return default # Path is invalid if intermediate value is not a dict
                return value
            except KeyError:
                return default
        else:
            # Access top-level key directly
            return self._config_data.get(key, default)


    def __getitem__(self, key: str) -> Any:
        """
        Allow dictionary-style access, supporting dot notation for nested keys.
        Raises KeyError if the key is not found.
        e.g., config['stt.model'] or config['dashscope_api_key']
        """
        if '.' in key:
            keys = key.split('.')
            value = self._config_data
            try:
                for k in keys:
                     if isinstance(value, dict):
                         value = value[k]
                     else:
                         raise KeyError(f"Configuration path invalid at '{k}' for key '{key}'.")
                return value
            except KeyError:
                raise KeyError(f"Configuration key '{key}' not found.")
        else:
            # Access top-level key directly
            try:
                return self._config_data[key]
            except KeyError:
                 raise KeyError(f"Configuration key '{key}' not found.")

    @property
    def data(self) -> Dict[str, Any]:
        """Return the raw configuration dictionary (read-only recommended)."""
        # Returning direct reference for performance. Be cautious about modifying it externally.
        return self._config_data

    def reload(self) -> None:
        """Reloads the configuration."""
        logger.info("Reloading configuration...")
        self._load_config()


# Create the singleton instance upon module import
# This ensures config is loaded once when the application starts
try:
    config = Config()
except Exception as e:
    # Critical error during initial config load
    logger.critical(f"CRITICAL ERROR during initial configuration load: {e}", exc_info=True)
    logger.critical("Application might not function correctly. Attempting to use fallback defaults.")
    # Attempt to provide a minimal fallback using _DEFAULT_CONFIG directly
    # Note: This won't have env vars or log level conversion applied correctly
    class FallbackConfig:
        _data = {k: v.copy() if isinstance(v, dict) else v for k, v in _DEFAULT_CONFIG.items()}
        # Ensure minimal structure exists
        if "logging" not in _data:
            _data["logging"] = {}
        _data["logging"]["level"] = _DEFAULT_CONFIG.get("logging", {}).get("level", "INFO")
        _data["logging"]["level_int"] = getattr(logging, _data["logging"]["level"], logging.INFO)
        # Ensure llm section exists (even in fallback, though keys won't be overridden)
        if "llm" not in _data:
            _data["llm"] = {}

        @property
        def data(self):
            return self._data
        def __getitem__(self, key): # Basic non-nested access
            try:
                return self._data[key]
            except KeyError:
                raise KeyError(f"Fallback config missing key '{key}'")
        def get(self, key, default=None): # Basic non-nested get
            return self._data.get(key, default)
    config = FallbackConfig()
    # The warning is logged when the instance is created, not part of the class definition itself
    logger.warning("Using fallback configuration due to critical error during load.")


# Optional: Provide a function alias for compatibility or preference
def get_config_data() -> Dict[str, Any]:
    """Returns the loaded configuration data dictionary."""
    return config.data

if __name__ == '__main__':
    print("Testing Config class...")
    # Accessing the singleton instance multiple times yields the same object
    c1 = Config()
    c2 = Config()
    print(f"Is c1 the same instance as c2? {c1 is c2}")
    # Check if config is the singleton or the fallback
    if isinstance(config, Config):
         print(f"Is c1 the same instance as the module-level 'config' instance? {c1 is config}")
    else:
         print("Module-level 'config' is a FallbackConfig instance.")


    print("\nLoaded Configuration:")
    import json
    # Use the get_config_data helper or access via instance.data
    print(json.dumps(get_config_data(), indent=2, ensure_ascii=False))

    # Test accessing values using different methods
    try:
        print(f"\nLogging Level (dict access): {config['logging']['level']}") # Direct dict access
        print(f"Logging Level (dot notation __getitem__): {config['logging.level']}") # Dot notation via __getitem__
        print(f"STT Model (direct instance data): {config.data['stt']['model']}") # Access raw data
        print(f"OSC Address (get method): {config.get('vrc_osc.address', 'default_ip')}") # Dot notation via get()
        print(f"OSC Port (get method): {config.get('vrc_osc.port', 9999)}")
        print(f"API Key (get method): {config.get('dashscope_api_key', 'NOT_SET')}") # Top-level get

        # Test non-existent key with get
        print(f"Non-existent key (get): {config.get('invalid.key', 'MISSING')}")

        # Test non-existent key with __getitem__
        print(f"Non-existent key (__getitem__): {config['non_existent.key']}")

    except KeyError as e:
        print(f"Caught KeyError: {e}")
    except Exception as e:
         print(f"Caught unexpected error during testing: {e}")


    # Test reload (optional)
    # print("\nSimulating config file change and reloading...")
    # # In a real scenario, you might modify config.yaml here
    # config.reload()
    # print("Configuration reloaded.")
    # print(json.dumps(config.data, indent=2, ensure_ascii=False))
