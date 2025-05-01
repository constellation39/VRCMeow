import yaml
import os
import logging
from typing import Dict, Any, Optional
import copy  # Import copy for deep copying config data
import pathlib  # Import pathlib

# Use standard logging; configuration (level etc.) is handled by logger_config later
logger = logging.getLogger(__name__)

# Set a basic handler and level for the config module's logger initially.
# This ensures messages during config loading are visible even before
# the main application logging is fully configured.
# This basic configuration will be overridden/replaced by logger_config.setup_logging().
if not logger.hasHandlers():
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)  # Default to INFO for initial config loading messages
    # logger.propagate = False # Optional: Prevent messages duplicating if root logger is configured

# Determine paths relative to the Current Working Directory (CWD)
CWD = pathlib.Path.cwd()
DEFAULT_CONFIG_FILENAME = "config.yaml"
DEFAULT_EXAMPLE_CONFIG_FILENAME = "config.example.yaml"
DEFAULT_CONFIG_PATH = CWD / DEFAULT_CONFIG_FILENAME
DEFAULT_EXAMPLE_CONFIG_PATH = CWD / DEFAULT_EXAMPLE_CONFIG_FILENAME


# --- 默认配置字典 ---
# 这个字典定义了所有配置项的默认值。
# 它应该与 `config.example.yaml` 中的结构和默认值保持一致。
_DEFAULT_CONFIG: Dict[str, Any] = {
    # -------------------------------------------------------------------------
    # Dashscope (阿里云灵积) 服务设置
    # -------------------------------------------------------------------------
    "dashscope": {
        # API 密钥 (必需) - 强烈建议通过环境变量 `DASHSCOPE_API_KEY` 设置。
        "api_key": "",
        # --- 语音转文本 (STT) / 语音翻译 (STT+Translation) 设置 ---
        "stt": {
            # 翻译的目标语言 (可选) - `None` 或空字符串 `""` 表示禁用翻译。
            "translation_target_language": None,  # 示例: "en", "ja", "ko"
            # 选择要使用的 Dashscope STT 模型 (必需)
            "selected_model": "gummy-realtime-v1",
            # 定义可用的 Dashscope STT 模型及其配置 (必需)
            "models": {
                "gummy-realtime-v1": {
                    "type": "gummy", # 添加类型字段
                    "sample_rate": 16000,
                    "supports_translation": True,
                },
                "paraformer-realtime-v2": {
                    "type": "paraformer", # 添加类型字段
                    "sample_rate": 16000,
                    "supports_translation": False,
                },
            },
            # 中间结果处理方式 (可选, 仅影响 VRChat OSC 输出)
            "intermediate_result_behavior": "ignore",  # 可选: "show_typing", "show_partial"
        },
    },
    # -------------------------------------------------------------------------
    # 音频输入设置
    # -------------------------------------------------------------------------
    "audio": {
        # 音频输入设备名称 (可选) - `None` 或 "Default" 表示使用系统默认设备。
        "device": None,  # 示例: "麦克风 (Realtek High Definition Audio)"
        # 音频采样率 (Hz) - 不再在此处配置，而是从所选的 STT 模型配置中获取。
        # 音频通道数 (可选) - Dashscope 模型通常需要 1。
        "channels": 1,
        # 音频数据类型 (可选) - Dashscope 模型通常需要 'int16'。
        "dtype": "int16",
        # 调试回声模式 (可选) - 将麦克风输入直接发送到扬声器。
        "debug_echo_mode": False,
    },
    # -------------------------------------------------------------------------
    # 大型语言模型 (LLM) 处理设置
    # -------------------------------------------------------------------------
    "llm": {
        # 是否启用 LLM 处理 (必需)
        "enabled": False,
        # OpenAI 兼容 API 密钥 (必需, 如果 enabled 为 true) - 强烈建议通过环境变量 `OPENAI_API_KEY` 设置。
        "api_key": "",
        # API 基础 URL (可选) - `None` 表示使用默认 OpenAI URL 或服务特定 URL。
        "base_url": None,  # 示例: "http://localhost:11434/v1" (本地 Ollama)
        # 使用的 LLM 模型名称 (必需, 如果 enabled 为 true)
        "model": "gpt-3.5-turbo",  # 示例: "gpt-4", "llama3"
        # 系统提示 (必需, 如果 enabled 为 true) - 指导 LLM 如何处理文本。
        "system_prompt": "You are a helpful assistant.",  # 默认提示
        # 温度 (可选) - 控制输出随机性 (0.0-2.0)。
        "temperature": 0.7,
        # 最大 Token 数 (可选) - 限制 LLM 响应长度。
        "max_tokens": 256,  # 之前是 150，与 YAML 示例同步为 256
        # Few-shot 示例 (可选) - 提供输入/输出对指导 LLM。
        "few_shot_examples": [],  # 示例: [{"user": "你好", "assistant": "你好呀！"}]
    },
    # -------------------------------------------------------------------------
    # 输出目标设置
    # -------------------------------------------------------------------------
    "outputs": {
        # --- VRChat OSC 输出 ---
        "vrc_osc": {
            # 是否启用 VRChat OSC 输出 (必需)
            "enabled": True,
            # VRChat 客户端 IP 地址 (必需)
            "address": "127.0.0.1",
            # VRChat OSC 输入端口 (必需)
            "port": 9000,
            # 消息发送最小间隔 (秒) (可选) - 建议 >= 1.333。
            "message_interval": 1.333,
            # 消息格式字符串 (可选) - `{text}` 会被替换。
            "format": "{text}",  # 与 YAML 示例同步，移除前后制表符
            # 是否立即发送 (可选) - `True` 直接显示, `False` 填充输入框。
            "send_immediately": True,
            # 是否播放通知音 (可选) - 仅当 `send_immediately` 为 `True` 时有效。
            "play_notification_sound": True,
        },
        # --- 控制台输出 ---
        "console": {
            # 是否启用控制台输出 (必需)
            "enabled": True,  # 之前是 False，与 YAML 示例同步为 True
            # 输出前缀 (可选)
            "prefix": "[VRCMeow Output]",  # 与 YAML 示例同步
        },
        # --- 文件输出 ---
        "file": {
            # 是否启用文件输出 (必需)
            "enabled": False,
            # 输出文件路径 (必需, 如果 enabled 为 true)
            "path": "vrcmeow_output.log",  # 与 YAML 示例同步
            # 文件记录格式字符串 (可选) - 可用 `{timestamp}`, `{text}`。
            "format": "{timestamp} - {text}",
        },
        # 在此添加其他输出类型 (未来扩展)
    },
    # -------------------------------------------------------------------------
    # 日志记录设置
    # -------------------------------------------------------------------------
    "logging": {
        # 控制台和文件日志的级别 (必需)
        "level": "INFO",  # 可选: "DEBUG", "WARNING", "ERROR", "CRITICAL"
        # --- 应用程序日志文件设置 ---
        "file": {
            # 是否启用应用程序日志文件 (必需) - 记录详细运行日志。
            "enabled": True,
            # 应用程序日志文件路径 (必需, 如果 enabled 为 true)
            "path": "vrcmeow_app.log",
        },
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
            else:  # If d[k] is not a dict (or doesn't exist), replace it
                d[k] = v
        else:
            d[k] = v
    return d


class Config:
    """Singleton class to load and hold application configuration."""

    _instance: Optional["Config"] = None
    _config_data: Dict[str, Any] = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            # Load config only once when the first instance is created
            # Load config using the default path based on CWD
            cls._instance._load_config(str(DEFAULT_CONFIG_PATH))
        return cls._instance

    def _load_config(self, config_path: str) -> None:  # Default removed, set in __new__
        """Loads configuration from file and environment variables."""
        # config_path is now expected to be an absolute path
        config_path_obj = pathlib.Path(config_path)  # Work with Path object
        self._loaded_config_path = f"Attempting: {config_path_obj}"  # Initialize status

        # Start with a deep copy of defaults to avoid modifying the original
        # Use recursive copy for nested dicts
        config = {}
        for k, v in _DEFAULT_CONFIG.items():
            if isinstance(v, dict):
                config[k] = (
                    v.copy()
                )  # Shallow copy for top-level dict is okay for defaults
            else:
                config[k] = v

        # 1. Load from file (using absolute path)
        try:
            with open(config_path_obj, "r", encoding="utf-8") as f:
                file_config = yaml.safe_load(f)
                if file_config and isinstance(file_config, dict):
                    config = _recursive_update(config, file_config)
                    logger.info(f"Loaded configuration from {config_path_obj}.")
                    self._loaded_config_path = str(
                        config_path_obj
                    )  # Set path on successful load
                elif file_config:  # Loaded something, but not a dict
                    logger.warning(
                        f"Config file {config_path_obj} does not contain a valid YAML dictionary. Using defaults."
                    )
                    self._loaded_config_path = f"Invalid YAML in {config_path_obj}, using defaults"  # Update status
                else:  # File is empty
                    logger.info(
                        f"Config file {config_path_obj} is empty. Using default configuration."
                    )
                    self._loaded_config_path = f"Empty file: {config_path_obj}, using defaults"  # Update status
        except FileNotFoundError:
            logger.warning(f"Config file '{config_path_obj}' not found in CWD.")
            self._loaded_config_path = f"Not found: {config_path_obj}"  # Update status: file not found initially
            # Look for example config in CWD
            example_config_path_obj = DEFAULT_EXAMPLE_CONFIG_PATH
            try:
                if example_config_path_obj.exists():
                    import shutil

                    shutil.copy2(
                        str(example_config_path_obj), str(config_path_obj)
                    )  # copy2 preserves metadata
                    logger.info(
                        f"Copied '{example_config_path_obj}' to '{config_path_obj}'."
                    )
                    # Now attempt to load the newly created file
                    with open(config_path_obj, "r", encoding="utf-8") as f:
                        file_config = yaml.safe_load(f)
                        if file_config and isinstance(file_config, dict):
                            config = _recursive_update(config, file_config)
                            logger.info(
                                f"Successfully loaded configuration from newly created '{config_path_obj}'."
                            )
                            self._loaded_config_path = f"Copied from example: {config_path_obj}"  # Update status: successfully copied and loaded example
                        else:
                            logger.warning(
                                f"Newly created '{config_path_obj}' is empty or invalid. Using defaults."
                            )
                            self._loaded_config_path = f"Copied example '{config_path_obj}' is invalid, using defaults"  # Update status: copied example is bad
                else:
                    logger.warning(
                        f"Example config file '{example_config_path_obj}' not found in CWD. Using default configuration."
                    )
                    self._loaded_config_path = f"Not found & example missing, using defaults (tried: {config_path_obj})"  # Update status: file and example missing
            except Exception as copy_err:
                logger.error(
                    f"Failed to copy '{example_config_path_obj}' to '{config_path_obj}': {copy_err}. Using default configuration.",
                    exc_info=True,
                )
                self._loaded_config_path = f"Error copying example to {config_path_obj}, using defaults"  # Update status: copy failed

        except yaml.YAMLError as e:
            logger.error(
                f"Error parsing config file {config_path_obj}: {e}. Using default config.",
                exc_info=True,
            )
            self._loaded_config_path = f"YAML error in {config_path_obj}, using defaults"  # Update status: YAML parse error
        except Exception as e:
            logger.error(
                f"Unknown error loading config file {config_path_obj}: {e}. Using default config.",
                exc_info=True,
            )
            self._loaded_config_path = f"Error loading {config_path_obj}, using defaults"  # Update status: other load error

        # 2. Environment variable override (Dashscope API Key)
        env_dash_api_key = os.getenv("DASHSCOPE_API_KEY")
        if env_dash_api_key:
            # Ensure 'dashscope' dict exists before setting the key
            if "dashscope" not in config:
                config["dashscope"] = {}
            if not isinstance(
                config.get("dashscope"), dict
            ):  # Check if existing dashscope is dict
                logger.warning(
                    "Dashscope config section is not a dictionary, cannot override API key. Check config.yaml structure."
                )
            else:
                config["dashscope"]["api_key"] = env_dash_api_key
                logger.info(
                    "Overridden Dashscope API Key using DASHSCOPE_API_KEY environment variable."
                )
        elif not config.get("dashscope", {}).get("api_key"):
            # Check if the key is missing/empty within the dashscope section
            logger.warning(
                "Dashscope API Key not found in config file or DASHSCOPE_API_KEY environment variable."
            )

        # 2b. Environment variable override (LLM API Key)
        env_llm_api_key = os.getenv("OPENAI_API_KEY")
        if env_llm_api_key:
            # Ensure 'llm' dict exists before setting the key
            if "llm" not in config:
                config["llm"] = {}
            if not isinstance(config.get("llm"), dict):  # Check if existing llm is dict
                logger.warning(
                    "LLM config section is not a dictionary, cannot override API key. Check config.yaml structure."
                )
            else:
                config["llm"]["api_key"] = env_llm_api_key
                logger.info(
                    "Overridden LLM API Key using OPENAI_API_KEY environment variable."
                )
        elif config.get("llm", {}).get("enabled") and not config.get("llm", {}).get(
            "api_key"
        ):
            # Check if LLM is enabled but the key is missing/empty
            if config.get("llm", {}).get("enabled") and not config.get("llm", {}).get(
                "api_key"
            ):
                logger.warning(
                    "LLM processing is enabled but API Key not found in config file or OPENAI_API_KEY environment variable."
                )

        # 3. Ensure LLM System Prompt exists (using config value or default)
        try:
            # Ensure llm structure exists
            if "llm" not in config or not isinstance(config.get("llm"), dict):
                config["llm"] = {}  # Initialize if missing or wrong type

            # Get the default prompt from the _DEFAULT_CONFIG dictionary
            default_prompt = _DEFAULT_CONFIG.get("llm", {}).get(
                "system_prompt", "You are a helpful assistant."
            )

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
            logger.error(
                f"Error processing LLM system prompt configuration: {e}. Using default prompt.",
                exc_info=True,
            )
            # Ensure a safe fallback if structure was bad
            if "llm" not in config or not isinstance(config.get("llm"), dict):
                config["llm"] = {}
            config["llm"]["system_prompt"] = _DEFAULT_CONFIG.get("llm", {}).get(
                "system_prompt", "You are a helpful assistant."
            )

        # 4. Validate and transform (Log level)
        try:
            # Ensure logging dict structure exists before accessing nested keys
            if "logging" not in config or not isinstance(config.get("logging"), dict):
                config["logging"] = {}  # Initialize if missing or wrong type

            log_level_str = config.get("logging", {}).get("level", "INFO").upper()
            log_level = getattr(logging, log_level_str, logging.INFO)
            if not isinstance(log_level, int):
                logger.warning(
                    f"Invalid log level '{log_level_str}', defaulting to INFO."
                )
                log_level = logging.INFO

            # Store both string and int level for convenience
            config["logging"]["level"] = logging.getLevelName(
                log_level
            )  # Ensure string matches level
            config["logging"]["level_int"] = log_level
        except Exception as e:
            logger.error(
                f"Error processing logging configuration: {e}. Using default log level INFO.",
                exc_info=True,
            )
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
        if "." in key:
            keys = key.split(".")
            value = self._config_data
            try:
                for k in keys:
                    if isinstance(value, dict):
                        value = value[k]
                    else:
                        return default  # Path is invalid if intermediate value is not a dict
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
        if "." in key:
            keys = key.split(".")
            value = self._config_data
            try:
                for k in keys:
                    if isinstance(value, dict):
                        value = value[k]
                    else:
                        raise KeyError(
                            f"Configuration path invalid at '{k}' for key '{key}'."
                        )
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

    @property
    def loaded_config_path(self) -> Optional[str]:
        """Returns the path or status of the configuration file that was loaded."""
        # This property was missing, causing the dashboard to show "Unknown"
        return getattr(
            self, "_loaded_config_path", None
        )  # Safely access the internal attribute

    def reload(self) -> None:
        """Reloads the configuration."""
        logger.info("Reloading configuration...")
        # Reload using the default path based on CWD
        self._load_config(str(DEFAULT_CONFIG_PATH))

    def save(
        self, config_path: str = str(DEFAULT_CONFIG_PATH)
    ) -> None:  # Use absolute default path
        """Saves the current configuration back to the YAML file."""
        logger.info(f"Attempting to save configuration to {config_path}...")
        config_path_obj = pathlib.Path(config_path)  # Work with Path object
        # Create a deep copy to avoid modifying the live config dict directly during preparation
        config_to_save = copy.deepcopy(self._config_data)

        # Remove runtime/derived values that shouldn't be saved
        if isinstance(config_to_save.get("logging"), dict):
            config_to_save["logging"].pop("level_int", None)

        # Decide how to handle secrets potentially loaded from env vars.
        # Option 1: Save the current values (might expose secrets if they came from env)
        # Option 2: Check if the current value matches the env var and save "" if it does. (More complex)
        # Option 3: Never save certain keys (e.g., api_key) - User must use env vars.
        # Let's go with Option 1 for now, assuming the user manages the config file appropriately.
        # If DASHSCOPE_API_KEY was used, config_to_save['dashscope_api_key'] will hold its value.
        # If OPENAI_API_KEY was used, config_to_save['llm']['api_key'] will hold its value.

        try:
            # Ensure the directory exists before saving
            config_path_obj.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path_obj, "w", encoding="utf-8") as f:
                yaml.dump(
                    config_to_save,
                    f,
                    allow_unicode=True,
                    sort_keys=False,
                    default_flow_style=False,
                )
            logger.info(f"Configuration successfully saved to {config_path_obj}.")
        except Exception as e:
            logger.error(
                f"Failed to save configuration to {config_path_obj}: {e}", exc_info=True
            )
            # Re-raise or handle as appropriate for the application context (e.g., show error in GUI)
            raise


# Create the singleton instance upon module import
# This ensures config is loaded once when the application starts
try:
    config = Config()
except Exception as e:
    # Critical error during initial config load
    logger.critical(
        f"CRITICAL ERROR during initial configuration load: {e}", exc_info=True
    )
    logger.critical(
        "Application might not function correctly. Attempting to use fallback defaults."
    )

    # Attempt to provide a minimal fallback using _DEFAULT_CONFIG directly
    # Note: This won't have env vars or log level conversion applied correctly
    class FallbackConfig:
        # Create a deep copy of defaults for fallback
        _data = copy.deepcopy(_DEFAULT_CONFIG)
        # Ensure minimal structure exists for logging (already handled by deepcopy if default is okay)
        if "logging" not in _data:
            _data["logging"] = {}
        _data["logging"]["level"] = _DEFAULT_CONFIG.get("logging", {}).get(
            "level", "INFO"
        )
        _data["logging"]["level_int"] = getattr(
            logging, _data["logging"]["level"], logging.INFO
        )
        # Ensure llm section exists (even in fallback, though keys won't be overridden)
        if "llm" not in _data:
            _data["llm"] = {}

        @property
        def data(self):
            return self._data

        def __getitem__(self, key):  # Basic non-nested access
            try:
                return self._data[key]
            except KeyError:
                raise KeyError(f"Fallback config missing key '{key}'")

        def get(self, key, default=None):  # Basic non-nested get
            return self._data.get(key, default)

    config = FallbackConfig()
    # The warning is logged when the instance is created, not part of the class definition itself
    logger.warning("Using fallback configuration due to critical error during load.")


# Optional: Provide a function alias for compatibility or preference
def get_config_data() -> Dict[str, Any]:
    """Returns the loaded configuration data dictionary."""
    return config.data

