import logging
import pathlib
import sys
from typing import Optional, Callable  # Add Callable
import os  # Ensure os module is imported

# Import the Flet handler (assuming gui_log.py is in the same directory or accessible)
try:
    from gui_log import FletLogHandler
except ImportError:
    # Handle case where GUI is not used or file is missing
    FletLogHandler = None

# Directly import the config instance and APP_DIR. If this fails, the application should exit.
try:
    from config import config as app_config, APP_DIR
except ImportError as e:
    # Log critical error and potentially exit if config/APP_DIR cannot be imported
    # Using print as logger might not be configured yet
    print(
        f"CRITICAL: Failed to import config or APP_DIR: {e}. Logging setup cannot proceed correctly.",
        file=sys.stderr,
    )
    # Fallback APP_DIR to CWD to allow *some* logging, but warn heavily
    APP_DIR = pathlib.Path.cwd()
    print(
        f"WARNING: Falling back to CWD for log file path resolution: {APP_DIR}",
        file=sys.stderr,
    )
    # Attempt to import just the config for level setting, might still fail
    try:
        from config import config as app_config
    except ImportError:
        app_config = None  # Indicate config is unavailable
        print(
            "WARNING: Config instance unavailable, using default log level INFO.",
            file=sys.stderr,
        )


# 配置日志格式
# 使用 %(filename)s 显示文件名，%(lineno)d 显示行号
LOG_FORMAT = "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
# 配置日期格式
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 创建一个全局的日志记录器名称
APP_LOGGER_NAME = "VRCMeowApp"


def setup_logging(
    level: Optional[int] = None,
    log_update_callback: Optional[
        Callable[[], None]
    ] = None,  # Add callback for Flet handler
):
    """
    配置应用程序的日志记录。
    如果未提供 level 参数，则尝试从配置中获取。
    如果提供了 log_update_callback，则添加 Flet GUI 日志处理器。
    """
    # 确定要使用的日志级别
    if level is None:
        try:
            # 尝试从配置获取数字级别
            level = app_config["logging.level_int"]
        except KeyError:
            # 如果配置中没有 level_int，尝试从 level 字符串获取
            level_str = app_config.get("logging.level", "INFO").upper()
            level = logging.getLevelName(level_str)
            if not isinstance(level, int):
                print(
                    f"Warning: Invalid logging level '{level_str}' in config. Defaulting to INFO.",
                    file=sys.stderr,
                )
                level = logging.INFO
        except Exception as e:
            # Catch potential errors reading from config (e.g., config file issues)
            print(
                f"Warning: Error reading logging level from config: {e}. Defaulting to INFO.",
                file=sys.stderr,
            )
            level = logging.INFO

    # --- 获取并配置根记录器 ---
    # Ensure app_config was imported successfully before proceeding
    # (Although if it failed, the program likely exited already)
    # assert app_config is not None, "Config must be loaded before setting up logging."

    root_logger = logging.getLogger()  # 获取根记录器
    # 设置根记录器的级别
    root_logger.setLevel(level)

    # 清理根记录器上可能存在的旧处理器
    if root_logger.hasHandlers():
        # Iterate over a copy for safe removal
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()  # Ensure handlers release resources

    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    # 设置处理器级别 (通常与根记录器级别相同，除非需要更精细的控制)
    console_handler.setLevel(level)

    # 创建格式化器
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    console_handler.setFormatter(formatter)

    # 将控制台处理器添加到根记录器
    root_logger.addHandler(console_handler)

    # --- 添加 Flet GUI 日志处理器 (如果提供了回调) ---
    if FletLogHandler and log_update_callback:
        try:
            flet_handler = FletLogHandler(level=level)  # Use the same level
            root_logger.addHandler(flet_handler)
            root_logger.info("Flet GUI logging handler enabled.")
            # Note: The actual UI update is triggered separately by periodically
            # calling log_update_callback which checks the queue filled by FletLogHandler.
        except Exception as e:
            root_logger.error(
                f"Failed to configure Flet GUI logging handler: {e}", exc_info=True
            )
    elif log_update_callback and not FletLogHandler:
        root_logger.warning(
            "FletLogHandler could not be imported. GUI logging disabled."
        )

    # --- 添加应用程序日志文件处理器 (如果已启用) ---
    try:
        # 检查配置是否启用了应用程序文件日志
        if app_config.get(
            "logging.file.enabled", False
        ):  # <-- 使用 logging.file.enabled
            log_file_path = app_config.get(
                "logging.file.path", "vrcmeow_app.log"
            )  # <-- 使用 logging.file.path，提供默认值
            # 确保目录存在 (如果路径包含目录)
            log_dir = os.path.dirname(log_file_path)
            if log_dir and not os.path.exists(log_dir):
                try:
                    os.makedirs(log_dir)
                    root_logger.info(f"Created log directory: {log_dir}")
                except OSError as e:
                    root_logger.error(
                        f"Failed to create log directory {log_dir}: {e}. File logging disabled.",
                        exc_info=True,
                    )
                    # 如果无法创建目录，则不添加文件处理器
                    log_file_path = None  # 阻止 FileHandler 创建

            if log_file_path:  # 仅在路径有效时尝试创建处理器
                # 创建文件处理器，使用 'a' 模式追加日志
                file_handler = logging.FileHandler(
                    log_file_path, mode="a", encoding="utf-8"
                )
                # 设置文件处理器的级别 (与根记录器相同)
                file_handler.setLevel(level)
                # 设置文件处理器的格式化器 (与控制台相同)
                file_handler.setFormatter(formatter)
                # 将文件处理器添加到根记录器
                root_logger.addHandler(file_handler)
                root_logger.info(
                    f"Application file logging enabled. Logging to: {log_file_path}"  # Use resolved path in log
                )
        else:
            root_logger.info(
                "Application file logging is disabled in the configuration."
            )

    except Exception as e:
        # 捕获创建或添加文件处理器时可能发生的任何错误
        root_logger.error(
            f"Failed to configure application file logging: {e}", exc_info=True
        )  # <-- 更新日志消息

    # 返回配置好的根记录器 (虽然通常不需要直接使用它)
    return root_logger


# 在模块加载时进行一次配置，或者在使用前调用 setup_logging
# 这里我们提供一个函数来获取已配置的记录器
def get_logger(name: str = APP_LOGGER_NAME) -> logging.Logger:
    """获取指定名称的日志记录器实例。

    假设 setup_logging() 已在应用程序启动时被调用以配置日志系统。
    """
    # 直接获取记录器实例。它将继承根记录器或已配置的 VRCMeowApp 记录器的设置。
    # 确保在获取 logger 前， setup_logging 已经被调用过一次
    logger = logging.getLogger(name)
    # If the logger has no handlers, it likely means setup_logging hasn't run
    # or didn't configure this specific logger (or the root logger).
    # This check helps diagnose configuration issues.
    # Note: This check might be too simplistic if complex logging hierarchies are used.
    # Check if the logger OR the root logger has handlers.
    # If neither has handlers, it means logging hasn't been configured yet.
    # Add a NullHandler to prevent "No handlers could be found" warnings
    # during initial module imports before setup_logging is called.
    # See: https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library
    if not logger.hasHandlers() and not logging.getLogger().hasHandlers():
        # Add a NullHandler to this specific logger
        logger.addHandler(logging.NullHandler())
        # Optionally, you could add it to the root logger instead,
        # but adding it here is more targeted.
        # logging.getLogger().addHandler(logging.NullHandler())

        # The previous warning print is removed as NullHandler addresses the issue.

    return logger


# Note: Calling setup_logging() here at module load time is generally discouraged
# as the config might not be fully loaded or other initializations might be pending.
# It should be called explicitly in the application entry point (e.g., main.py)
# after the configuration is ready.
