import logging
import sys
from typing import Optional

# 尝试导入 config 实例以获取默认日志级别
try:
    from config import config as app_config
    # logger = logging.getLogger(__name__) # 获取一个本地 logger 用于记录配置问题
    # logger.debug("Successfully imported 'config' in logger_config.")
except ImportError:
    # 如果 config 无法导入 (例如，在非常早期的设置阶段或循环依赖？)，
    # 则无法获取配置的日志级别，回退到默认值。
    # logger = logging.getLogger(__name__)
    # logger.warning("Could not import 'config' in logger_config. Using default log level INFO for setup.", exc_info=True)
    app_config = None # 标记配置不可用


# 配置日志格式
# 使用 %(filename)s 显示文件名，%(lineno)d 显示行号
LOG_FORMAT = '%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s'
# 配置日期格式
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# 创建一个全局的日志记录器名称
APP_LOGGER_NAME = 'VRCMeowApp'

def setup_logging(level: Optional[int] = None):
    """
    配置应用程序的日志记录。
    如果未提供 level 参数，则尝试从配置中获取。
    """
    # 确定要使用的日志级别
    if level is None:
        if app_config:
            try:
                # 尝试从配置获取数字级别
                level = app_config['logging.level_int']
            except KeyError:
                 # 如果配置中没有 level_int，尝试从 level 字符串获取
                 level_str = app_config.get('logging.level', 'INFO').upper()
                 level = logging.getLevelName(level_str)
                 if not isinstance(level, int):
                     print(f"Warning: Invalid logging level '{level_str}' in config. Defaulting to INFO.", file=sys.stderr)
                     level = logging.INFO
            except Exception as e:
                print(f"Warning: Error reading logging level from config: {e}. Defaulting to INFO.", file=sys.stderr)
                level = logging.INFO
        else:
             # 如果无法导入 app_config，默认使用 INFO
             level = logging.INFO
             print("Warning: Application config not available for logging setup. Defaulting to INFO level.", file=sys.stderr)

    # --- 获取并配置根记录器 ---
    root_logger = logging.getLogger() # 获取根记录器
    # 设置根记录器的级别
    root_logger.setLevel(level)

    # 清理根记录器上可能存在的旧处理器
    if root_logger.hasHandlers():
        # Iterate over a copy for safe removal
        for handler in root_logger.handlers[:]:
             root_logger.removeHandler(handler)
             handler.close() # Ensure handlers release resources

    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    # 设置处理器级别 (通常与根记录器级别相同，除非需要更精细的控制)
    console_handler.setLevel(level)


    # 创建格式化器
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    console_handler.setFormatter(formatter)

    # 将处理器添加到根记录器
    root_logger.addHandler(console_handler)

    # 可以选择添加文件处理器
    # try:
    #     file_handler = logging.FileHandler("app.log")
    #     file_handler.setLevel(level) # Set file handler level
    # file_handler.setFormatter(formatter)
    #     file_handler.setFormatter(formatter)
    #     root_logger.addHandler(file_handler)
    # except Exception as e:
    #     root_logger.error(f"Failed to configure file logging: {e}", exc_info=True)


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
    if not logger.hasHandlers() and not logging.getLogger().hasHandlers():
         # Attempt to set up basic logging if not configured
         print(f"Warning: Logger '{name}' has no handlers. Attempting basic setup.", file=sys.stderr)
         logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT)
         logger = logging.getLogger(name) # Re-get logger after basicConfig

    return logger

# Note: Calling setup_logging() here at module load time might be too early
# if the config file hasn't been read yet. It's better called explicitly
# in the application entry point (like main.py) after config is loaded.
# setup_logging()
