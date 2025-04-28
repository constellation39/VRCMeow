import asyncio

# import os
import sounddevice as sd
from audio_recorder import start_audio_processing
from typing import Optional  # Import Optional for type hints

# Directly from config module
from config import config
from logger_config import setup_logging, get_logger

# Component Imports - Assume these imports succeed. If not, the program will fail, which is intended.
from osc_client import VRCClient
from llm_client import LLMClient
from output_dispatcher import OutputDispatcher


async def main():
    # 1. 配置日志 (日志配置现在从 config 模块内部读取默认级别)
    # setup_logging 会使用 config['logging.level_int']
    setup_logging()  # 执行日志配置
    logger = get_logger(__name__)  # 获取 main 模块的 logger

    logger.info("--- VRCMeow 启动 ---")
    # 使用 config 实例访问配置
    logger.debug(f"日志级别设置为: {config['logging.level']}")  # Changed to DEBUG

    # 2. 获取配置值 (直接使用 config 实例)
    dashscope_api_key = config.get("dashscope_api_key")  # 使用 get 以防万一
    # Access OSC settings under the 'outputs' section using dot notation
    osc_address = config.get(
        "outputs.vrc_osc.address", "127.0.0.1"
    )  # Use get with default
    osc_port = config.get("outputs.vrc_osc.port", 9000)  # Use get with default
    osc_interval = config.get(
        "outputs.vrc_osc.message_interval", 1.333
    )  # Use get with default
    debug_mode = config.get("audio.debug_echo_mode", False)  # Use get with default
    # 不再需要传递 audio_config 和 stt_config 字典

    # 3. 检查关键配置：API Key
    if not dashscope_api_key:
        logger.error(
            "错误：Dashscope API Key 未在 config.yaml 或 DASHSCOPE_API_KEY 环境变量中设置。"
        )
        logger.error("请设置 API Key 后重试。程序即将退出。")
        return

    logger.debug("Dashscope API Key 已加载。")  # Changed to DEBUG
    # 警告：在日志中打印 API 密钥存在安全风险。仅用于调试目的。
    # Consider removing or further restricting this log even at DEBUG level in production environments.
    logger.debug(
        f"使用的 Dashscope API Key (部分隐藏): {dashscope_api_key[:4]}...{dashscope_api_key[-4:]}"
        if dashscope_api_key
        else "未设置"
    )  # Masked key
    logger.info(f"OSC 输出将发送到: {osc_address}:{osc_port}")
    logger.info(f"OSC 消息最小间隔: {osc_interval} 秒")
    logger.info(f"音频调试回声模式: {'启用' if debug_mode else '禁用'}")

    # --- 日志记录音频设备 ---
    try:
        devices = sd.query_devices()
        default_input_device_index = sd.default.device[0]
        default_output_device_index = sd.default.device[1]

        if default_input_device_index != -1 and default_input_device_index < len(
            devices
        ):
            input_device_name = devices[default_input_device_index]["name"]
            logger.info(f"默认输入设备: {input_device_name}")
        else:
            logger.warning("未找到默认输入设备或索引无效。")

        if default_output_device_index != -1 and default_output_device_index < len(
            devices
        ):
            output_device_name = devices[default_output_device_index]["name"]
            logger.info(f"默认输出设备: {output_device_name}")
        else:
            logger.warning("未找到默认输出设备或索引无效。")
    except Exception as e:
        logger.error(f"查询音频设备时出错: {e}", exc_info=True)

    # --- 日志记录 STT 配置 (直接从 config 实例读取) ---
    stt_model = config["stt.model"]
    stt_target_language = config.get(
        "stt.translation_target_language"
    )  # 使用 get 处理 None
    stt_intermediate_behavior = config.get("stt.intermediate_result_behavior", "ignore")
    translation_enabled = bool(stt_target_language)

    logger.info(f"STT 翻译: {'启用' if translation_enabled else '禁用'}")
    if translation_enabled:
        logger.info(f"STT 翻译目标语言: {stt_target_language}")
    logger.info(f"STT 模型: {stt_model}")
    logger.info(f"STT 中间结果处理: {stt_intermediate_behavior}")

    # --- 日志记录 LLM 配置 ---
    llm_enabled = config.get("llm.enabled", False)
    logger.info(f"LLM 处理: {'启用' if llm_enabled else '禁用'}")
    if llm_enabled:
        # Assume LLMClient was imported successfully
        logger.info(f"LLM 模型: {config.get('llm.model')}")
        # Log other LLM parameters at DEBUG level if desired
        logger.debug(f"LLM Base URL: {config.get('llm.base_url')}")
        logger.debug(f"LLM System Prompt: {config.get('llm.system_prompt')[:50]}...")
        logger.debug(
            f"LLM Temp: {config.get('llm.temperature')}, Max Tokens: {config.get('llm.max_tokens')}"
        )

    # 4. 实例化 VRCClient (如果 VRC OSC 输出启用)
    vrc_client_instance: Optional[VRCClient] = None
    vrc_osc_enabled = config.get("outputs.vrc_osc.enabled", False)
    if vrc_osc_enabled:
        # Assume VRCClient was imported successfully
        vrc_client_instance = VRCClient(
            address=osc_address, port=osc_port, interval=osc_interval
        )
        logger.info("VRCClient 已初始化。")
    else:
        logger.info("VRC OSC 输出已禁用，跳过 VRCClient 初始化。")

    # 5. 实例化 LLMClient (如果 LLM 启用)
    llm_client_instance: Optional[LLMClient] = None
    if llm_enabled:
        # Assume LLMClient was imported successfully
        llm_client_instance = LLMClient()
        if (
            not llm_client_instance.enabled
        ):  # Check if internal initialization failed (e.g., missing API key)
            logger.warning("LLMClient 初始化失败或 API Key 缺失，LLM 处理将被禁用。")
            llm_client_instance = None  # Ensure it's None if disabled internally
        else:
            logger.info("LLMClient 已初始化。")

    # 6. 实例化 OutputDispatcher
    # Assume OutputDispatcher was imported successfully
    # Pass the VRC client instance (it can be None if VRC OSC is disabled)
    output_dispatcher_instance = OutputDispatcher(
        vrc_client_instance=vrc_client_instance
    )
    logger.info("OutputDispatcher 已初始化。")

    # 7. 启动处理流程
    try:
        # Start VRCClient context manager *only if* it was successfully created
        if vrc_client_instance:
            async with vrc_client_instance:
                # Start audio processing, passing the necessary components
                await start_audio_processing(
                    # vrc_client removed
                    llm_client=llm_client_instance,
                    output_dispatcher=output_dispatcher_instance,
                )
        else:
            # If VRC client is disabled or failed, just run audio processing directly
            # The OutputDispatcher will handle not sending to VRC OSC
            await start_audio_processing(
                # vrc_client removed
                llm_client=llm_client_instance,
                output_dispatcher=output_dispatcher_instance,
            )

        # If VRCClient instantiation failed or OSC is disabled, this part is skipped.
        # 则需要重新引入条件逻辑，但不是基于导入是否成功。

    except KeyboardInterrupt:
        # KeyboardInterrupt is handled by asyncio.run and the finally blocks
        logger.info("\n主程序检测到 Ctrl+C，开始退出...")
        # Cleanup is handled by __aexit__ in VRCClient and finally in start_audio_processing


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # asyncio.run 内部会处理 KeyboardInterrupt，
        # 但为了确保打印最终消息，这里可以再捕获一次（尽管通常不需要）
        # 获取 logger 的标准方式
        get_logger(__name__).info("\n主程序捕获到 KeyboardInterrupt，确保退出。")
