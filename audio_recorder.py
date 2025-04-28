import asyncio
from typing import Optional
from typing import Union

import numpy as np
import sounddevice as sd

# 直接从 config 模块导入 config 实例
from config import config
# 获取该模块的 logger 实例
from logger_config import get_logger

logger = get_logger(__name__)

# 导入 Dashscope 基础识别器类型用于类型提示
from dashscope.audio.asr import TranslationRecognizerRealtime, Recognition

# 导入 VRC 客户端
try:
    from osc_client import VRCClient
except ImportError:
    VRCClient = None  # Define as None if import fails

# 导入特定引擎的创建函数
try:
    from stt_gummy import create_gummy_recognizer
except ImportError as e:
    logger.error(f"无法导入 Gummy STT 模块: {e}")
    create_gummy_recognizer = None
try:
    from stt_paraformer import create_paraformer_recognizer
except ImportError as e:
    logger.error(f"无法导入 Paraformer STT 模块: {e}")
    create_paraformer_recognizer = None

# --- 异步队列 ---
audio_queue = asyncio.Queue()

from typing import TYPE_CHECKING  # Use TYPE_CHECKING for conditional imports

# Component Imports for Type Hinting
if TYPE_CHECKING:
    try:
        from llm_client import LLMClient
    except ImportError:
        LLMClient = None  # type: ignore
    try:
        from output_dispatcher import OutputDispatcher
    except ImportError:
        OutputDispatcher = None  # type: ignore
    try:
        from osc_client import VRCClient
    except ImportError:
        VRCClient = None  # type: ignore


# Actual imports needed at runtime (handled elsewhere, these are just for hints)


# --- STT 处理核心逻辑 ---
async def stt_processor(
        vrc_client: Optional['VRCClient'],  # Use forward reference string hint
        llm_client: Optional['LLMClient'],  # Use forward reference string hint
        output_dispatcher: 'OutputDispatcher',  # Use forward reference string hint (required)
        stop_event: asyncio.Event  # Pass stop_event
):
    """异步任务，根据配置管理 Dashscope STT 引擎 (Gummy 或 Paraformer) 并处理音频队列。"""
    # --- Directly import required components if needed inside the function ---
    # (Imports are usually at module level, this is just an example if needed)
    # try:
    #     from llm_client import LLMClient # Not typically needed here if passed as arg
    # except ImportError:
    #      LLMClient = None
    # try:
    #     from output_dispatcher import OutputDispatcher # Not typically needed here
    # except ImportError:
    #     OutputDispatcher = None

    # --- 直接从 config 实例获取配置 ---
    api_key = config['dashscope_api_key']
    model = config['stt.model']
    sample_rate = config['audio.sample_rate']  # 必须存在 (由 start_audio_processing 保证)
    channels = config['audio.channels']
    target_language = config.get('stt.translation_target_language')  # 使用 get 处理 None

    logger.info(f"STT 处理任务 (Dashscope, 模型: {model}) 启动中...")
    recognizer: Optional[Union[TranslationRecognizerRealtime, Recognition]] = None  # 统一变量名
    main_loop = asyncio.get_running_loop()

    # API Key 已经在 main.py 检查过

    # 根据预读取的 target_language 是否设置来决定是否启用翻译
    enable_translation = bool(target_language)

    # --- 根据模型选择 API 和参数 ---
    is_gummy_model = model.startswith("gummy-")
    is_paraformer_model = model.startswith("paraformer-")

    if not is_gummy_model and not is_paraformer_model:
        logger.error(f"不支持的 Dashscope 模型: {model}。请使用 'gummy-' 或 'paraformer-' 开头的模型。")
        stop_event.set()
        return

    # 检查翻译配置与模型的兼容性
    if enable_translation and is_paraformer_model:
        logger.warning(f"模型 '{model}' (Paraformer) 不支持翻译。'translation_target_language' 配置将被忽略。")
        enable_translation = False  # 强制禁用翻译
    # Gummy 模型需要 target_language 才能翻译，enable_translation 已由此派生，无需额外检查

    try:
        # --- 根据模型选择并创建识别器 ---
        if is_gummy_model:
            if create_gummy_recognizer is None:
                logger.error("Gummy STT 模块未能加载，无法创建识别器。")
                stop_event.set()
                return
            engine_type = "Gummy"
            # 创建 Gummy 识别器, 传递所需客户端和分发器
            # Check if the imported function exists before calling
            if create_gummy_recognizer:
                recognizer = create_gummy_recognizer(
                    main_loop=main_loop,
                    vrc_client=vrc_client,
                    llm_client=llm_client,
                    output_dispatcher=output_dispatcher  # Must be provided
                )
            else:
                logger.error("create_gummy_recognizer function not available. Cannot create Gummy recognizer.")
                stop_event.set()
                return

        elif is_paraformer_model:
            # Check if the imported function exists before calling
            if create_paraformer_recognizer:
                engine_type = "Paraformer"
                # 创建 Paraformer 识别器, 传递所需客户端和分发器
                # 注意: Paraformer 回调也需要更新以使用 LLMClient 和 OutputDispatcher
                # TODO: Update Paraformer callback and pass llm_client/output_dispatcher
                recognizer = create_paraformer_recognizer(
                    main_loop=main_loop,
                    vrc_client=vrc_client,
                    # llm_client=llm_client, # Pass when Paraformer callback is updated
                    # output_dispatcher=output_dispatcher # Pass when Paraformer callback is updated
                )
                logger.warning("Paraformer STT 引擎当前未完全集成 LLM 处理和多目标输出，其回调函数需要更新。")
            else:
                logger.error(
                    "create_paraformer_recognizer function not available. Cannot create Paraformer recognizer.")
                stop_event.set()
                return
        # else branch handled earlier

        # --- 启动识别器和主处理循环 ---
        if recognizer:  # 确保识别器已成功创建
            recognizer.start()
            logger.info(f"Dashscope {engine_type} Recognizer 已启动。")
        else:
            # 如果 recognizer 仍然是 None (理论上不应发生，因为前面有检查)
            logger.error(f"未能初始化 Dashscope {engine_type} Recognizer。")
            stop_event.set()
            return

        # --- 主处理循环 ---
        while not stop_event.is_set():
            try:
                # 从队列中获取音频数据 (int16 numpy array)
                audio_data = await asyncio.wait_for(audio_queue.get(), timeout=0.1)

                # 将 numpy 数组转换为 bytes 发送给选定的识别器
                recognizer.send_audio_frame(audio_data.tobytes())

                audio_queue.task_done()  # 标记任务完成
            except asyncio.TimeoutError:
                # 队列为空，继续检查停止信号
                continue
            except asyncio.QueueFull:
                # 理论上不应发生，因为我们是从队列中 get
                logger.warning("STT 处理器中的音频队列意外已满。")
                continue
            except Exception as e:
                # 捕获发送音频帧或其他可能的错误
                logger.error(f"STT 处理任务在发送音频时出错: {e}", exc_info=True)
                # 短暂暂停以避免错误刷屏
                await asyncio.sleep(1)

    except Exception as e:
        # 捕获初始化或启动过程中的错误
        logger.error(f"初始化或运行 Dashscope Recognizer 时出错: {e}", exc_info=True)
        stop_event.set()  # 确保其他部分停止
    finally:
        # engine_type 在 try 块中设置，如果初始化失败则为 "Unknown"
        logger.info(f"STT 处理任务 (Dashscope {engine_type}) 正在停止...")
        # 检查 recognizer 是否已定义并成功初始化
        if 'recognizer' in locals() and recognizer:
            try:
                logger.info(f"正在停止 Dashscope {engine_type} Recognizer...")
                # 停止识别器，这会关闭连接并可能调用 on_close/on_complete
                recognizer.stop()
                logger.info(f"Dashscope {engine_type} Recognizer 已停止。")
            except Exception as e:
                logger.error(f"停止 Dashscope {engine_type} Recognizer 时出错: {e}", exc_info=True)
        else:
            logger.info("Recognizer 未初始化，无需停止。")

        # 清空队列中可能剩余的任务（如果应用要求）
        logger.info("正在清空剩余音频队列...")
        processed_count = 0
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
                audio_queue.task_done()
                processed_count += 1
            except asyncio.QueueEmpty:
                break
        if processed_count > 0:
            logger.info(f"已从队列中丢弃 {processed_count} 个剩余音频块。")
        # 使用 engine_type 更新最终日志
        logger.info(f"STT 处理任务 (Dashscope {engine_type}) 已停止。")


# --- 音频捕获和主控制逻辑 ---
async def start_audio_processing(
        vrc_client: Optional['VRCClient'],  # Use forward reference string hints
        llm_client: Optional['LLMClient'],  # Use forward reference string hints
        output_dispatcher: 'OutputDispatcher'  # Use forward reference string hints (required)
):
    """启动实时音频捕获和 Dashscope STT 处理。"""
    stop_event = asyncio.Event()
    stt_task = None
    default_input_device_info = None  # 初始化设备信息变量

    # --- 直接从 config 实例获取配置 ---
    # 使用点表示法或 get 方法访问嵌套配置
    sample_rate_config = config.get('audio.sample_rate')  # 使用 get 处理 None
    channels = config['audio.channels']
    dtype = config['audio.dtype']
    debug_echo_mode = config['audio.debug_echo_mode']
    model = config['stt.model']
    target_language = config.get('stt.translation_target_language')

    # --- 定义音频回调函数 (闭包访问 debug_echo_mode) ---
    def audio_callback(indata: np.ndarray, outdata: np.ndarray, frames: int, time, status: sd.CallbackFlags):
        """同步回调，处理音频 IO 并将输入放入队列。"""
        if status:
            logger.warning(f"音频回调状态: {status}")

        # 回声模式（用于调试）
        if debug_echo_mode:
            outdata[:] = indata
        else:
            outdata.fill(0)  # 正常模式下输出静音

        # 将音频数据放入队列 (确保数据类型正确)
        try:
            # 确保 indata 是我们期望的 dtype，尽管 sounddevice 通常会处理好
            if indata.dtype != np.dtype(dtype):
                logger.warning(f"接收到的音频数据类型 ({indata.dtype}) 与期望 ({dtype}) 不符，尝试转换。")
                indata = indata.astype(dtype)  # 尝试转换
            audio_queue.put_nowait(indata.copy())
        except asyncio.QueueFull:
            logger.warning("音频处理队列已满，丢弃当前音频帧。")
            pass

    try:
        # --- 查询音频设备信息并确定采样率 ---
        try:
            default_input_device_info = sd.query_devices(kind='input')  # 只查询输入设备
            default_output_device_info = sd.query_devices(kind='output')  # 查询输出设备用于日志记录
            # Log detailed device info at DEBUG level
            if default_input_device_info:
                logger.debug(f"默认麦克风: {default_input_device_info['name']}")
                logger.debug(f"  - 最大输入声道: {default_input_device_info['max_input_channels']}")
                logger.debug(f"  - 默认采样率: {default_input_device_info['default_samplerate']} Hz")
            else:
                logger.warning("无法获取默认麦克风信息。")  # Keep warning as INFO/WARN is appropriate
            if default_output_device_info:
                logger.debug(f"默认扬声器: {default_output_device_info['name']}")
                logger.debug(f"  - 最大输出声道: {default_output_device_info['max_output_channels']}")
                logger.debug(f"  - 默认采样率: {default_output_device_info['default_samplerate']} Hz")
            else:
                logger.warning("无法获取默认扬声器信息。")  # Keep warning

            # --- 决定最终使用的采样率 ---
            if sample_rate_config is None:
                if default_input_device_info and 'default_samplerate' in default_input_device_info:
                    # 使用设备默认采样率，并转换为整数
                    sample_rate = int(default_input_device_info['default_samplerate'])
                    logger.debug(
                        f"配置中未指定 sample_rate，将使用默认麦克风的采样率: {sample_rate} Hz")  # Changed to DEBUG
                else:
                    # 如果无法获取设备信息或默认采样率，回退到一个标准值
                    sample_rate = 16000  # 回退值
                    logger.warning(f"无法获取默认麦克风的采样率，将回退到默认值: {sample_rate} Hz")
            else:
                # 使用配置文件中指定的采样率
                sample_rate = int(sample_rate_config)  # 确保是整数
                logger.info(f"将使用配置文件中指定的采样率: {sample_rate} Hz")

        except Exception as e:
            logger.warning(f"查询音频设备信息时出错: {e}", exc_info=True)
            # 即使查询失败，也要决定采样率
            if sample_rate_config is None:
                sample_rate = 16000  # 回退值
                logger.warning(f"查询设备失败且配置未指定 sample_rate，将回退到默认值: {sample_rate} Hz")
            else:
                sample_rate = int(sample_rate_config)
                logger.warning(f"查询设备失败，将使用配置文件中指定的采样率: {sample_rate} Hz")
            # 尝试继续

        # --- 记录最终使用的配置 ---
        # 更新 config 实例中的 sample_rate (如果之前是 None)
        # 警告：直接修改 config._config_data 不是最佳实践，但为了兼容现有逻辑暂时保留
        # 更好的方法是在 stt_processor 中重新读取 sample_rate
        if sample_rate_config is None:
            try:
                # 尝试直接修改内部字典 (谨慎使用)
                config._config_data['audio']['sample_rate'] = sample_rate
                logger.debug(f"已将运行时确定的采样率 ({sample_rate} Hz) 更新回配置。")  # Changed to DEBUG
            except Exception as e:
                logger.error(f"无法更新配置中的运行时采样率: {e}")
        # 或者，确保 stt_processor 总是从 config 中读取最新的 'audio.sample_rate'

        # 使用最终确定的 sample_rate 记录日志
        logger.info(f"最终使用的采样率: {sample_rate} Hz")  # Keep as INFO
        logger.info(f"最终使用的声道数: {channels}")  # Keep as INFO
        logger.info(f"最终使用的音频格式: {dtype}")  # Keep as INFO
        # logger.info(f"调试回声模式: {'启用' if debug_echo_mode else '禁用'}") # Removed, logged in main.py
        logger.info(f"VRChat OSC 输出: {'启用' if vrc_client else '禁用'}")  # Keep as INFO
        # logger.info(f"STT 引擎: Dashscope (模型: {model})") # Removed, logged in main.py
        # logger.info(f"  - 翻译 (...): ...") # Removed translation status log here, covered in main.py

        # 启动 STT 处理任务，传入 API Key 和配置字典
        # 注意：这里不再需要传递原始的 audio_config 和 stt_config 字典
        # 因为 stt_processor 函数也期望接收这些字典（尽管它现在会在内部预读取值）
        # audio_config 字典已被修改以包含正确的 sample_rate
        # 传递客户端/分发器实例给 stt_processor
        stt_task = asyncio.create_task(stt_processor(
            vrc_client=vrc_client,  # Pass instance (can be None)
            llm_client=llm_client,  # Pass instance (can be None)
            output_dispatcher=output_dispatcher,  # Pass instance (required)
            stop_event=stop_event
        ))

        # 使用 sounddevice Stream 上下文管理器启动音频流
        # 使用从配置加载的参数
        with sd.Stream(samplerate=sample_rate,
                       channels=channels,
                       dtype=dtype,
                       callback=audio_callback) as stream:  # 捕获流对象
            # 记录实际使用的采样率
            logger.info(f"麦克风实际使用的采样率: {stream.samplerate} Hz")
            logger.info("音频流已启动。按 Ctrl+C 停止。")
            # 等待停止信号 (来自 KeyboardInterrupt 或其他错误)
            await stop_event.wait()

    except sd.PortAudioError as e:
        logger.error(f"音频错误: PortAudio 错误 - {e}", exc_info=True)
        logger.error("请确保：")
        logger.error("  - 麦克风和扬声器已连接并被系统识别。")
        logger.error("  - 没有其他应用独占音频设备。")
        # 使用实际配置值显示错误消息
        logger.error(f"  - 默认麦克风支持 {sample_rate} Hz, {channels} 声道, {dtype} 格式。")
        stop_event.set()  # 触发停止
    except ValueError as e:
        logger.error(f"音频参数错误: {e}", exc_info=True)
        # 使用实际配置值显示错误消息
        logger.error(
            f"  - 检查采样率 ({sample_rate}), 声道数 ({channels}), 数据类型 ({dtype}) 是否受支持。")
        stop_event.set()  # 触发停止
    except Exception as e:
        logger.error(f"启动音频处理时发生未知错误: {e}", exc_info=True)
        stop_event.set()  # 触发停止
    finally:
        logger.info("\n正在停止音频流和 STT 任务...")
        stop_event.set()  # 确保停止事件被设置，通知 stt_processor 停止

        if stt_task and not stt_task.done():
            try:
                logger.info("等待 STT 任务完成...")
                # 等待 STT 任务优雅地停止 (包括 Dashscope 关闭)
                await asyncio.wait_for(stt_task, timeout=5.0)  # 增加超时以允许网络关闭
                logger.info("STT 任务已完成。")
            except asyncio.TimeoutError:
                logger.warning("STT 任务停止超时。可能未能完全关闭 Dashscope 连接。")
                stt_task.cancel()  # 尝试取消任务
            except asyncio.CancelledError:
                logger.info("STT 任务被取消。")
            except Exception as e:
                logger.error(f"等待 STT 任务停止时发生错误: {e}", exc_info=True)

        logger.info("音频处理和 STT 任务已停止。")

    logger.info("start_audio_processing 函数执行完毕。")
