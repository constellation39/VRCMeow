import asyncio
import time
from typing import Any
from logger_config import get_logger # 导入日志获取器

# 获取该模块的 logger 实例
logger = get_logger(__name__) # 使用模块名作为 logger 名称

# 直接导入，如果失败则程序会退出
from pythonosc import udp_client as pythonosc_udp_client  # 重命名以避免冲突
from pythonosc.osc_message_builder import ArgValue


class VRCClient:
    """
    一个线程安全的客户端，用于使用 OSC 协议向 VRChat 发送消息。
    仅保留并发送最新的消息。
    """

    # 移除 __init__ 中的默认值，这些值现在来自配置
    def __init__(
            self,
            address: str,
            port: int,
            interval: float,
    ) :
        """
        初始化 VRChat OSC 客户端。

        Args:
            address (str): OSC 服务器地址 (来自配置)
            port (int): OSC 服务器端口 (来自配置)
            interval (float): 消息之间的最小时间间隔（秒）(来自配置)
        """
        # 使用传入的参数初始化客户端
        self._osc_client = pythonosc_udp_client.SimpleUDPClient(address, port)
        self.interval = interval
        self._address = address # 可以存储起来用于日志记录
        self._port = port       # 可以存储起来用于日志记录

        self._lock = asyncio.Lock()  # 用于保护共享资源的异步锁
        self._last_send_time = 0.0  # 上次发送消息的时间戳
        self._current_message: ArgValue | None = None  # 当前待发送的消息
        self._message_updated = asyncio.Event()  # 用于通知消息已更新的事件

        self._running = asyncio.Event()  # 用于控制后台任务运行状态的事件
        self._process_task: asyncio.Task | None = None  # 后台消息处理任务

    async def start(self) -> None:
        """启动消息处理循环。"""
        async with self._lock:
            if not self._running.is_set():
                self._running.set()  # 标记为运行中
                # 创建并启动后台消息处理任务
                self._process_task = asyncio.create_task(self._process_messages())
                logger.info(f"VRCClient 已启动，目标: {self._address}:{self._port}, 间隔: {self.interval}s")

    async def stop(self) -> None:
        """停止消息处理循环并进行清理。"""
        async with self._lock:
            if self._running.is_set():
                self._running.clear()  # 标记为停止
                self._message_updated.set()  # 唤醒处理循环以使其退出

                # 等待后台任务结束
                if self._process_task and not self._process_task.done():
                    try:
                        # 等待任务完成，设置超时时间
                        await asyncio.wait_for(self._process_task, timeout=5.0)
                    except asyncio.TimeoutError:  # 从 TimeoutError 更改
                        logger.warning("消息处理任务在关闭时超时")
                    finally:
                        self._process_task = None  # 清理任务引用

                await self._clear_typing_status()  # 清除 VRChat 中的输入状态
                logger.info("VRCClient 已停止")

    async def send_chatbox(self, content: ArgValue) -> None:
        """
        更新要发送到 VRChat 的当前消息。

        Args:
            content (ArgValue): 要发送的消息内容
        """
        if not content:  # 如果内容为空则不处理
            return

        if not self._running.is_set():  # 如果客户端未运行则发出警告
            logger.warning("客户端未运行，无法发送消息")
            return

        async with self._lock:
            current_time = time.time()
            # 检查是否已超过发送间隔
            if current_time - self._last_send_time >= self.interval:
                # 如果超过间隔，立即发送
                await self._send_message(content)
            else:
                # 如果未超过间隔，更新当前消息并显示输入状态
                self._current_message = content
                await self._show_typing_status()  # 显示正在输入...
                self._message_updated.set()  # 通知后台任务有新消息

    async def _send_message(self, content: ArgValue) -> None:
        """通过 OSC 发送消息。"""
        try:
            # 首先发送 typing=False 清除输入状态
            self._osc_client.send_message("/chatbox/typing", False)
            # 然后发送消息内容，send_now=True 表示立即发送
            self._osc_client.send_message("/chatbox/input", [content, True])  # VRC 需要列表 [message, send_now=True]
            self._last_send_time = time.time()  # 更新上次发送时间
            logger.info(f"OSC 消息已发送: {content}") # 使用 logger.info
        except Exception as e:
            logger.error(f"发送 OSC 消息时出错: {e}", exc_info=True) # 添加 exc_info 获取堆栈跟踪

    async def _show_typing_status(self) -> None:
        """在 VRChat 中显示输入状态。"""
        try:
            self._osc_client.send_message("/chatbox/typing", True)
        except Exception as e:
            logger.warning(f"设置 OSC 输入状态 (typing=True) 失败: {e}")

    async def _clear_typing_status(self) -> None:
        """在 VRChat 中清除输入状态。"""
        try:
            self._osc_client.send_message("/chatbox/typing", False)
        except Exception as e:
            logger.warning(f"清除 OSC 输入状态 (typing=False) 失败: {e}")

    async def _process_messages(self) -> None:
        """
        处理最新消息的后台任务。
        仅在冷却期过后才发送最新的消息。
        """
        try:
            while self._running.is_set():
                await self._message_updated.wait()  # 等待新消息或停止信号
                if not self._running.is_set():  # 如果收到停止信号，则退出循环
                    break

                async with self._lock:
                    if self._current_message is None:
                        # 消息可能已被清除、通过其他路径发送，或者是虚假唤醒
                        self._message_updated.clear()  # 确保下次等待真正的更新
                        continue

                    # 存储我们打算处理的消息
                    message_to_process = self._current_message
                    current_time = time.time()
                    time_since_last_send = current_time - self._last_send_time
                    wait_time = 0.0  # 需要等待的时间

                    # 如果距离上次发送的时间小于间隔
                    if time_since_last_send < self.interval:
                        # 计算需要等待的时间
                        wait_time = self.interval - time_since_last_send

                    if wait_time > 0:
                        # 在持有锁的情况下等待冷却期结束
                        # 这确保我们在延迟后发送 'message_to_process'
                        await asyncio.sleep(wait_time)
                        # 检查在睡眠期间是否被停止
                        if not self._running.is_set():
                            break  # 如果已停止则退出循环

                    # 现在，发送在开始处理时（或等待所需冷却时间后）的当前消息
                    # 我们知道在等待开始时 self._current_message 是 message_to_process，
                    # 并且锁阻止了更改。现在发送它。
                    await self._send_message(message_to_process)
                    # *仅当* 消息没有被更新的消息替换时才清除它
                    # （尽管锁在这里阻止了这种情况）。
                    # 仅作检查，主要是为了确保我们清除正确的状态。
                    if self._current_message == message_to_process:
                        self._current_message = None

                    self._message_updated.clear()  # 清除事件，为下一条消息做好准备

        except asyncio.CancelledError:
            logger.debug("消息处理任务已取消") # 使用 logger.debug
        finally:
            # 确保在任务结束时清除输入状态
            await self._clear_typing_status()

    async def __aenter__(self) -> "VRCClient":
        """异步上下文管理器入口点。"""
        await self.start()  # 启动客户端
        return self  # 返回客户端实例

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """异步上下文管理器退出点。"""
        await self.stop()  # 停止客户端并清理
        return None  # 显式返回 None
