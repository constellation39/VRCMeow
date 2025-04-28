> **申明：** 本程序代码主要由大型语言模型（LLM）生成，其功能、行为及潜在问题与本人无关。使用者需自行承担使用风险。

# VRCMeow

VRCMeow 是一个旨在通过语音识别和可选的语言模型处理来增强 VRChat 交互的工具。它可以将你的语音实时转录（和翻译）并通过 OSC 发送到 VRChat 聊天框。

## 功能

*   **实时语音转录:** 使用 Dashscope 的 STT 服务（Paraformer 或 Gummy）将你的麦克风输入转换为文本。
*   **实时翻译:** （可选，使用 Gummy 模型时）将你的语音翻译成指定语言。
*   **VRChat OSC 集成:** 将转录或翻译的文本发送到 VRChat 聊天框。
*   **LLM 集成:** （可选）将转录的文本发送给兼容 OpenAI API 的 LLM 进行处理，并将结果发送到 VRChat。
*   **灵活配置:** 通过 `config.yaml` 文件自定义 API 密钥、模型、语言和其他行为。
*   **多输出目标:** 除了 VRChat，还可以将结果输出到控制台和文件。

## 设置与安装

1.  **克隆仓库:**
    ```bash
    git clone https://github.com/your-username/VRCMeow.git # 替换为你的仓库 URL
    cd VRCMeow
    ```
2.  **创建虚拟环境 (推荐):**
    ```bash
    # 使用 uv 创建虚拟环境 (如果尚不存在，默认创建在 .venv)
    uv venv
    # 激活虚拟环境 (uv 会自动检测并使用 .venv 目录)
    # Windows (PowerShell/cmd):
    .venv\Scripts\activate
    # macOS/Linux (bash/zsh):
    # source .venv/bin/activate
    # 或者，你可以直接使用 uv run 来执行命令，无需手动激活
    ```
3.  **安装依赖:**
    *   确保你安装了 [PortAudio](http://www.portaudio.com/download.html) (`sounddevice` 依赖)。在某些系统上可能需要手动安装。
    *   使用 `uv` 安装/同步 Python 依赖 (依赖项在 `pyproject.toml` 中定义):
        ```bash
        uv sync
        ```

## 配置

1.  复制 `config.example.yaml` 并将其重命名为 `config.yaml`。
2.  编辑 `config.yaml` 文件：
    *   **`dashscope_api_key`:** **必需。** 填入你的 Dashscope API 密钥。强烈建议通过环境变量 `DASHSCOPE_API_KEY` 设置以提高安全性。
    *   **`stt` 部分:**
        *   `model`: 选择要使用的 Dashscope STT 模型（例如 `"gummy-realtime-v1"` 支持翻译，`"paraformer-realtime-v2"` 仅转录）。
        *   `translation_target_language`: 如果使用支持翻译的模型（如 Gummy）并需要翻译，设置目标语言代码（例如 `"en"`, `"ja"`）。留空则禁用翻译。
        *   `intermediate_result_behavior`: 配置如何处理中间（非最终）识别结果 (`"ignore"`, `"show_typing"`, `"show_partial"`)。
    *   **`audio` 部分:**
        *   `sample_rate`, `channels`, `dtype`: 通常保持 Dashscope 模型要求的默认值（例如 Gummy 为 16000, 1, "int16"）。
        *   `debug_echo_mode`: 设置为 `true` 以将麦克风输入直接回显到扬声器，用于测试音频输入。
    *   **`llm` 部分:** （如果需要 LLM 处理）
        *   设置 `enabled: true`。
        *   `api_key`: 填入你的 OpenAI 兼容服务的 API 密钥。强烈建议通过环境变量 `OPENAI_API_KEY` 设置。
        *   `base_url`: （可选）如果你使用本地 LLM (如 Ollama) 或代理，请设置 API 端点 URL。
        *   `model`: 选择要使用的 LLM 模型名称。
        *   `system_prompt`: 定义 LLM 的行为指令。
        *   `temperature`, `max_tokens`: 控制 LLM 输出的随机性和长度。
        *   `few_shot_examples`: （可选）提供示例以指导 LLM 的响应风格。
    *   **`outputs` 部分:**
        *   **`vrc_osc`:**
            *   `enabled`: 设置为 `true` 以启用 VRChat OSC 输出。
            *   `address`, `port`: 确认 VRChat OSC 的 IP 地址和端口（通常是 `127.0.0.1` 和 `9000`）。
            *   `message_interval`: 发送消息的最小间隔（秒），避免 VRChat 速率限制。
        *   **`console`:**
            *   `enabled`: 设置为 `true` 以在运行 VRCMeow 的控制台中打印最终文本。
            *   `prefix`: （可选）为控制台输出添加前缀。
        *   **`file`:**
            *   `enabled`: 设置为 `true` 以将最终文本附加到文件。
            *   `path`: 指定输出文件的路径。
            *   `format`: 定义文件中每行的格式（可用占位符：`{timestamp}`, `{text}`）。
    *   **`logging` 部分:**
        *   `level`: 设置日志记录的详细程度 (`"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"`)。

## 用法

1.  确保你的 VRChat 客户端已启用 OSC (Settings -> OSC -> Enable OSC)。
2.  激活你的 Python 虚拟环境（如果需要手动激活，请参见安装步骤）。
3.  运行主脚本：
    ```bash
    # 如果已激活虚拟环境
    python main.py
    # 或者使用 uv run (无需手动激活环境)
    # uv run python main.py
    ```
4.  程序将开始监听你的默认麦克风。当你说话时，它会进行转录（和翻译，如果已配置），通过 LLM 处理（如果已配置），并将最终结果发送到配置的输出目标（VRChat、控制台、文件）。
5.  按 `Ctrl+C` 停止程序。

## 贡献

欢迎贡献！如果你发现任何错误或有改进建议，请随时创建 Issue 或提交 Pull Request。

## 许可证

*（在此处添加许可证信息，例如 MIT, Apache 2.0 等。如果未定，可以暂时留空或写 "待定"）*
