> **申明：** 本程序代码主要由大型语言模型（LLM）生成，其功能、行为及潜在问题与本人无关。使用者需自行承担使用风险。

# VRCMeow

VRCMeow 是一个旨在通过语音识别和可选的语言模型处理来增强 VRChat 交互的工具。它可以将你的语音实时转录（和翻译）并通过 OSC
发送到 VRChat 聊天框。

## 功能

* **实时语音转录:** 使用 Dashscope 的 STT 服务（Paraformer 或 Gummy）将你的麦克风输入转换为文本。
* **实时翻译:** （可选，使用 Gummy 模型时）将你的语音翻译成指定语言。
* **VRChat OSC 集成:** 将转录或翻译的文本发送到 VRChat 聊天框。
* **LLM 集成:** （可选）将转录的文本发送给兼容 OpenAI API 的 LLM 进行处理，并将结果发送到 VRChat。
* **灵活配置:** 通过 `config.yaml` 文件自定义 API 密钥、模型、语言和其他行为。
* **多输出目标:** 除了 VRChat，还可以将结果输出到控制台和文件。

## 设置与安装

1. **克隆仓库:**
   ```bash
   git clone https://github.com/your-username/VRCMeow.git
   cd VRCMeow
   ```
2. **创建虚拟环境 (推荐):**
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
3. **安装依赖:**
    * 确保你安装了 [PortAudio](http://www.portaudio.com/download.html) (`sounddevice` 依赖)。在某些系统上可能需要手动安装。
    * 使用 `uv` 安装/同步 Python 依赖 (依赖项在 `pyproject.toml` 中定义):
      ```bash
      uv sync
      ```

## 配置

1. **复制配置文件:** 将项目根目录下的 `config.example.yaml` 文件复制一份，并重命名为 `config.yaml`。
2. **编辑 `config.yaml`:** 根据你的需求修改 `config.yaml` 文件。以下是主要配置项的说明：

    * **`dashscope` 部分 (必需):**
        * `api_key`: **必需。** 填入你的 Dashscope API 密钥。**强烈建议**通过设置环境变量 `DASHSCOPE_API_KEY`
          来提供密钥，这样更安全。如果在 `config.yaml` 中设置了 `api_key`，程序会使用它，但环境变量的优先级更高。
        * **`stt` 子部分 (必需):**
            * `translation_target_language`: (可选) 设置翻译的目标语言代码，例如 `"en"` (英语), `"ja"` (日语), `"ko"` (
              韩语)。如果留空 (`""`) 或设置为 `null`，则禁用翻译功能，只进行语音识别。**注意：** 确保你选择的 `model`
              支持你设置的目标语言。
            * `model`: **必需。** 选择要使用的 Dashscope 实时语音模型。
                * `"gummy-realtime-v1"`: 支持实时语音识别和翻译。
                * `"paraformer-realtime-v2"`, `"paraformer-realtime-v1"`: 仅支持实时语音识别。
                * 请查阅 Dashscope 文档获取最新的模型列表。
            * `intermediate_result_behavior`: (可选) 配置如何处理 STT 引擎返回的中间（非最终）识别结果，这主要影响 VRChat
              OSC 输出。
                * `"ignore"`: (默认) 忽略中间结果。
                * `"show_typing"`: 在 VRChat 中显示 "正在输入..." 状态。
                * `"show_partial"`: 在 VRChat 中显示部分识别的文本。

    * **`audio` 部分 (可选):**
        * `device`: (可选) 指定要使用的麦克风设备名称。你可以在应用程序启动后的 "仪表盘" 选项卡中看到可用的设备列表。设置为
          `"Default"` 或留空 (`null`) 以使用系统默认输入设备。
        * `sample_rate`: (可选) 音频采样率 (Hz)。设置为 `null` (默认) 会尝试自动检测设备采样率，失败则回退到 16000 Hz。*
          *注意：** 确保采样率与所选 STT 模型兼容 (例如 Gummy 通常需要 16000 Hz)。
        * `channels`: (可选) 音频通道数。Gummy 模型通常需要 `1` (单声道)。
        * `dtype`: (可选) 音频数据类型。Gummy 模型通常需要 `"int16"`。
        * `debug_echo_mode`: (可选) 设置为 `true` 以将麦克风输入直接回显到扬声器，用于测试音频输入是否正常。**警告：**
          可能产生啸叫，请谨慎使用。

    * **`llm` 部分 (可选):** 用于启用和配置大型语言模型处理。
        * `enabled`: 设置为 `true` 以启用 LLM 处理。如果为 `false` (默认)，则忽略此部分的其他设置。
        * `api_key`: **必需 (如果 `enabled` 为 `true`)。** 填入你的 OpenAI 兼容服务的 API 密钥。**强烈建议**通过设置环境变量
          `OPENAI_API_KEY` 来提供。环境变量优先于配置文件中的设置。
        * `base_url`: (可选) 指定 OpenAI 兼容 API 的端点 URL。对于 OpenAI 官方 API，通常无需设置。如果你使用本地运行的
          LLM (例如 Ollama 的 OpenAI 兼容接口，通常是 `"http://localhost:11434/v1"`) 或 API 代理服务，则需要设置此项。
        * `model`: **必需 (如果 `enabled` 为 `true`)。** 指定要使用的 LLM 模型名称 (例如 `"gpt-3.5-turbo"`, `"gpt-4o"`,
          或本地模型的名称如 `"llama3"`)。
        * `system_prompt`: **必需 (如果 `enabled` 为 `true`)。** 定义给 LLM 的指令，告诉它如何处理输入的文本 (
          例如，进行风格转换、总结等)。
        * `temperature`: (可选) 控制 LLM 输出的随机性 (0.0 到 2.0 之间，默认 0.7)。较低的值更确定，较高的值更有创意。
        * `max_tokens`: (可选) 限制 LLM 生成响应的最大长度 (以 token 计，大致等于单词或音节数，默认 256)。
        * `few_shot_examples`: (可选) 提供一个包含输入 (`user`) 和期望输出 (`assistant`) 的示例列表，以更精确地指导 LLM
          的行为，特别适用于风格模仿或特定格式转换。

    * **`outputs` 部分 (必需):** 配置处理结果的输出目的地。
        * **`vrc_osc` 子部分 (可选):** 用于将结果发送到 VRChat。
            * `enabled`: 设置为 `true` (默认) 以启用 VRChat OSC 输出。
            * `address`: **必需。** VRChat 客户端的 IP 地址 (通常是 `"127.0.0.1"`)。
            * `port`: **必需。** VRChat OSC 输入端口 (默认 `9000`)。
            * `message_interval`: (可选) 发送消息之间的最小间隔时间 (秒，默认 1.333)。设置太低可能导致 VRChat 速率限制。
            * `format`: (可选) 定义发送到 VRChat 聊天框的消息格式。`{text}` 会被替换为最终文本。示例: `"{text}"` (默认),
              `"Meow: {text}"`。
            * `send_immediately`: (可选) 设置为 `true` (默认) 直接发送消息，设置为 `false` 则将文本填充到输入框。
            * `play_notification_sound`: (可选) 设置为 `true` (默认) 在 VRChat 中播放消息提示音 (仅当 `send_immediately`
              为 `true` 时有效)。
        * **`console` 子部分 (可选):** 用于在控制台打印结果。
            * `enabled`: 设置为 `true` (默认) 以在运行 VRCMeow 的控制台中打印最终文本。
            * `prefix`: (可选) 为控制台输出的每行文本添加前缀 (例如 `"[VRCMeow Output]"`).
        * **`file` 子部分 (可选):** 用于将结果记录到文件。
            * `enabled`: 设置为 `true` 以将最终文本附加到文件。默认为 `false`。
            * `path`: **必需 (如果 `enabled` 为 `true`)。** 指定输出文件的路径 (例如 `"vrcmeow_output.log"`)。
            * `format`: (可选) 定义文件中每行的格式。可用占位符：`{timestamp}`, `{text}`。示例: `"{timestamp} - {text}"` (
              默认)。

    * **`logging` 部分 (可选):** 配置应用程序本身的日志记录行为。
        * `level`: 设置日志记录的详细程度。可选值 (从最详细到最不详细): `"DEBUG"`, `"INFO"` (默认), `"WARNING"`,
          `"ERROR"`, `"CRITICAL"`。
        * **`file` 子部分 (可选):** 用于将应用程序运行日志保存到文件。
            * `enabled`: 设置为 `true` (默认) 以启用日志文件记录。这对于排查问题很有用。
            * `path`: **必需 (如果 `enabled` 为 `true`)。** 指定保存应用程序运行日志的文件路径 (例如
              `"vrcmeow_app.log"`)。

## 用法

1. 确保你的 VRChat 客户端已启用 OSC (Settings -> OSC -> Enable OSC)。
2. 激活你的 Python 虚拟环境（如果需要手动激活，请参见安装步骤）。
3. 运行主脚本：
   ```bash
   # 如果已激活虚拟环境
   python main.py
   # 或者使用 uv run (无需手动激活环境)
   # uv run python main.py
   ```
4. 程序将开始监听你的默认麦克风。当你说话时，它会进行转录（和翻译，如果已配置），通过 LLM
   处理（如果已配置），并将最终结果发送到配置的输出目标（VRChat、控制台、文件）。
5. 按 `Ctrl+C` 停止程序。

## 贡献

欢迎贡献！如果你发现任何错误或有改进建议，请随时创建 Issue 或提交 Pull Request。
