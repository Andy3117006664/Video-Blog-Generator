# Explain Video To Me（YouTube & Bilibili）

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

[English README](README.md)

有一个很长的技术视频但没时间看？这个增强版 LLM 应用可以把 YouTube 或 Bilibili 的长视频，转换成带结构化大纲、准确时间戳、基于幻灯片截图的高质量技术博客（HTML）。

## 核心功能与实现位置

1. **LLM 与 API 集成**
   - **OpenRouter 适配与模型配置**
     - 实现位置：`utils/call_llm.py`
     - 细节：将 API 调用适配到 OpenRouter，默认使用 `google/gemini-3-pro-preview`，并启用推理能力（`reasoning: True`）。你可以在该文件中自定义模型与 Key 配置。

2. **视频与字幕处理（针对 Bilibili 优化）**
   - **Bilibili 支持与清晰度控制**
     - 实现位置：`utils/video_processor.py` → `get_video_info`
     - 细节：通过 `yt-dlp` 解析 Bilibili，配置下载参数优先/强制 720p，并支持 `cookies.txt` 登录以访问更高质量内容。
   - **鲁棒的字幕解析（SRT / BCC JSON）**
     - 实现位置：`utils/video_processor.py` → `parse_srt` 以及 `get_video_info` 内字幕解析逻辑
     - 细节：支持 Bilibili 特有的 BCC JSON 格式解析，并增强 SRT 正则以兼容不同平台的换行符差异。

3. **多模态深度大纲（STRUCTURED OUTLINE）**
   
   为了提升圆桌会议、多人对谈等复杂场景下的章节识别质量，实现了三条路径融合的大纲提取方案：
   - **路径 A：简介解析**
     - 实现位置：`utils/video_processor.py` → `parse_outline_from_description`
     - 细节：自动抓取视频简介中带时间戳的章节信息。
   - **路径 B：语义 Shift 识别**
     - 实现位置：`utils/video_processor.py` → `detect_topic_shifts_with_llm`
     - 细节：调用 LLM 分析字幕流，识别话题转折并给出“锚点语句”（Anchor Quote）。
   - **路径 C：视觉补全（PPT 翻页检测）**
     - 实现位置：`utils/video_processor.py` → `extract_frames`（5 秒步长）以及 `get_video_info` 的合并逻辑
     - 细节：当视觉上 PPT 标题变动但文本未能识别到话题时，系统会自动捕捉变化点并请求 LLM 批量冠名补全章节标题。

4. **精确时间戳回溯（Timestamp Backtracking）**
   - 功能：解决大纲中出现 `Unknown` 的问题
   - 实现位置：`utils/video_processor.py` → `get_precise_timestamp`
   - 细节：利用 LLM 提供的锚点语句在原始 `.srt` 中进行模糊匹配，找回精确到秒的开始时间；若匹配失败，则使用 LLM 预估的 `approx_timestamp` 兜底。

5. **性能与稳定性优化**
   - **跳过重复下载**
     - 实现位置：`utils/video_processor.py` → `find_existing_video`
     - 细节：识别本地已有的 `temp_video.mp4` 或分离视频流并复用，节省下载时间与带宽。
   - **批量 LLM 请求**
     - 实现位置：`utils/video_processor.py` → `get_video_info`（视觉话题冠名部分）
     - 细节：将多个视觉跳变点的命名任务合并为一个 JSON 列表发送给 LLM，减少网络往返与 API 消耗。
   - **保留字幕资产**
     - 实现位置：`utils/video_processor.py` 末尾清理逻辑
     - 细节：保留 `.srt` / `.vtt` 文件，作为后续时间戳回溯的持久参考。

6. **博客生成逻辑（Flow & Prompt）**
   - **语言自适应与细粒度控制**
     - 实现位置：`flow.py` → `GenerateBlogNode`
     - 细节：Prompt 强制要求与视频语言一致（中文视频输出中文博客），并严格遵循 `STRUCTURED OUTLINE` 分段，生成极其详细（Extremely detailed）的技术内容。
   - **图像数量提升/自适应**
     - 实现位置：`main.py` → `shared` 配置
     - 细节：最初配置 `image_count` 为 18–25，后演进为依据视觉变化自适应调整。

得益于本地视频与字幕资产的保留，即使遇到 OpenRouter 偶发连接中断（例如 `Connection broken`），也可以直接重跑：系统会跳过耗时的下载/抽帧步骤，优先复用本地缓存并重试生成流程。

## 设计

基于 [Pocket Flow](https://github.com/The-Pocket/PocketFlow) 编排多节点工作流：

- **ProcessVideoNode**：负责视频下载、抽帧、字幕解析与时间戳回溯。
- **GenerateBlogNode**：负责多模态大纲融合与最终博客生成。

## 运行方式

1. **前置依赖**：确保系统已安装 `ffmpeg`。
2. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```
3. **配置 LLM**：项目默认使用 OpenRouter；如需切换模型或配置 Key，请查看 `utils/call_llm.py`。
4. **运行**：
   ```bash
   python main.py --url "YOUR_VIDEO_URL"
   ```
5. **输出**：生成的博客在 `output.html`；完整字幕与大纲等中间产物也会落盘（例如 `transcript.txt`）。

## 本地缓存与资产

- **视频缓存**：复用已有 `temp_video.mp4`，跳过重复下载。
- **字幕缓存**：保留 `.srt` 用于精确时间戳回溯。
- **截图帧**：保存在 `frames/` 目录。

---

Built with ❤️ using [Pocket Flow](https://github.com/The-Pocket/PocketFlow).
