# Explain Video To Me (YouTube & Bilibili)

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Have a long technical video but no time to watch it? This enhanced LLM application transforms long YouTube or Bilibili videos into structured, high-quality technical blogs with accurate timestamps and slide-based images.

## Core Features

- **YouTube & Bilibili Support**: Seamlessly download and process videos from both platforms.
- **Multimodal Topic Detection**:
  - **Description Parsing**: Automatically extracts outlines and timestamps from video descriptions.
  - **Visual Change Detection**: Identifies PPT slide transitions or visual topic changes (every 5 seconds).
  - **LLM Semantic Analysis**: Analyzes transcript flow to detect technical topic shifts.
- **Precise Timestamps**: Uses fuzzy matching against `.srt` files to recover exact-to-the-second timestamps for every section.
- **Adaptive Image Inclusion**: Automatically titles and inserts screenshots of PPT slides based on visual changes.
- **Detailed Technical Blogs**: Generates extremely detailed, textbook-quality HTML blogs in the video's original language.
- **OpenRouter Integration**: Powered by high-end reasoning models (like Gemini 3 Pro) via OpenRouter for deep technical insight.

## Design

Built with [Pocket Flow](https://github.com/The-Pocket/PocketFlow), a lightweight LLM framework that orchestrates complex workflows through nodes.

- **ProcessVideoNode**: Handles video downloading, frame extraction, and subtitle recovery.
- **GenerateBlogNode**: Orchestrates the multi-modal outline merging and final LLM blog generation.

## How to Run

1. **Prerequisites**: Ensure you have `ffmpeg` installed on your system.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure API**: The system is pre-configured with OpenRouter. You can customize the model or key in `utils/call_llm.py`.
4. **Run**:
   ```bash
   python main.py --url "YOUR_VIDEO_URL"
   ```
5. **Output**: Open `output.html` to view the generated blog. The full transcript and structured outline are also saved to `transcript.txt`.

## Local Assets & Cache

The program is optimized for performance:
- **Video Cache**: Existing `temp_video.mp4` files are reused to skip redownloading.
- **Subtitle Cache**: `.srt` files are preserved for precise timestamp backtracking.
- **Frames**: Visual snapshots are stored in the `frames/` directory.

---
Built with ❤️ using [Pocket Flow](https://github.com/The-Pocket/PocketFlow).
