import re
import cv2
import os
import requests
import yt_dlp
import logging
import glob
import json
import difflib
from datetime import timedelta
from youtube_transcript_api import YouTubeTranscriptApi
from utils.call_llm import call_llm

logger = logging.getLogger(__name__)

def normalize_text(x: str) -> str:
    """Normalize text for fuzzy matching."""
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"[，。！？、,.!?;:：；（）()“”\"'’\-—]", "", x)
    return x.strip().lower()

def parse_srt(path: str):
    """Parse SRT file into a list of (start_sec, text)."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().replace("\r\n", "\n").strip()
        
        blocks = re.split(r"\n\n+", content)
        items = []
        for b in blocks:
            lines = [x.strip() for x in b.split("\n") if x.strip()]
            if len(lines) < 2:
                continue
            
            # Find the time line (usually the 2nd line, but can be 1st if index is missing)
            time_line = ""
            text_start_idx = 0
            for i, line in enumerate(lines):
                if "-->" in line:
                    time_line = line
                    text_start_idx = i + 1
                    break
            
            if not time_line:
                continue
                
            m = re.search(r"(\d{2}:\d{2}:\d{2}),\d{3}\s*-->", time_line)
            if not m:
                continue
            
            ts = m.group(1)
            h, m2, s2 = [int(x) for x in ts.split(":")]
            start_sec = h * 3600 + m2 * 60 + s2
            text = " ".join(lines[text_start_idx:])
            items.append((start_sec, text))
        return items
    except Exception as e:
        logger.error(f"Error parsing SRT {path}: {e}")
        return []

def get_precise_timestamp(anchor_quote, srt_items, min_ratio=0.6):
    """Find the precise timestamp of an anchor quote in SRT items."""
    aq = normalize_text(anchor_quote)
    if not aq or not srt_items:
        return None

    # 1. Exact inclusion match
    for sec, txt in srt_items:
        if aq in normalize_text(txt):
            return sec

    # 2. Fuzzy match
    best = (0.0, None)
    for sec, txt in srt_items:
        t = normalize_text(txt)
        r = difflib.SequenceMatcher(None, aq, t).ratio()
        if r > best[0]:
            best = (r, sec)
    
    return best[1] if best[0] >= min_ratio else None

def parse_outline_from_description(description: str):
    """Parse timestamps and titles from video description."""
    lines = [l.strip() for l in (description or "").splitlines()]
    
    # Try to find an outline start
    start_idx = 0
    found_marker = False
    for i, l in enumerate(lines):
        if re.search(r"\bOutline\b|目录|大纲|时间轴|Timestamps", l, re.IGNORECASE):
            start_idx = i + 1
            found_marker = True
            break
    
    # If no marker, just try all lines (some descriptions start with timestamps)
    search_lines = lines[start_idx:] if found_marker else lines
    
    items = []
    # Pattern: 0:00, 00:00, 1:00:00 followed by text
    pat = re.compile(r"^(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\s+[-–—:]?\s*(?P<title>.+?)\s*$")
    
    for l in search_lines:
        m = pat.match(l)
        if m:
            ts_str = m.group("ts")
            parts = [int(p) for p in ts_str.split(":")]
            if len(parts) == 2:
                h, m_val, s = 0, parts[0], parts[1]
            else:
                h, m_val, s = parts[0], parts[1], parts[2]
            
            clean_ts = f"{h:02d}:{m_val:02d}:{s:02d}"
            items.append({"timestamp": clean_ts, "title": m.group("title")})
        elif items and not l.strip():
            # Stop at first empty line after finding some items if we found a marker
            if found_marker: break
            
    return items

def detect_topic_shifts_with_llm(transcript_text, title):
    """Use LLM to detect topic shifts from transcript and return anchor quotes."""
    prompt = f"""
Analyze the following video transcript for topic shifts. 
Identify the most important technical transitions or new subject matters.
For each shift, provide:
1. A concise section title.
2. An 'anchor_quote': A unique short sentence or phrase (10-20 words) from the transcript that EXACTLY marks the beginning of this new topic.
3. An 'approx_timestamp': The estimated time (HH:MM:SS) where this topic begins based on the provided block timestamps.

Return ONLY a valid JSON list of objects:
[
  {{
    "title": "Section Title", 
    "anchor_quote": "Exact quote from transcript...",
    "approx_timestamp": "HH:MM:SS"
  }},
  ...
]

VIDEO TITLE: {title}

TRANSCRIPT:
{transcript_text[:20000]} # Increased limit to see more of the video
"""
    try:
        logger.info("Calling LLM to detect topic shifts...")
        response = call_llm(prompt)
        # Clean response from potential markdown backticks
        response = re.sub(r"```json\s*|\s*```", "", response).strip()
        shifts = json.loads(response)
        return shifts
    except Exception as e:
        logger.error(f"Error detecting topic shifts with LLM: {e}")
        return []

def extract_frames(video_path, output_dir, sensitivity=0.15):
    """Extract frames from video based on visual difference (PPT slide changes)."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        logger.error("Could not get FPS for video")
        return []
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_secs = int(total_frames / fps)
    
    frame_info = []
    last_gray = None
    
    # Check one frame every 5 seconds (as requested by user)
    step = 5
    for sec in range(0, duration_secs, step):
        if sec % 600 == 0: # Log progress every 10 minutes of video
            logger.info(f"Processing video frames: {sec // 60} / {duration_secs // 60} minutes...")
            
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale and blur for stability
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if last_gray is not None:
            # Calculate frame difference
            frame_delta = cv2.absdiff(last_gray, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            # Calculate ratio of changed pixels
            change_ratio = cv2.countNonZero(thresh) / float(gray.shape[0] * gray.shape[1])
            
            # If change exceeds sensitivity, save new frame (new PPT slide)
            if change_ratio > sensitivity:
                timestamp_str = str(timedelta(seconds=sec)).split('.')[0].zfill(8)
                filename = f"{timestamp_str.replace(':', '_')}.jpg"
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, frame)
                frame_info.append({"timestamp": timestamp_str, "path": f"frames/{filename}"})
                last_gray = gray
        else:
            # First frame always saved
            timestamp_str = "00:00:00"
            filename = "00_00_00.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            frame_info.append({"timestamp": timestamp_str, "path": f"frames/{filename}"})
            last_gray = gray
            
    cap.release()
    return frame_info

def find_existing_video():
    """Robustly find any existing video file that can be used."""
    # 1. Preferred: merged mp4
    if os.path.exists("temp_video.mp4"):
        return "temp_video.mp4"
    # 2. Falling back to video stream files from yt-dlp
    cands = sorted(glob.glob("temp_video.f*.mp4"))
    if cands:
        return cands[0]
    return None

def get_video_id(url):
    """Extract video ID from YouTube or Bilibili URL."""
    if "youtube.com" in url or "youtu.be" in url:
        pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11})'
        match = re.search(pattern, url)
        return match.group(1) if match else None
    elif "bilibili.com" in url or "b23.tv" in url:
        pattern = r'(BV[0-9A-Za-z]{10})'
        match = re.search(pattern, url)
        return match.group(1) if match else None
    return None

def get_video_info(url):
    """Download video, extract transcript and frames using text-based outline."""
    video_id = get_video_id(url)
    if not video_id:
        return {"error": "Unsupported or invalid video URL"}

    ydl_opts = {
        'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]',
        'outtmpl': 'temp_video.%(ext)s',
        'skip_download': False,
        'quiet': False,
        'no_warnings': False,
        'cookiefile': 'cookies.txt',
        'merge_output_format': 'mp4',
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['zh-Hans', 'zh-CN', 'zh', 'en', 'all'],
        'retries': 10,
        'fragment_retries': 10,
        'nocheckcertificate': True,
        'socket_timeout': 30,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    }
    
    try:
        video_path = find_existing_video()
        if video_path:
            logger.info(f"Using existing video file: {video_path}")
            
            # 0. Check if subtitles exist, if not, re-fetch them without downloading video
            base_path = os.path.splitext(video_path)[0]
            potential_subs = [f for f in os.listdir('.') if f.startswith(os.path.basename(base_path)) and f.endswith(('.srt', '.vtt'))]
            
            if not potential_subs:
                logger.info("Video exists but subtitles missing. Fetching subtitles only...")
                sub_ydl_opts = ydl_opts.copy()
                sub_ydl_opts['skip_download'] = True
                with yt_dlp.YoutubeDL(sub_ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True) # download=True with skip_download=True only gets subs
            else:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
        else:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_path = ydl.prepare_filename(info)
                if not os.path.exists(video_path):
                    base, _ = os.path.splitext(video_path)
                    if os.path.exists(base + ".mp4"):
                        video_path = base + ".mp4"
        
        title = info.get('title', 'Video')
        description = info.get('description', '')
        
        # 1. Transcript extraction
        transcript_text = ""
        srt_path = ""
        base_path = os.path.splitext(video_path)[0]
        potential_subs = [f for f in os.listdir('.') if f.startswith(os.path.basename(base_path)) and f.endswith(('.srt', '.vtt'))]
        
        if potential_subs:
            chosen_sub = None
            for preferred in ['ai-zh', 'zh-Hans', 'zh-CN', 'zh', 'en']:
                for ps in potential_subs:
                    if preferred in ps:
                        chosen_sub = ps
                        break
                if chosen_sub: break
            
            if not chosen_sub: chosen_sub = potential_subs[0]
            if chosen_sub.endswith('.srt'): srt_path = chosen_sub
            
            try:
                with open(chosen_sub, 'r', encoding='utf-8') as f:
                    raw_content = f.read()
                    if chosen_sub.endswith('.srt'):
                        clean_raw = raw_content.replace('\r\n', '\n')
                        segments = re.findall(r'\d+\n(\d{2}:\d{2}:\d{2}),\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n(.*?)(?=\n\n|\n\d+\n|$)', clean_raw, re.DOTALL)
                        
                        blocks = []
                        current_block_texts = []
                        last_block_id = -1
                        first_ts = ""
                        for ts_str, text in segments:
                            h, m, s = map(int, ts_str.split(':'))
                            block_id = (h * 3600 + m * 60 + s) // 600 # 10 min
                            clean_text = text.replace('\n', ' ').strip()
                            if not clean_text: continue
                            if block_id != last_block_id:
                                if current_block_texts:
                                    blocks.append(f"[{first_ts}] " + " ".join(current_block_texts))
                                current_block_texts = [clean_text]
                                first_ts, last_block_id = ts_str, block_id
                            else:
                                current_block_texts.append(clean_text)
                        if current_block_texts:
                            blocks.append(f"[{first_ts}] " + " ".join(current_block_texts))
                        transcript_text = "\n\n".join(blocks)
                    else:
                        vtt_lines = raw_content.split('\n')
                        transcript_text = "\n".join([l for l in vtt_lines if '-->' not in l and l.strip() and not l.strip().isdigit() and l != 'WEBVTT'])
            except Exception as e:
                logger.error(f"Error reading sub: {e}")

        # Fallback to description if no transcript
        if not transcript_text:
            description = info.get('description', '')
            transcript_text = f"No transcript found. Using description:\n{description}"

        # 2. Outline / Topic Shift detection
        outline = parse_outline_from_description(description)
        if not outline:
            logger.info("No outline in description, detecting via LLM...")
            shifts = detect_topic_shifts_with_llm(transcript_text, title)
            if shifts:
                srt_items = parse_srt(srt_path) if srt_path else []
                for s in shifts:
                    precise_sec = get_precise_timestamp(s['anchor_quote'], srt_items)
                    if precise_sec is not None:
                        s['timestamp'] = str(timedelta(seconds=precise_sec)).zfill(8)
                    else:
                        # Use the LLM's estimated timestamp as fallback if anchor quote fails
                        s['timestamp'] = s.get('approx_timestamp', "Unknown")
                outline = shifts

        # 2.1 Multimodal enhancement: Supplement with visual topic shifts (PPT changes)
        # If the visual change occurs far from any existing outline item, we title it and add it.
        frames = extract_frames(video_path, "frames")
        if frames and outline:
            logger.info("Merging visual topic shifts into the outline...")
            
            # Helper to convert HH:MM:SS to seconds
            def ts_to_sec(ts):
                try:
                    h, m, s = map(int, ts.split(':'))
                    return h * 3600 + m * 60 + s
                except: return 0

            existing_secs = [ts_to_sec(o['timestamp']) for o in outline if o.get('timestamp') != "Unknown"]
            
            candidates = []
            for f in frames:
                f_sec = ts_to_sec(f['timestamp'])
                # If this visual frame is more than 2 minutes away from any outline point, it might be a missed topic (like panel)
                if not any(abs(f_sec - es) < 120 for es in existing_secs):
                    # Extract a snippet around this frame to help LLM title it
                    # Get surrounding transcript from the raw segments (approximate)
                    snippet = ""
                    if srt_path:
                        srt_items = parse_srt(srt_path)
                        nearby = [txt for sec, txt in srt_items if abs(sec - f_sec) < 60]
                        snippet = " ".join(nearby)[:500]
                    
                    if snippet:
                        candidates.append({"timestamp": f['timestamp'], "snippet": snippet})
                        existing_secs.append(f_sec) # Don't add multiple visual hits for the same section
            
            if candidates:
                logger.info(f"Titling {len(candidates)} visual topic shifts in batch...")
                # Batch ask LLM to title each boundary using nearby transcript snippet
                # Limit to first 20 candidates to avoid context window issues
                batch = candidates[:20]
                prompt = "You are given a list of timestamps where the video visual changed (likely a new PPT slide or topic title).\n"
                prompt += "For each item, generate a concise section title in the SAME LANGUAGE as the snippet.\n"
                prompt += "Return ONLY a valid JSON list of objects: [{\"timestamp\": \"HH:MM:SS\", \"title\": \"...\"}, ...]\n\n"
                prompt += json.dumps(batch, ensure_ascii=False)
                
                try:
                    resp = call_llm(prompt)
                    # Clean response from potential markdown backticks
                    resp = re.sub(r"```json\s*|\s*```", "", resp).strip()
                    new_visual_items = json.loads(resp)
                    if new_visual_items:
                        outline.extend(new_visual_items)
                        outline = sorted(outline, key=lambda x: ts_to_sec(x['timestamp']))
                except Exception as e:
                    logger.error(f"Failed to title visual boundaries in batch: {e}")

        # 3. Final construction for LLM
        
        transcript_with_name = f"VIDEO TITLE: {title}\n\n"
        transcript_with_name += f"VIDEO DESCRIPTION:\n{description}\n\n"
        
        if outline:
            transcript_with_name += "STRUCTURED OUTLINE:\n"
            for item in outline:
                ts = item.get('timestamp', 'N/A')
                transcript_with_name += f"- [{ts}] {item.get('title', 'New Section')}\n"
            transcript_with_name += "\n"

        transcript_with_name += "AVAILABLE SCREENSHOTS:\n"
        for f in frames:
            transcript_with_name += f"Timestamp: {f['timestamp']}, Image Path: {f['path']}\n"
        
        transcript_with_name += "\nFULL TRANSCRIPT (10-minute blocks):\n"
        transcript_with_name += transcript_text
        
        with open("transcript.txt", "w", encoding="utf-8") as f:
            content = f"VIDEO TITLE: {title}\n"
            content += f"VIDEO DESCRIPTION:\n{description}\n\n"
            if outline:
                content += "STRUCTURED OUTLINE:\n"
                for item in outline:
                    ts = item.get('timestamp', 'N/A')
                    content += f"- [{ts}] {item.get('title', 'New Section')}\n"
                content += "\n"
            content += "="*50 + "\n\nFULL TRANSCRIPT:\n" + transcript_text
            f.write(content)

        # Cleanup local subtitle files (Keep the video AND SRT!)
        base_path = os.path.splitext(video_path)[0]
        potential_subs = [f for f in os.listdir('.') if f.startswith(os.path.basename(base_path)) and f.endswith(('.srt', '.vtt', '.xml'))]
        for ps in potential_subs:
            try:
                # Keep SRT and VTT for future runs and analysis
                if ps.endswith(('.srt', '.vtt')):
                    continue
                os.remove(ps)
            except:
                pass
            
        return {
            "title": title,
            "transcript_with_name": transcript_with_name,
            "frames": frames,
            "url": url,
            "outline": outline
        }
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return {"error": str(e)}
