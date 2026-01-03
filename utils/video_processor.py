import re
import cv2
import os
import requests
import yt_dlp
import logging
import glob
import json
import difflib
import easyocr
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
    {transcript_text}
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

def extract_frames(video_path, output_dir, sensitivity=0.05):
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
    
    def ts_to_sec(ts):
        try:
            if not ts or ts == "Unknown": return 0
            parts = ts.split(':')
            if len(parts) == 3: h, m, s = map(int, parts)
            elif len(parts) == 2: h, m, s = 0, int(parts[0]), int(parts[1])
            else: return 0
            return h * 3600 + m * 60 + s
        except: return 0

    def load_frames_from_folder(frames_dir: str):
        if not os.path.isdir(frames_dir):
            return []
        pat = re.compile(r"^\d{2}_\d{2}_\d{2}\.jpg$")
        names = [n for n in os.listdir(frames_dir) if pat.match(n)]
        names.sort()
        return [{"timestamp": n[:-4].replace("_", ":"), "path": f"{frames_dir}/{n}"} for n in names]

    video_id = get_video_id(url)
    if not video_id:
        return {"error": "Unsupported or invalid video URL"}

    cached_frames = load_frames_from_folder("frames")
    if cached_frames and os.path.exists("transcript.txt"):
        try:
            with open("transcript.txt", "r", encoding="utf-8") as f:
                transcript_with_name = f.read()
        except Exception as e:
            logger.warning(f"Failed to read transcript.txt cache: {e}")
            transcript_with_name = ""

        title = "Video"
        if transcript_with_name:
            first_line = transcript_with_name.splitlines()[0].strip()
            if first_line.lower().startswith("video title:"):
                title = first_line.split(":", 1)[1].strip() or title

        return {
            "title": title,
            "transcript_with_name": transcript_with_name,
            "frames": cached_frames,
            "url": url,
            "outline": [],
            "enhanced_segments": []
        }

    ydl_opts = {
        'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]/best',
        'outtmpl': 'temp_video.%(ext)s',
        'skip_download': False,
        'quiet': False,
        'no_warnings': False,
        'cookiefile': 'cookies.txt',
        'merge_output_format': 'mp4',
        'writesubtitles': False, # Download subtitles separately to avoid 429 killing video download
        'writeautomaticsub': False,
        'subtitleslangs': ['zh-Hans', 'en'],
        'retries': 10,
        'fragment_retries': 10,
        'nocheckcertificate': True,
        'socket_timeout': 30,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    }
    
    info = {}
    video_path = None
    try:
        video_path = find_existing_video()
        if video_path:
            logger.info(f"Using existing video file: {video_path}")
            # Get info even if video exists
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
        else:
            if cached_frames:
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
        
        # Now try to get subtitles separately
        base_prefix = None
        if video_path:
            base_prefix = os.path.basename(os.path.splitext(video_path)[0])
        else:
            base_prefix = "temp_video"

        potential_subs = [f for f in os.listdir('.') if f.startswith(base_prefix) and f.endswith(('.srt', '.vtt'))]
        if not potential_subs:
            logger.info("Fetching subtitles...")
            sub_opts = ydl_opts.copy()
            sub_opts.update({'skip_download': True, 'writesubtitles': True, 'writeautomaticsub': True})
            try:
                with yt_dlp.YoutubeDL(sub_opts) as ydl:
                    ydl.download([url])
            except Exception as sub_e:
                logger.warning(f"Subtitles download failed (likely 429): {sub_e}")
    except Exception as e:
        logger.warning(f"yt-dlp download/info extraction failed: {e}. Trying to proceed with whatever is available...")
    
    try:
        title = info.get('title', 'Video')
        description = info.get('description', '')
        
        # 1. Transcript extraction
        transcript_text = ""
        srt_path = ""
        base_prefix = None
        if video_path:
            base_prefix = os.path.basename(os.path.splitext(video_path)[0])
        else:
            base_prefix = "temp_video"
        potential_subs = [f for f in os.listdir('.') if f.startswith(base_prefix) and f.endswith(('.srt', '.vtt'))]
        
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

        # Fallback to YouTubeTranscriptApi if yt-dlp failed
        if not transcript_text and ("youtube.com" in url or "youtu.be" in url):
            logger.info("yt-dlp failed to get subtitles. Trying YouTubeTranscriptApi...")
            try:
                # Use YouTubeTranscriptApi class directly if possible, or handle instance
                try:
                    ts_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['zh-Hans', 'zh-CN', 'zh', 'en'])
                except AttributeError:
                    # Some versions might require instantiation or have different naming
                    ts_list = YouTubeTranscriptApi().get_transcript(video_id, languages=['zh-Hans', 'zh-CN', 'zh', 'en'])
                
                blocks = []
                for entry in ts_list:
                    start = int(entry['start'])
                    ts_str = str(timedelta(seconds=start)).split('.')[0].zfill(8)
                    blocks.append(f"[{ts_str}] {entry['text']}")
                transcript_text = "\n".join(blocks)
                logger.info("Successfully retrieved transcript via YouTubeTranscriptApi.")
            except Exception as e:
                logger.warning(f"YouTubeTranscriptApi also failed: {e}")

        # Fallback to description if no transcript
        if not transcript_text:
            description = info.get('description', '')
            transcript_text = f"No transcript found. Using description:\n{description}"

        # 2. Outline / Topic Shift detection
        outline = parse_outline_from_description(description)
        
        logger.info("Detecting additional topic shifts via LLM to refine outline...")
        llm_shifts = detect_topic_shifts_with_llm(transcript_text, title)
        if llm_shifts:
            srt_items = parse_srt(srt_path) if srt_path else []
            for s in llm_shifts:
                precise_sec = get_precise_timestamp(s['anchor_quote'], srt_items)
                if precise_sec is not None:
                    s['timestamp'] = str(timedelta(seconds=precise_sec)).zfill(8)
                else:
                    # Use the LLM's estimated timestamp as fallback if anchor quote fails
                    s['timestamp'] = s.get('approx_timestamp', "Unknown")
            
            if not outline:
                outline = llm_shifts
            else:
                # Merge: Supplement description outline with LLM-detected technical shifts
                existing_secs = [ts_to_sec(o['timestamp']) for o in outline if o.get('timestamp') != "Unknown"]
                for s in llm_shifts:
                    s_sec = ts_to_sec(s['timestamp'])
                    # If this LLM shift is not too close to any existing description item (3 mins), add it
                    if not any(abs(s_sec - es) < 180 for es in existing_secs):
                        outline.append(s)
                outline = sorted(outline, key=lambda x: ts_to_sec(x['timestamp']))

        # 2.1 Multimodal enhancement: Supplement with visual topic shifts (PPT changes)
        # If the visual change occurs far from any existing outline item, we title it and add it.
        frames = cached_frames if cached_frames else (extract_frames(video_path, "frames") if video_path else [])
        if frames and outline:
            logger.info("Merging visual topic shifts into the outline...")
            
            existing_secs = [ts_to_sec(o['timestamp']) for o in outline if o.get('timestamp') != "Unknown"]
            
            candidates = []
            for f in frames:
                f_sec = ts_to_sec(f['timestamp'])
                # If this visual frame is more than 1 minute away from any outline point, it might be a missed topic (like panel)
                if not any(abs(f_sec - es) < 60 for es in existing_secs):
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
                
                # Sort candidates by time to ensure sampling is chronological
                candidates = sorted(candidates, key=lambda x: ts_to_sec(x['timestamp']))
                
                # Pick candidates covering the whole timeline (head, tail, and middle)
                def pick_diverse(items, k):
                    n = len(items)
                    if n <= k: return items
                    # Include head, tail, and evenly spaced middle items
                    head_count = max(1, k // 4)
                    tail_count = max(1, k // 4)
                    mid_count = k - head_count - tail_count
                    
                    picked_indices = set()
                    # Head
                    for i in range(head_count): picked_indices.add(i)
                    # Tail
                    for i in range(n - tail_count, n): picked_indices.add(i)
                    # Middle
                    if mid_count > 0:
                        step = (n - head_count - tail_count) / (mid_count + 1)
                        for i in range(1, mid_count + 1):
                            idx = int(head_count + i * step)
                            if idx < n: picked_indices.add(idx)
                    
                    return [items[i] for i in sorted(list(picked_indices))]

                # Use a larger total limit (e.g., 40) but process in smaller chunks
                diverse_candidates = pick_diverse(candidates, 40)
                
                all_new_visual_items = []
                chunk_size = 10
                for i in range(0, len(diverse_candidates), chunk_size):
                    chunk = diverse_candidates[i:i + chunk_size]
                    logger.info(f"Processing visual titles chunk {i//chunk_size + 1}...")
                    
                    prompt = "You are given a list of timestamps where the video visual changed (likely a new PPT slide or topic title).\n"
                    prompt += "For each item, generate a concise section title in the SAME LANGUAGE as the snippet.\n"
                    prompt += "Return ONLY a valid JSON list of objects: [{\"timestamp\": \"HH:MM:SS\", \"title\": \"...\"}, ...]\n\n"
                    prompt += json.dumps(chunk, ensure_ascii=False)
                    
                    try:
                        resp = call_llm(prompt)
                        resp_clean = re.sub(r"```json\s*|\s*```", "", resp).strip()
                        new_items = json.loads(resp_clean)
                        if isinstance(new_items, list):
                            all_new_visual_items.extend(new_items)
                    except Exception as e:
                        logger.error(f"Failed to title visual boundaries chunk: {e}")

                if all_new_visual_items:
                    outline.extend(all_new_visual_items)
                    outline = sorted(outline, key=lambda x: ts_to_sec(x['timestamp']))

            # 2.2 OCR Enhancement for Panel Discussion (Directly using frames folder for tail topics)
            logger.info("Checking for panel discussion topics via OCR using frames folder...")
            
            # Use fixed start time for panel OCR as requested: 01:30:50
            PANEL_OCR_START = "01_30_50"
            
            def _is_time_jpg(name):
                return bool(re.match(r"^\d{2}_\d{2}_\d{2}\.jpg$", name))

            def _fname_to_ts(name):
                return name[:-4].replace("_", ":")

            try:
                # List all frames in the folder
                all_frame_files = [n for n in os.listdir("frames") if _is_time_jpg(n)]
                all_frame_files.sort()
                
                # Filter frames from 01_30_50 onwards
                target_frames = [n for n in all_frame_files if n >= f"{PANEL_OCR_START.replace(':', '_')}.jpg"]
                
                if target_frames:
                    logger.info(f"Performing OCR on {len(target_frames)} frames from {PANEL_OCR_START} onwards...")
                    reader = easyocr.Reader(['ch_sim','en'])
                    topics = {} # num -> (text, timestamp)
                    
                    # Patterns: 话题1: Title, 1. Title, Topic 1: Title
                    pat_topic = re.compile(r'(?:话题|Topic)\s*([0-9]{1,2})\s*[:：\.、]\s*(.+)', re.IGNORECASE)
                    pat_num = re.compile(r'^\s*([0-9]{1,2})\s*[\.\、:：]\s*(.+)\s*$')
                    
                    for fname in target_frames:
                        img_path = os.path.join("frames", fname)
                        ts = _fname_to_ts(fname)
                        
                        results = reader.readtext(img_path, detail=0)
                        text_joined = " ".join(results)
                        
                        # Optimization: only look for topics if specific keywords or number patterns are found
                        if not any(k in text_joined for k in ["话题", "Topic"]) and not re.search(r"\b\d+[\.\、:：]\b", text_joined):
                            continue
                            
                        for line in results:
                            line = line.strip()
                            m = pat_topic.search(line) or pat_num.match(line)
                            if m:
                                num = int(m.group(1))
                                title_text = m.group(2).strip()
                                if len(title_text) > 3:
                                    # Keep the longest title for a given number
                                    if num not in topics or len(title_text) > len(topics[num][0]):
                                        topics[num] = (title_text, ts)
                        
                        # If we've found enough topics (e.g. 6), we can stop
                        if len(topics) >= 6: break
                    
                    if topics:
                        logger.info(f"OCR extracted {len(topics)} topics from frames folder.")
                        for num in sorted(topics.keys()):
                            t_text, t_ts = topics[num]
                            # Fuzzy check duplicate
                            if not any(difflib.SequenceMatcher(None, t_text, o.get('title', '')).ratio() > 0.7 for o in outline):
                                outline.append({"timestamp": t_ts, "title": f"圆桌话题 {num}: {t_text}"})
                        
                        outline = sorted(outline, key=lambda x: ts_to_sec(x['timestamp']))
                    else:
                        logger.info("No panel topics extracted via OCR from frames folder.")
            except Exception as e:
                logger.error(f"OCR panel enhancement failed: {e}")

        # 3. Final construction for LLM: Guided Segments
        # This binds each outline item to its corresponding transcript text and image.
        
        logger.info("Constructing guided segments for LLM...")
        srt_items = parse_srt(srt_path) if srt_path else []
        enhanced_segments = []
        
        for i, item in enumerate(outline):
            start_sec = ts_to_sec(item.get('timestamp'))
            # End of this segment is the start of the next outline item
            end_sec = ts_to_sec(outline[i+1]['timestamp']) if i+1 < len(outline) else 999999
            
            # 1. Retrieve detailed transcript for this segment
            # We use the raw SRT items to get the most detailed text possible.
            segment_text = " ".join([txt for sec, txt in srt_items if start_sec <= sec < end_sec])
            
            # 2. Match the most relevant image frame
            # We look for a frame within a short window around the start timestamp.
            best_frame = None
            for f in frames:
                f_sec = ts_to_sec(f['timestamp'])
                # Prefer frame closest to start_sec, but allow some drift (e.g. 30s)
                if abs(f_sec - start_sec) < 30:
                    best_frame = f['path']
                    break
            
            enhanced_segments.append({
                "timestamp": item.get('timestamp', 'Unknown'),
                "title": item.get('title', 'New Section'),
                "image": best_frame,
                "content": segment_text.strip()
            })

        # Build the final prompt input: Guided structure
        transcript_with_name = f"VIDEO TITLE: {title}\n\n"
        transcript_with_name += f"VIDEO DESCRIPTION:\n{description}\n\n"
        
        transcript_with_name += "GUIDED BLOG STRUCTURE (STRICTLY FOLLOW THIS HIERARCHY):\n"
        transcript_with_name += "Each section below contains a title, a timestamp, a key image, and the EXACT reference text to summarize.\n\n"
        
        for seg in enhanced_segments:
            transcript_with_name += f"--- SECTION START ---\n"
            transcript_with_name += f"TIMESTAMP: [{seg['timestamp']}]\n"
            transcript_with_name += f"SECTION TITLE: {seg['title']}\n"
            if seg['image']:
                transcript_with_name += f"KEY IMAGE: <img src=\"{seg['image']}\" caption=\"{seg['title']}\">\n"
            transcript_with_name += f"REFERENCE TRANSCRIPT FOR THIS SECTION:\n{seg['content'] if seg['content'] else '(No transcript available for this segment)'}\n"
            transcript_with_name += f"--- SECTION END ---\n\n"

        # Also provide the raw transcript for context at the bottom
        transcript_with_name += "\n" + "="*50 + "\n"
        transcript_with_name += "FULL TRANSCRIPT (FOR OVERALL CONTEXT ONLY):\n"
        transcript_with_name += transcript_text
        
        # Save the new detailed structure to transcript.txt for inspection
        with open("transcript.txt", "w", encoding="utf-8") as f:
            f.write(transcript_with_name)

        # Cleanup local subtitle files (Keep the video AND SRT!)
        if video_path:
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
            "outline": outline,
            "enhanced_segments": enhanced_segments
        }
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return {"error": str(e)}
