from typing import List, Dict, Any
import logging
from pocketflow import Node, Flow
from utils.call_llm import call_llm
from utils.video_processor import get_video_info

# Removed redundant logging.basicConfig to allow main.py to handle it
logger = logging.getLogger(__name__)

class ProcessVideoNode(Node):
    """Download video and extract transcript + frames."""
    def prep(self, shared):
        return shared.get("url", "")
    
    def exec(self, url):
        logger.info(f"Downloading and processing: {url}")
        return get_video_info(url)
    
    def post(self, shared, prep_res, exec_res):
        if "error" in exec_res:
            raise ValueError(exec_res["error"])
        shared["video_info"] = exec_res
        return "default"

class GenerateBlogNode(Node):
    """Generate HTML blog using the complex prompt."""
    def prep(self, shared):
        return {
            "video_info": shared.get("video_info", {}),
            "blog_detail": shared.get("blog_detail", "Extremely detailed, covering every technical nuance and discussion point."),
        }
    
    def exec(self, data):
        video_info = data["video_info"]
        transcript_with_name = video_info.get("transcript_with_name", "")
        blog_detail = data["blog_detail"]
        
        # Style templates (can be customized)
        desired_writing_style = f"{blog_detail} Textbook quality, balanced narrative and technical code blocks. Focus on pedagogical clarity."
        desired_visual_style = "Modern web blog layout, responsive images with captions, clear hierarchy with H1/H2, 16px sans-serif body text."
        
        prompt = f"""
Human: Here is a transcript of a video, structured into GUIDED SEGMENTS with timestamps and images.

Your task is to transform this guided transcript into a high-quality, textbook-level HTML blog.

{transcript_with_name}

<desired_writing_style>
{desired_writing_style}
</desired_writing_style>

<desired_visual_style>
{desired_visual_style}
</desired_visual_style>

### CRITICAL GUIDELINES:

1. **STRICTLY FOLLOW THE GUIDED STRUCTURE**: 
   I have provided the video content in "--- SECTION START ---" blocks. You MUST create a separate H2 or H3 heading for EVERY single one of these sections. Do not skip any section, and do not merge multiple sections into one.

2. **COMPREHENSIVE CONTENT (NO CONDENSING)**:
   For each section, you are provided with a "REFERENCE TRANSCRIPT". You must use ALL the technical details and speaker viewpoints contained in that specific reference text. 
   - Since this is a "Textbook quality" blog, your goal is depth. 
   - If a section contains a panel discussion or multiple speakers, summarize each of their key contributions in detail. 
   - The final sections (e.g., Round Table/Panel Discussion) must be just as detailed as the beginning sections.

3. **SECTION LENGTH (MULTI-PARAGRAPH REQUIRED)**:
   Every section must contain multiple paragraphs (use <p> tags), not a single block.
   - Write at least 4 paragraphs per section.
   - Each paragraph should be 5–6 sentences.
   - Use plain paragraphs for expansion (avoid adding extra headings beyond the required H2/H3 for that section).

4. **MANDATORY IMAGE INCLUSION**:
   If a section contains a "KEY IMAGE" (e.g., <img src="frames/hh_mm_ss.jpg">), you MUST insert that image at the beginning of that section in the final HTML. Add a descriptive caption based on the section title.

5. **TIMESTAMPS**:
   Include the section timestamp at the beginning of the first paragraph of each section (e.g., **[01:30:50]**).

6. **LANGUAGE**:
   Use the SAME LANGUAGE as the provided reference transcript (if it's Chinese, write in Chinese).

7. **HTML OUTPUT**:
   Output valid HTML with clean CSS styling as described in <desired_visual_style>.

Assistant: <!DOCTYPE html>
"""
        logger.info("Calling LLM to generate blog...")
        response = call_llm(prompt)
        
        # Ensure it starts with the expected DOCTYPE if the model omitted it after the prefix
        if not response.strip().startswith("<html>") and not response.strip().startswith("<!DOCTYPE"):
            response = "<!DOCTYPE html>\n" + response
            
        return response
    
    def post(self, shared, prep_res, exec_res):
        shared["html_output"] = exec_res
        with open("output.html", "w", encoding="utf-8") as f:
            f.write(exec_res)
        logger.info("Blog generated successfully and saved to output.html")
        return "default"

def create_video_processor_flow():
    """Create the new blog generation flow."""
    process_video = ProcessVideoNode(max_retries=1)
    generate_blog = GenerateBlogNode(max_retries=2)
    
    process_video >> generate_blog
    
    return Flow(start=process_video)
