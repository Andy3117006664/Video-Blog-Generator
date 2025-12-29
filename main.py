import argparse
import logging
import sys
import os
from flow import create_video_processor_flow

# Set up logging
# Use force=True to ensure this config is applied even if logging was already initialized
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("video_blog_generator.log", mode='w', encoding='utf-8')
    ],
    force=True
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the Video Blog Generator."""
    
    parser = argparse.ArgumentParser(
        description="Transform a video (YouTube/Bilibili) into a structured HTML blog with screenshots."
    )
    parser.add_argument(
        "--url", 
        type=str, 
        help="Video URL to process",
        required=False
    )
    args = parser.parse_args()
    
    url = args.url
    if not url:
        url = input("Enter Video URL (YouTube/Bilibili): ")
    
    # Create output directory for frames if it doesn't exist
    if not os.path.exists("frames"):
        os.makedirs("frames")

    # Create flow
    flow = create_video_processor_flow()
    
    # Configure blog detail and image count here if needed
    shared = {
        "url": url,
        "blog_detail": "Extremely detailed and comprehensive, capturing all technical discussions, examples, and conclusions."
    }
    
    try:
        flow.run(shared)
        print("\n" + "=" * 50)
        print("Success! The blog has been generated.")
        print(f"File: {os.path.abspath('output.html')}")
        print("Screenshots: frames/ folder")
        print("=" * 50 + "\n")
    except Exception as e:
        logger.error(f"Flow failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
