import os
from googleapiclient.discovery import build
from dotenv import load_dotenv
from langchain_core.tools import Tool

# Load environment variables
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

def search_youtube_videos(query: str, max_results=5):
    """Search YouTube for videos related to the query and return structured data."""
    if not YOUTUBE_API_KEY:
        print("❌ Error: YouTube API Key not found. Set it in your .env file.")
        return []

    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

    try:
        search_response = youtube.search().list(
            q=query,
            part="snippet",
            type="video",
            maxResults=max_results
        ).execute()

        videos = []
        for item in search_response.get("items", []):
            video_title = item["snippet"]["title"]
            video_id = item["id"]["videoId"]
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            channel_name = item["snippet"]["channelTitle"]
            publish_date = item["snippet"]["publishedAt"]
            thumbnail_url = item["snippet"]["thumbnails"]["high"]["url"]

            videos.append({
                "title": video_title,
                "url": video_url,
                "channel": channel_name,
                "published_at": publish_date,
                "thumbnail": thumbnail_url
            })

        return videos

    except Exception as e:
        print(f"❌ YouTube API Error: {e}")
        return []

youtube_search_tool = Tool(
    name="YouTube Video Search",
    description="Searches for top YouTube videos related to a given topic.",
    func=search_youtube_videos
)
