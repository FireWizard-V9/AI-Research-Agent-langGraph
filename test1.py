import asyncpraw
import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Reddit API Credentials
REDDIT_CLIENT_ID = "uXfRYLHFXM7InBT-PEw8Sg"
REDDIT_CLIENT_SECRET = "bxYNfd6q4mHFXAQhWpFP_7pZTsJwjA"
REDDIT_USER_AGENT = "trendAgentAI"

async def fetch_subreddit_posts(subreddit_name):
    """Fetches the top posts from a subreddit."""
    
    # Move Reddit instance inside async function
    reddit = asyncpraw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    
    subreddit = await reddit.subreddit(subreddit_name)
    posts = []
    
    async for submission in subreddit.hot(limit=5):  # Fetch top 5 hot posts
        posts.append({
            "title": submission.title,
            "url": submission.url,
            "score": submission.score
        })
    
    await reddit.close()  # Ensure session is closed properly
    return posts

async def main():
    subreddit = "llms"  # Change to your desired subreddit
    posts = await fetch_subreddit_posts(subreddit)
    
    print("\n🔹 Top 5 Posts from r/{}:".format(subreddit))
    for post in posts:
        print(f"🔥 {post['title']} (🔗 {post['url']}) - 👍 {post['score']}")

if __name__ == "__main__":
    asyncio.run(main())
