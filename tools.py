import os
import requests
from dotenv import load_dotenv
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_core.tools import Tool

# ✅ Load environment variables
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# ✅ Wikipedia Search Tool
def search_wikipedia(query: str):
    wiki = WikipediaAPIWrapper()
    return wiki.run(query)

wikipedia_tool = Tool(
    name="Wikipedia Search",
    description="Fetch structured summaries from Wikipedia for a given topic.",
    func=search_wikipedia
)

# ✅ Hacker News Search Tool
def search_hackernews(query: str, num_results=5):
    url = f"https://hn.algolia.com/api/v1/search?query={query}&hitsPerPage={num_results}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        return [
            {
                "title": item.get("title", "No Title Available"),
                "url": item.get("url", "#")  # If URL is missing, return "#"
            }
            for item in data.get("hits", [])  # Ensure "hits" exist before iterating
        ]

    except requests.RequestException as e:
        print(f"❌ Error fetching Hacker News: {e}")
        return []

hackernews_tool = Tool(
    name="Hacker News Search",
    description="Find trending discussions from Hacker News related to a topic.",
    func=search_hackernews
)

# ✅ NewsAPI Search Tool
def search_newsapi(query: str, num_results=5):
    if not NEWS_API_KEY:
        print("❌ Error: NewsAPI key is missing.")
        return []

    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&pageSize={num_results}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        return [
            {"title": article["title"], "url": article["url"]}
            for article in data.get("articles", [])  # Ensure "articles" exist
        ]

    except requests.RequestException as e:
        print(f"❌ Error fetching NewsAPI: {e}")
        return []

newsapi_tool = Tool(
    name="NewsAPI Search",
    description="Fetch the latest news articles related to a topic.",
    func=search_newsapi
)

# ✅ Arxiv Research Paper Search Tool
def search_arxiv(query: str, num_results=5):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={num_results}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.text.split("<entry>")
        papers = []

        for entry in data[1:num_results+1]:  
            title = entry.split("<title>")[1].split("</title>")[0].strip()
            link = entry.split("<id>")[1].split("</id>")[0].strip()
            papers.append({"title": title, "url": link})

        return papers

    except requests.RequestException as e:
        print(f"❌ Error fetching Arxiv: {e}")
        return []

arxiv_tool = Tool(
    name="Arxiv Paper Search",
    description="Fetch the latest AI research papers from Arxiv.",
    func=search_arxiv
)
