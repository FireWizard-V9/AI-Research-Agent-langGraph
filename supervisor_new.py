import asyncio
import os
import nest_asyncio
from dotenv import load_dotenv
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# ✅ Import all tool functions
from tools import (
    wikipedia_tool,
    hackernews_tool,
    newsapi_tool,
    arxiv_tool
)
from reddit import fetch_reddit_posts
from tavily import search_tavily  # ⬅️ This is a sync function, don't use `await`
from youtube import youtube_search_tool

# ✅ Load environment variables
load_dotenv()

# ✅ Apply nest_asyncio to avoid event loop issues in Jupyter/VS Code
nest_asyncio.apply()

# ✅ Ensure API Keys are properly loaded
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

if not NEWS_API_KEY:
    print("❌ Warning: NewsAPI key is missing. Ensure it's set in the environment.")
if not YOUTUBE_API_KEY:
    print("❌ Warning: YouTube API key is missing. Ensure it's set in the environment.")

# ✅ Initialize OpenAI model for structuring responses
model = ChatOpenAI(
    temperature=0.8,
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini"
)

# ✅ Create LangGraph Agents for Each Tool
reddit_agent = create_react_agent(
    model=model,
    tools=[fetch_reddit_posts],
    name="Reddit Agent",
    prompt="You are an expert Reddit researcher. Fetch trending posts related to {input}. Return the post title, URL, and upvotes."
)

tavily_agent = create_react_agent(
    model=model,
    tools=[search_tavily],  # No need to await, it's a normal function
    name="Web Search Agent",
    prompt="You are an advanced web researcher. Use Tavily to fetch the latest articles and insights for {input}. Return article titles and URLs."
)

youtube_agent = create_react_agent(
    model=model,
    tools=[youtube_search_tool],
    name="YouTube Agent",
    prompt="You are an expert YouTube researcher. Fetch trending videos related to {input}. Return the video title, URL, and views."
)

wikipedia_agent = create_react_agent(
    model=model,
    tools=[wikipedia_tool],
    name="Wikipedia Agent",
    prompt="You are a Wikipedia researcher. Fetch structured summaries for {input}. Provide the summary and the source link."
)

hackernews_agent = create_react_agent(
    model=model,
    tools=[hackernews_tool],
    name="Hacker News Agent",
    prompt="You are a Hacker News researcher. Find trending discussions on {input}. Provide the discussion title, URL, and popularity."
)

newsapi_agent = create_react_agent(
    model=model,
    tools=[newsapi_tool],
    name="News Agent",
    prompt="You are a news analyst. Fetch the latest AI news related to {input}. Provide article titles and URLs."
)

arxiv_agent = create_react_agent(
    model=model,
    tools=[arxiv_tool],
    name="Arxiv Research Agent",
    prompt="You are an AI researcher. Fetch the latest AI research papers on {input}. Provide paper titles and URLs."
)

# ✅ Supervisor AI (Combines all agent responses)
workflow = create_supervisor(
    [
        reddit_agent, 
        tavily_agent, 
        youtube_agent, 
        wikipedia_agent, 
        hackernews_agent, 
        newsapi_agent, 
        arxiv_agent
    ],
    model=model,
    prompt="You are a research supervisor. Combine responses from different agents and structure them for the user."
)

# ✅ Compile Workflow
app = workflow.compile()

# ✅ Main Execution
async def main():
    query = input("🔍 Enter a topic to search: ").strip()

    print(f"\n🔍 Searching for '{query}'...\n")

    # ✅ Ensure async tools are properly awaited
    reddit_results = await fetch_reddit_posts(query)  # Directly await
    tavily_results = search_tavily(query)  # ⬅️ FIXED: Removed `await`, since it's a normal function

    # ✅ Run other tools in threads (for synchronous LangChain tools)
    youtube_results = await asyncio.to_thread(youtube_search_tool.invoke, query)
    wikipedia_results = await asyncio.to_thread(wikipedia_tool.invoke, query)
    hackernews_results = await asyncio.to_thread(hackernews_tool.invoke, query)
    newsapi_results = await asyncio.to_thread(newsapi_tool.invoke, query)
    arxiv_results = await asyncio.to_thread(arxiv_tool.invoke, query)

    # Print Raw Results
    print("\n🟥 Raw Reddit Results:\n", reddit_results)
    print("\n🌍 Raw Tavily Results:\n", tavily_results)
    print("\n📺 Raw YouTube Results:\n", youtube_results)
    print("\n📖 Raw Wikipedia Results:\n", wikipedia_results)
    print("\n📰 Raw Hacker News Results:\n", hackernews_results)
    print("\n🗞️ Raw NewsAPI Results:\n", newsapi_results)
    print("\n📄 Raw Arxiv Results:\n", arxiv_results)

    print("\n🔄 Passing results to Supervisor AI...\n")

    # Invoke Supervisor AI
    result = app.invoke({
        "messages": [
            {"role": "user", "content": query},
            {"role": "assistant", "content": f"Reddit Results: {reddit_results}"},
            {"role": "assistant", "content": f"Tavily Results: {tavily_results}"},
            {"role": "assistant", "content": f"YouTube Results: {youtube_results}"},
            {"role": "assistant", "content": f"Wikipedia Results: {wikipedia_results}"},
            {"role": "assistant", "content": f"Hacker News Results: {hackernews_results}"},
            {"role": "assistant", "content": f"NewsAPI Results: {newsapi_results}"},
            {"role": "assistant", "content": f"Arxiv Results: {arxiv_results}"}
        ]
    })

    print("\n🔹 Final Structured Response:\n", result)

if __name__ == "__main__":
    asyncio.run(main())
