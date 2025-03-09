import asyncio
import os
from dotenv import load_dotenv
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# ✅ Import tools from tools.py
from tools import (
    wikipedia_tool,
    hackernews_tool,
    newsapi_tool,
    arxiv_tool
)
from reddit import fetch_reddit_posts
from tavily import search_tavily
from youtube import youtube_search_tool

# ✅ Load environment variables
load_dotenv()

# ✅ Initialize OpenAI model for structuring responses
model = ChatOpenAI(
    temperature=0.8,
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini"
)

# ✅ Async function for Reddit (since it's already async)
async def reddit_search_tool(query: str):
    """Fetch trending Reddit posts asynchronously."""
    return await fetch_reddit_posts(query)

# ✅ Tavily (already synchronous)
def tavily_search_tool(query: str):
    """Fetch web search results using Tavily."""
    return search_tavily(query)

# ✅ Handle synchronous tools correctly
async def run_tool(tool, query):
    """Runs LangChain Tools & normal functions correctly."""
    if hasattr(tool, "invoke"):  # If it's a LangChain Tool
        return await asyncio.to_thread(tool.invoke, query)
    else:  # If it's a normal function
        return await asyncio.to_thread(tool, query)

# ✅ Create LangGraph Agents for Each Tool
reddit_agent = create_react_agent(
    model=model,
    tools=[reddit_search_tool],
    name="Reddit Agent",
    prompt="You are an expert Reddit researcher. Fetch trending posts related to {input}. Return the post title, URL, and upvotes."
)

tavily_agent = create_react_agent(
    model=model,
    tools=[tavily_search_tool],
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

    # Run all agents concurrently
    tasks = [
        asyncio.create_task(reddit_search_tool(query)),
        asyncio.create_task(run_tool(tavily_search_tool, query)),
        asyncio.create_task(run_tool(youtube_search_tool, query)),
        asyncio.create_task(run_tool(wikipedia_tool, query)),
        asyncio.create_task(run_tool(hackernews_tool, query)),
        asyncio.create_task(run_tool(newsapi_tool, query)),
        asyncio.create_task(run_tool(arxiv_tool, query))
    ]

    # Collect results
    results = await asyncio.gather(*tasks)
    (
        reddit_results, tavily_results, youtube_results, 
        wikipedia_results, hackernews_results, 
        newsapi_results, arxiv_results
    ) = results

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
