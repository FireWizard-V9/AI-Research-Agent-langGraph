import asyncio
import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

# ✅ Import tools
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

# ✅ Initialize OpenAI model
model = ChatOpenAI(
    temperature=0.8,
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini"
)

# ✅ Async function for Reddit
async def reddit_search_tool(query: str):
    """Fetch trending Reddit posts asynchronously."""
    return await fetch_reddit_posts(query)

# ✅ Tavily Search (Synchronous)
def tavily_search_tool(query: str):
    """Fetch web search results using Tavily."""
    return search_tavily(query)

# ✅ Async tool runner
async def run_tool(tool, query):
    """Runs LangChain Tools & normal functions correctly."""
    return await asyncio.to_thread(tool.invoke, query) if hasattr(tool, "invoke") else await asyncio.to_thread(tool, query)

# ✅ Create LangGraph Agents
agents = {
    "Reddit": create_react_agent(model, [reddit_search_tool], name="Reddit Agent", prompt="Fetch trending Reddit posts."),
    "Tavily": create_react_agent(model, [tavily_search_tool], name="Web Search Agent", prompt="Fetch articles using Tavily."),
    "YouTube": create_react_agent(model, [youtube_search_tool], name="YouTube Agent", prompt="Fetch trending YouTube videos."),
    "Wikipedia": create_react_agent(model, [wikipedia_tool], name="Wikipedia Agent", prompt="Fetch Wikipedia summaries."),
    "Hacker News": create_react_agent(model, [hackernews_tool], name="Hacker News Agent", prompt="Find trending Hacker News."),
    "NewsAPI": create_react_agent(model, [newsapi_tool], name="News Agent", prompt="Fetch the latest news articles."),
    "Arxiv": create_react_agent(model, [arxiv_tool], name="Arxiv Research Agent", prompt="Fetch latest AI research papers."),
}

def combine_results(state):
    """Combine results into structured response."""
    query = state["messages"][0].content
    source_results = {msg.content.split(":")[0]: msg.content.split(":", 1)[1].strip() for msg in state["messages"][1:] if isinstance(msg, dict)}
    
    prompt = f"""
    Organize the research findings for: "{query}".
    Summarize key points and format as a markdown report:
    - A brief summary
    - Findings categorized by source
    - Markdown formatting for readability
    """
    structured_response = model.invoke(prompt).content
    state["messages"].append({"role": "assistant", "content": structured_response})
    return state

def create_custom_supervisor():
    """Creates LangGraph-based supervisor workflow."""
    workflow = StateGraph()  # ✅ FIXED: Correct initialization

    for agent_name, agent in agents.items():
        workflow.add_node(agent_name, agent)
    workflow.add_node("combiner", combine_results)

    for agent_name in agents.keys():
        workflow.add_edge("__start__", agent_name)
        workflow.add_edge(agent_name, "combiner")

    workflow.add_edge("combiner", END)
    return workflow

# ✅ Compile Workflow
workflow = create_custom_supervisor().compile()

async def get_agent_results(query):
    """Run all agents asynchronously."""
    tasks = [
        asyncio.create_task(reddit_search_tool(query)),
        asyncio.create_task(run_tool(tavily_search_tool, query)),
        asyncio.create_task(run_tool(youtube_search_tool, query)),
        asyncio.create_task(run_tool(wikipedia_tool, query)),
        asyncio.create_task(run_tool(hackernews_tool, query)),
        asyncio.create_task(run_tool(newsapi_tool, query)),
        asyncio.create_task(run_tool(arxiv_tool, query))
    ]
    return await asyncio.gather(*tasks)

async def run_supervisor_flow(query):
    """Runs full research process asynchronously."""
    results = await get_agent_results(query)

    state = {
        "messages": [
            {"role": "user", "content": query},
            {"role": "assistant", "content": f"Reddit Results: {results[0]}"},
            {"role": "assistant", "content": f"Tavily Results: {results[1]}"},
            {"role": "assistant", "content": f"YouTube Results: {results[2]}"},
            {"role": "assistant", "content": f"Wikipedia Results: {results[3]}"},
            {"role": "assistant", "content": f"Hacker News Results: {results[4]}"},
            {"role": "assistant", "content": f"NewsAPI Results: {results[5]}"},
            {"role": "assistant", "content": f"Arxiv Results: {results[6]}"}
        ]
    }

    final_state = combine_results(state)
    return {"final_response": final_state["messages"][-1]["content"], "raw_results": dict(zip(agents.keys(), results))}
