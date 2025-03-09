import asyncio
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

# ✅ Import tools
from tools import wikipedia_tool, hackernews_tool, newsapi_tool, arxiv_tool
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

# ✅ Rich Console for Colorful Output
console = Console()

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

# ✅ Create Supervisor Workflow
workflow = create_supervisor(
    list(agents.values()),
    model=model,
    prompt="You are a research supervisor. Combine responses from different agents and structure them for the user."
)

# ✅ Compile Workflow
app = workflow.compile()

# ✅ Main Execution with Rich Formatting
async def main():
    query = console.input("[bold cyan]🔍 Enter a topic to search: [/bold cyan]").strip()
    console.print(Panel(f"[bold cyan]Searching for: {query}[/bold cyan]", expand=False))

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

    # ✅ Display Results in a Table
    table = Table(title=f"[bold cyan]🔍 Research Results for '{query}'[/bold cyan]")
    table.add_column("Source", style="bold magenta", justify="left")
    table.add_column("Top Results", style="bold yellow", justify="left")

    def extract_titles(data):
        """Extracts top 3 titles from results"""
        return "\n".join([f"- {item['title']}" for item in data[:3]]) if isinstance(data, list) else str(data)

    table.add_row("📢 Reddit", extract_titles(reddit_results))
    table.add_row("🌍 Tavily", extract_titles(tavily_results))
    table.add_row("📺 YouTube", extract_titles(youtube_results))
    table.add_row("📖 Wikipedia", extract_titles(wikipedia_results))
    table.add_row("📰 Hacker News", extract_titles(hackernews_results))
    table.add_row("🗞️ NewsAPI", extract_titles(newsapi_results))
    table.add_row("📄 Arxiv", extract_titles(arxiv_results))

    console.print(table)

    console.print("\n[bold green]🔄 Passing results to Supervisor AI...[/bold green]\n")

    # ✅ Invoke Supervisor AI
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

    console.print(Panel("[bold green]🔹 Final Structured Response:[/bold green]", expand=False))

    # ✅ Fix: Handle Missing 'final_response'
    structured_response = result.get("final_response", "[bold red]⚠ No structured response generated![/bold red]")
    console.print(structured_response, style="bold white")

if __name__ == "__main__":
    asyncio.run(main())
