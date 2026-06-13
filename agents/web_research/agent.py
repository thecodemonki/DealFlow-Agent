"""
Web Research Agent — DealFlow AI
Framework: LangGraph
Model: Claude Haiku via AI/ML API

Fast and cost-effective for high-volume search + summarization.
Handles market sizing, competitive landscape, news, and founder research.
"""

import asyncio
import json
import logging
import os

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from thenvoi import Agent
from thenvoi.adapters import LangGraphAdapter
from thenvoi.config import load_agent_config

from shared.prompts import WEB_RESEARCH_PROMPT

load_dotenv()
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Model — Claude Haiku via AI/ML API (fast + cheap for search tasks)
# -------------------------------------------------------------------
llm = ChatOpenAI(
    model="claude-haiku-4-5-20251001",
    base_url="https://api.aimlapi.com/v1",
    api_key=os.environ["AIML_API_KEY"],
    temperature=0.2,
)

# -------------------------------------------------------------------
# Web research tools
# -------------------------------------------------------------------

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo's instant answer API.
    Returns a summary of top results for a given query.
    """
    try:
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
        resp = httpx.get(url, params=params, timeout=10.0)
        data = resp.json()

        results = []
        if data.get("AbstractText"):
            results.append(f"Summary: {data['AbstractText']}")
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if "Text" in topic:
                results.append(topic["Text"])

        return "\n\n".join(results) if results else f"No results found for: {query}"
    except Exception as e:
        return f"Search error: {e}"


@tool
def search_crunchbase_style(company_name: str) -> str:
    """
    Search for funding, investors, and company info (uses public web search as proxy).
    In production, replace with actual Crunchbase or PitchBook API.
    """
    return web_search(f"{company_name} startup funding rounds investors total raised site:crunchbase.com OR site:techcrunch.com")


@tool
def search_market_size(industry: str) -> str:
    """Search for market size and growth rate data for a given industry/sector."""
    return web_search(f"{industry} total addressable market size 2024 2025 growth rate billion CAGR")


@tool
def search_competitors(company_name: str, industry: str) -> str:
    """Find top competitors for a company in its market."""
    return web_search(f"{company_name} competitors alternatives {industry} comparison")


@tool
def search_news_sentiment(company_name: str) -> str:
    """Search for recent news about a company to assess sentiment."""
    return web_search(f'"{company_name}" news 2024 2025 site:techcrunch.com OR site:reuters.com OR site:bloomberg.com')


@tool
def search_founders(company_name: str) -> str:
    """Research the founding team background and track record."""
    return web_search(f"{company_name} founders CEO CTO background experience previous companies LinkedIn")


@tool
def format_market_signal(research: dict) -> str:
    """Format MarketResearch dict as a SIGNAL message for the Band room."""
    return f"SIGNAL:market_research\n{json.dumps(research, indent=2)}"


# -------------------------------------------------------------------
# Agent setup
# -------------------------------------------------------------------

def create_web_research() -> Agent:
    agent_id, api_key = load_agent_config("web_research")

    adapter = LangGraphAdapter(
        llm=llm,
        checkpointer=InMemorySaver(),
        additional_tools=[
            web_search,
            search_crunchbase_style,
            search_market_size,
            search_competitors,
            search_news_sentiment,
            search_founders,
            format_market_signal,
        ],
        system_prompt=WEB_RESEARCH_PROMPT,
    )

    return Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
    )


async def run_web_research():
    logger.info("Web Research Agent starting (Claude Haiku via AI/ML API)...")
    agent = create_web_research()
    await agent.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_web_research())
