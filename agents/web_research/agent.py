"""
Web Research Agent — DealFlow AI
Framework: LangGraph
Model: GPT-4o via AI/ML API

Key design: do_complete_research() runs ALL searches, posts the SIGNAL to Band via
raw HTTP, and sends the @Synthesis/@Orchestrator handoff — all inside the tool.
The LLM's only job is to call this one tool. It cannot deviate on posting or handoff.
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

BAND_ROOM_ID = "8f4ebded-2988-4a75-915c-bcb80ad8a815"
BAND_API_BASE = "https://app.thenvoi.com/api/v1"
# Agent IDs for @mention routing (from runtime logs)
SYNTHESIS_AGENT_ID = "8c88e034-0f7e-43e2-b528-ab8b43b3e5bb"
ORCHESTRATOR_AGENT_ID = "1771a605-be42-431c-8003-dbddd3a25b35"

# -------------------------------------------------------------------
# Model — GPT-4o via AI/ML API
# -------------------------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o",
    base_url="https://api.aimlapi.com/v1",
    api_key=os.environ["AIML_API_KEY"],
    temperature=0.1,
)

# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------

# Phrases to match in Wikipedia-style company text (longer first for specificity).
_INDUSTRY_LEXICON = (
    "financial technology",
    "payment processing",
    "consumer electronics",
    "financial services",
    "telecommunications",
    "pharmaceutical",
    "biotechnology",
    "manufacturing",
    "e-commerce",
    "healthcare",
    "health care",
    "cryptocurrency",
    "semiconductor",
    "aerospace",
    "automotive",
    "insurance",
    "banking",
    "logistics",
    "transportation",
    "real estate",
    "agriculture",
    "fintech",
    "software",
    "internet",
    "retail",
    "energy",
    "media",
    "saas",
    "technology",
)


def _infer_industry_from_text(text: str) -> str | None:
    """Infer a short industry label from company overview text (e.g. Wikipedia extract)."""
    if not text:
        return None
    t = text.lower()
    for phrase in sorted(_INDUSTRY_LEXICON, key=len, reverse=True):
        if phrase in t:
            return phrase.replace("-", " ").title()
    return None


def _web_search(query: str) -> str:
    """Wikipedia REST API search with DuckDuckGo fallback. Returns text or empty string."""
    # --- Wikipedia ---
    try:
        search_resp = httpx.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": query,
                    "format": "json", "srlimit": 3},
            timeout=10.0,
        )
        if search_resp.status_code == 200:
            hits = search_resp.json().get("query", {}).get("search", [])
            if hits:
                title = hits[0]["title"].replace(" ", "_")
                summary_resp = httpx.get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
                    timeout=10.0,
                )
                if summary_resp.status_code == 200:
                    extract = summary_resp.json().get("extract", "")
                    if extract:
                        return extract[:500]
    except Exception:
        pass

    # --- DuckDuckGo fallback ---
    try:
        resp = httpx.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=10.0,
        )
        data = resp.json()
        results = []
        if data.get("AbstractText"):
            results.append(data["AbstractText"])
        for topic in data.get("RelatedTopics", [])[:3]:
            if "Text" in topic:
                results.append(topic["Text"])
        return " | ".join(results)[:400] if results else ""
    except Exception:
        return ""


def _post_to_band(content: str, api_key: str, mention_ids: list[str]) -> int:
    """Post a message to the Band room. Returns HTTP status code."""
    try:
        url = f"{BAND_API_BASE}/agent/chats/{BAND_ROOM_ID}/messages"
        headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json",
        }
        body = {
            "message": {
                "content": content,
                "mentions": [{"id": mid} for mid in mention_ids],
            }
        }
        resp = httpx.post(url, json=body, headers=headers, timeout=10.0)
        if resp.status_code != 201:
            logger.error(f"Band API error {resp.status_code}: {resp.text[:500]}")
        return resp.status_code
    except Exception as e:
        logger.error(f"Failed to post to Band: {e}")
        return 0


# -------------------------------------------------------------------
# Agent setup — tool created inside factory so it captures api_key
# -------------------------------------------------------------------

def _get_participant_ids(api_key: str) -> list[dict]:
    """Fetch all room participants and return their full objects."""
    try:
        url = f"{BAND_API_BASE}/agent/chats/{BAND_ROOM_ID}/participants"
        resp = httpx.get(url, headers={"X-API-Key": api_key}, timeout=10.0)
        if resp.status_code == 200:
            data = resp.json()
            participants = data.get("data", data) if isinstance(data, dict) else data
            # Log the FULL first participant so we can see all available fields
            if participants:
                logger.info(f"Participant object keys: {list(participants[0].keys())}")
                logger.info(f"First participant full object: {participants[0]}")
            return participants
        else:
            logger.error(f"Failed to fetch participants: {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        logger.error(f"Error fetching participants: {e}")
    return []


def create_web_research() -> Agent:
    agent_id, api_key = load_agent_config("web_research")

    # Fetch participants to discover correct mention ID field
    participants = _get_participant_ids(api_key)
    # Pick the Orchestrator participant (it IS in the room) to use as mention target
    mention_id = None
    for p in participants:
        pname = (p.get("name") or p.get("handle") or "").lower()
        if "orchestrator" in pname:
            # Try participant_id first, then id
            mention_id = p.get("participant_id") or p.get("id")
            logger.info(f"Orchestrator participant object: {p}")
            break
    if not mention_id:
        logger.error("Could not find Orchestrator participant — cannot post mentions")


    @tool
    def do_complete_research(company_name: str) -> str:
        """
        THE ONLY TOOL YOU NEED. Call this ONCE when asked to research a company.
        Searches the web, posts the SIGNAL:market_research to Band, and sends the
        handoff to @Synthesis and @Orchestrator — all automatically.
        Industry is inferred from the company article text plus a direct web search
        for sector/market context (no industry argument).
        Returns a status string when complete. Do NOT call thenvoi_send_message.
        """
        # 1. Fetch company Wikipedia page directly (single call, most reliable)
        def _wiki_summary(title: str) -> str:
            try:
                r = httpx.get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
                    timeout=10.0,
                )
                if r.status_code == 200:
                    return r.json().get("extract", "")
            except Exception:
                pass
            return ""

        # Normalize company name to likely Wikipedia title (e.g. "Stripe" → "Stripe,_Inc.")
        wiki_title = company_name.replace(" ", "_")
        company_extract = (
            _wiki_summary(f"{wiki_title},_Inc.")
            or _wiki_summary(wiki_title)
            or _web_search(f"{company_name} company")
        )
        industry_extract = _web_search(f"{company_name} industry sector market")
        inferred_industry = _infer_industry_from_text(company_extract) or "technology"

        # Extract specific fields from the company Wikipedia article
        def _extract_sentence(text: str, keywords: list[str]) -> str:
            """Return first sentence containing any keyword."""
            for sent in text.split(". "):
                if any(kw.lower() in sent.lower() for kw in keywords):
                    return sent.strip()[:200]
            return text[:150] if text else "Not found"

        founders_data = _extract_sentence(company_extract, ["founded", "founder", "brothers", "CEO", "Patrick", "John"])
        funding_data = _extract_sentence(company_extract, ["valued", "valuation", "billion", "funding", "raised", "investors"])
        tam_raw = industry_extract or _web_search(f"{inferred_industry} market size") or ""
        tam_data = tam_raw[:250] if tam_raw else "Unknown"
        _competitor_kw = ["competitor", "compete", "rival", "alternative", "versus"]
        competitors_data = (
            _web_search(f"{company_name} main competitors")
            or _extract_sentence(company_extract, _competitor_kw)
            or "Unknown"
        )
        news_data = _extract_sentence(company_extract, ["2023", "2024", "recent", "launched", "announced", "expanded"]) or "No recent news retrieved"

        research = {
            "company_name": company_name,
            "industry": inferred_industry,
            "company_overview": company_extract[:200] if company_extract else "Unknown",
            "tam_estimate": tam_data,
            "top_competitors": competitors_data,
            "news_sentiment": news_data,
            "founders": founders_data,
            "funding_info": funding_data,
            "data_source": "Wikipedia REST API",
        }

        payload = json.dumps(research, separators=(',', ':'))
        signal = f"SIGNAL:market_research\n{payload}"
        if len(signal) > 1900:
            signal = signal[:1890] + "...}"

        # 2. Post SIGNAL — mention Orchestrator (confirmed in room) so mention validates
        mid = mention_id or ORCHESTRATOR_AGENT_ID
        status1 = _post_to_band(signal, api_key, [mid])
        logger.info(f"Posted SIGNAL:market_research → HTTP {status1}")

        # 3. Post handoff
        handoff = "@Synthesis Market research complete. Please synthesize into investment memo. @Orchestrator WebResearch done."
        status2 = _post_to_band(handoff, api_key, [mid])
        logger.info(f"Posted handoff → HTTP {status2}")

        return f"Research complete. SIGNAL posted (HTTP {status1}). Handoff sent to @Synthesis (HTTP {status2})."

    adapter = LangGraphAdapter(
        llm=llm.bind(system=WEB_RESEARCH_PROMPT),
        checkpointer=InMemorySaver(),
        additional_tools=[do_complete_research],
    )

    return Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
    )


async def run_web_research():
    logger.info("Web Research Agent starting (GPT-4o via AI/ML API)...")
    agent = create_web_research()
    await agent.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_web_research())
