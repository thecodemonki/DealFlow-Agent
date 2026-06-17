"""
Synthesis Agent — DealFlow AI
Framework: LangGraph
Model: GPT-4o via AI/ML API

The final agent. Waits for all specialist signals, synthesizes findings,
writes the investment memo, and generates a PDF report.
"""

import asyncio
import json
import logging
import os

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from thenvoi import Agent
from thenvoi.adapters import LangGraphAdapter
from thenvoi.config import load_agent_config

from shared.pdf_memo import build_memo_pdf_bytes, normalize_deal_score_fields
from shared.prompts import SYNTHESIS_PROMPT

load_dotenv()
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Model — GPT-4o via AI/ML API (best reasoning for synthesis)
# Claude models have a 200-char tool name limit incompatible with Band SDK
# -------------------------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o",
    base_url="https://api.aimlapi.com/v1",
    api_key=os.environ["AIML_API_KEY"],
    temperature=0.3,
)

# -------------------------------------------------------------------
# Synthesis and PDF generation tools
# -------------------------------------------------------------------


@tool
def parse_signal_payload(message: str) -> dict:
    """
    Extract the JSON payload from a SIGNAL: message posted by a specialist agent.
    Returns {signal_type, payload} or {error}.
    """
    try:
        lines = message.strip().split("\n", 1)
        if lines[0].startswith("SIGNAL:"):
            signal_type = lines[0].replace("SIGNAL:", "").strip()
            payload = json.loads(lines[1]) if len(lines) > 1 else {}
            return {"signal_type": signal_type, "payload": payload}
    except Exception as e:
        return {"error": str(e)}
    return {"error": "Not a valid SIGNAL message"}


@tool
def generate_pdf_memo(memo_data: dict) -> str:
    """
    Validate and render the investment memo as a PDF in memory.
    Returns a confirmation string — downloads are served by the API from memo_summary.
    """
    try:
        pdf_bytes = build_memo_pdf_bytes(memo_data)
        logger.info("PDF memo generated in memory (%d bytes)", len(pdf_bytes))
        return (
            f"PDF ready ({len(pdf_bytes)} bytes). "
            "Download is served by the API from stored memo_summary."
        )
    except Exception as e:
        logger.error("PDF generation failed: %s", e)
        return f"ERROR: {e}"


@tool
def format_final_signal(memo: dict) -> str:
    """Format InvestmentMemo as a SIGNAL message for the Band room.
    Posts a concise summary — full memo PDF is generated on download by the API.
    Keeps content under 2000 chars to comply with Band API limits."""
    memo = normalize_deal_score_fields(dict(memo))
    summary = {
        "company_name": memo.get("company_name", ""),
        "recommendation": memo.get("recommendation", ""),
        "confidence": memo.get("confidence", ""),
        "deal_score": memo.get("deal_score", 50),
        "deal_verdict": memo.get("deal_verdict", "CONDITIONAL"),
        "risks_flagged_count": memo.get("risks_flagged_count", 0),
        "executive_summary": (memo.get("executive_summary", "") or "")[:400],
        "red_flags": (memo.get("red_flags", []) or [])[:3],
    }
    signal = f"SIGNAL:investment_memo\n{json.dumps(summary, indent=2)}"
    if len(signal) > 2000:
        signal = signal[:1990] + "\n...}"
    return signal


# -------------------------------------------------------------------
# Agent setup
# -------------------------------------------------------------------

def create_synthesis() -> Agent:
    agent_id, api_key = load_agent_config("synthesis")

    adapter = LangGraphAdapter(
        llm=llm.bind(system=SYNTHESIS_PROMPT),
        checkpointer=InMemorySaver(),
        additional_tools=[
            parse_signal_payload,
            generate_pdf_memo,
            format_final_signal,
        ],
    )

    return Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
    )


async def run_synthesis():
    logger.info("Synthesis Agent starting (GPT-4o via AI/ML API)...")
    agent = create_synthesis()
    await agent.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_synthesis())
