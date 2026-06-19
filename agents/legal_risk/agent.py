"""
Legal Risk Agent — DealFlow AI
Framework: LangGraph
Model: GPT-4o-mini via AI/ML API

Reliable tool calling + strong instruction-following for legal risk analysis.
Switched from Featherless Llama-3.1-70B to eliminate 422 tool format errors.
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
from shared.agent_config import load_agent_config

from shared.prompts import LEGAL_RISK_PROMPT

load_dotenv()
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Model — GPT-4o-mini via AI/ML API (reliable tool calling, no Featherless 422s)
# -------------------------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    base_url="https://api.aimlapi.com/v1",
    api_key=os.environ["AIML_API_KEY"],
    temperature=0.1,
)

# -------------------------------------------------------------------
# Legal analysis tools
# -------------------------------------------------------------------

@tool
def parse_documents_signal(signal_json: str) -> dict:
    """Parse the ParsedDocuments JSON from a Band message signal."""
    try:
        return json.loads(signal_json)
    except Exception as e:
        return {"error": str(e)}


@tool
def identify_change_of_control_risk(contract_text: str) -> list:
    """
    Scan contract text for change-of-control clauses.
    Returns a list of flagged clause descriptions.
    """
    red_flag_phrases = [
        "change of control", "change-of-control", "acquisition", "merger",
        "assignment", "consent required", "termination upon", "accelerate",
        "anti-assignment", "successor"
    ]
    found = []
    lower = contract_text.lower()
    for phrase in red_flag_phrases:
        if phrase in lower:
            found.append(f"Clause contains '{phrase}' — review for CoC trigger")
    return found


@tool
def format_legal_signal(analysis: dict) -> str:
    """Format LegalRiskAnalysis as a SIGNAL message for the Band room.
    Keeps content under 2000 chars to comply with Band API limits."""
    payload = json.dumps(analysis, indent=2)
    signal = f"SIGNAL:legal_risk\n{payload}"
    if len(signal) > 2000:
        # Truncate to a compact version
        compact = json.dumps(analysis, separators=(',', ':'))
        signal = f"SIGNAL:legal_risk\n{compact}"
    if len(signal) > 2000:
        signal = signal[:1990] + "\n...}"
    return signal


# -------------------------------------------------------------------
# Agent setup
# -------------------------------------------------------------------

def create_legal_risk() -> Agent:
    agent_id, api_key = load_agent_config("legal_risk")

    adapter = LangGraphAdapter(
        llm=llm.bind(system=LEGAL_RISK_PROMPT),
        checkpointer=InMemorySaver(),
        additional_tools=[
            parse_documents_signal,
            identify_change_of_control_risk,
            format_legal_signal,
        ],
    )

    return Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
    )


async def run_legal_risk():
    logger.info("Legal Risk Agent starting (GPT-4o-mini via AI/ML API)...")
    agent = create_legal_risk()
    await agent.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_legal_risk())
