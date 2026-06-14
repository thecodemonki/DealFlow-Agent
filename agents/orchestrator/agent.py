"""
Orchestrator Agent — DealFlow AI
Framework: LangGraph
Model: GPT-4o via AI/ML API

The brain of the system. Creates the Band deal room, recruits specialist agents,
coordinates handoffs, escalates to humans when needed, and confirms completion.
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

from shared.prompts import ORCHESTRATOR_PROMPT

load_dotenv()
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# LLM — GPT-4o via AI/ML API (OpenAI-compatible endpoint)
# Claude models have a 200-char tool name limit incompatible with Band SDK
# -------------------------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o",
    base_url="https://api.aimlapi.com/v1",
    api_key=os.environ["AIML_API_KEY"],
    temperature=0.1,
)

# -------------------------------------------------------------------
# Tools — all Band platform tools are injected automatically by the SDK.
# We add a custom tool for the Orchestrator to parse incoming deal signals.
# -------------------------------------------------------------------

@tool
def parse_agent_signal(message_content: str) -> dict:
    """
    Parse a structured SIGNAL message posted by a specialist agent.
    Returns the signal type and payload dict.
    """
    try:
        lines = message_content.strip().split("\n", 1)
        if lines[0].startswith("SIGNAL:"):
            signal_type = lines[0].replace("SIGNAL:", "").strip()
            payload = json.loads(lines[1]) if len(lines) > 1 else {}
            return {"signal_type": signal_type, "payload": payload, "valid": True}
    except Exception as e:
        logger.warning(f"Failed to parse signal: {e}")
    return {"signal_type": "unknown", "payload": {}, "valid": False}


@tool
def build_deal_kickoff_message(company_name: str, file_paths: list, notes: str = "") -> str:
    """
    Build the formatted message to send to @DocumentParser to kick off analysis.
    """
    paths_str = "\n".join(f"  - {p}" for p in file_paths) if file_paths else "  (no files uploaded)"
    return (
        f"@DocumentParser Please begin document analysis for this deal.\n\n"
        f"**Company:** {company_name}\n"
        f"**Files:**\n{paths_str}\n"
        f"**Notes:** {notes or 'None'}\n\n"
        f"Extract all financial data, contracts, cap table, and key dates. "
        f"Post your SIGNAL:parsed_documents result and then @mention @FinancialAnalyst and @LegalRisk."
    )


# -------------------------------------------------------------------
# Agent setup
# -------------------------------------------------------------------

def create_orchestrator() -> Agent:
    agent_id, api_key = load_agent_config("orchestrator")

    adapter = LangGraphAdapter(
        llm=llm.bind(system=ORCHESTRATOR_PROMPT),
        checkpointer=InMemorySaver(),
        additional_tools=[parse_agent_signal, build_deal_kickoff_message],
    )

    return Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
    )


async def run_orchestrator():
    logger.info("Orchestrator starting (GPT-4o via AI/ML API)...")
    agent = create_orchestrator()
    await agent.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_orchestrator())
