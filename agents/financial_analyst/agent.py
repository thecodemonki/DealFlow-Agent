"""
Financial Analyst Agent — DealFlow AI
Framework: LangGraph
Model: GPT-4o-mini via AI/ML API

Quantitative analysis is driven by SIGNAL:parsed_documents from the Librarian.
There is no hardcoded demo path in this module — behavior is defined in FINANCIAL_ANALYST_PROMPT
(shared/prompts.py): use real parsed figures when present; otherwise post the explicit no-documents message.
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

from shared.models import ParsedDocuments, FinancialAnalysis
from shared.prompts import FINANCIAL_ANALYST_PROMPT

load_dotenv()
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Model — GPT-4o-mini via AI/ML API (reliable tool calling, no Featherless 429s)
# -------------------------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    base_url="https://api.aimlapi.com/v1",
    api_key=os.environ["AIML_API_KEY"],
    temperature=0.0,  # Zero temp for deterministic financial calculations
)

# -------------------------------------------------------------------
# Financial calculation tools
# -------------------------------------------------------------------

@tool
def calculate_cagr(start_value: float, end_value: float, years: float) -> float:
    """Calculate Compound Annual Growth Rate."""
    if start_value <= 0 or years <= 0:
        return 0.0
    return ((end_value / start_value) ** (1 / years) - 1) * 100


@tool
def calculate_runway(cash_balance: float, monthly_burn: float) -> float:
    """Calculate cash runway in months."""
    if monthly_burn <= 0:
        return float("inf")
    return cash_balance / monthly_burn


@tool
def estimate_valuation(arr: float, growth_rate_pct: float, gross_margin_pct: float) -> dict:
    """
    Estimate valuation range using revenue multiples adjusted for growth and margin.
    Returns {low_usd, mid_usd, high_usd, methodology}.
    """
    # Rule of 40: growth_rate + margin. High Rule of 40 = premium multiple.
    rule_of_40 = growth_rate_pct + gross_margin_pct
    if rule_of_40 >= 60:
        multiples = (10, 15, 20)
    elif rule_of_40 >= 40:
        multiples = (6, 9, 12)
    elif rule_of_40 >= 20:
        multiples = (3, 5, 7)
    else:
        multiples = (1, 2, 3)

    return {
        "low_usd": arr * multiples[0],
        "mid_usd": arr * multiples[1],
        "high_usd": arr * multiples[2],
        "methodology": f"ARR multiple ({multiples[0]}-{multiples[2]}x) based on Rule of 40 score of {rule_of_40:.0f}",
    }


@tool
def parse_financial_signal(signal_json: str) -> dict:
    """Parse the ParsedDocuments JSON from a Band message signal."""
    try:
        data = json.loads(signal_json)
        return data
    except Exception as e:
        return {"error": str(e)}


@tool
def format_financial_signal(analysis: dict) -> str:
    """Format FinancialAnalysis as a SIGNAL message for the Band room.
    Keeps content under 2000 chars to comply with Band API limits."""
    payload = json.dumps(analysis, indent=2)
    signal = f"SIGNAL:financial_analysis\n{payload}"
    if len(signal) > 2000:
        compact = json.dumps(analysis, separators=(',', ':'))
        signal = f"SIGNAL:financial_analysis\n{compact}"
    if len(signal) > 2000:
        signal = signal[:1990] + "\n...}"
    return signal


# -------------------------------------------------------------------
# Agent setup
# -------------------------------------------------------------------

def create_financial_analyst() -> Agent:
    agent_id, api_key = load_agent_config("financial_analyst")

    adapter = LangGraphAdapter(
        llm=llm.bind(system=FINANCIAL_ANALYST_PROMPT),
        checkpointer=InMemorySaver(),
        additional_tools=[
            calculate_cagr,
            calculate_runway,
            estimate_valuation,
            parse_financial_signal,
            format_financial_signal,
        ],
    )

    return Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
    )


async def run_financial_analyst():
    logger.info("Financial Analyst starting (GPT-4o-mini via AI/ML API)...")
    agent = create_financial_analyst()
    await agent.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_financial_analyst())
