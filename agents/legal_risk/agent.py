"""
Legal Risk Agent — DealFlow AI
Framework: CrewAI
Model: Meta-Llama-3.1-70B-Instruct via Featherless

Strong at instruction-following and long-context legal document comprehension.
CrewAI framework to demonstrate cross-framework diversity via Band.
"""

import asyncio
import json
import logging
import os

from dotenv import load_dotenv
from crewai import Agent as CrewAgent, Task, Crew, LLM
from thenvoi import Agent
from thenvoi.adapters import CrewAIAdapter
from thenvoi.config import load_agent_config

from shared.prompts import LEGAL_RISK_PROMPT

load_dotenv()
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Model — Llama-3.1-70B via Featherless (strong legal comprehension)
# CrewAI uses LiteLLM under the hood; "openai/" prefix = OpenAI-compatible API
# -------------------------------------------------------------------
llm = LLM(
    model="openai/meta-llama/Meta-Llama-3.1-70B-Instruct",
    base_url="https://api.featherless.ai/v1",
    api_key=os.environ["FEATHERLESS_API_KEY"],
    temperature=0.1,
)

# -------------------------------------------------------------------
# CrewAI agent + task definition
# The Band SDK's CrewAI adapter passes incoming Band messages as the task input.
# -------------------------------------------------------------------

legal_agent = CrewAgent(
    role="Legal Risk Analyst",
    goal=(
        "Identify all legal and contractual risks in M&A target company documents. "
        "Flag deal-breakers immediately. Be conservative — missing a risk is worse than over-flagging."
    ),
    backstory=(
        "You are a senior M&A attorney with 15 years of experience reviewing acquisition targets "
        "across tech, SaaS, and fintech. You specialize in IP, change-of-control clauses, "
        "indemnification exposure, and regulatory compliance risks."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

legal_task = Task(
    description=(
        "Analyze the following company documents and parsed data for legal risks. "
        "Input: {input}\n\n"
        "Output a JSON object matching the LegalRiskAnalysis schema:\n"
        "- risk_level: 'low' | 'medium' | 'high' | 'deal_breaker'\n"
        "- ip_issues: list of IP ownership concerns\n"
        "- liability_flags: list of liability exposure issues\n"
        "- change_of_control_clauses: list of CoC triggers found\n"
        "- deal_breakers: critical issues that should stop the deal\n"
        "- recommendations: suggested conditions or mitigations\n"
        "- requires_human_review: bool\n"
        "- human_review_reason: explanation if true\n"
        "- summary: 2-3 sentence legal risk summary\n\n"
        "Format output as:\nSIGNAL:legal_risk\n{{json}}"
    ),
    agent=legal_agent,
    expected_output="SIGNAL:legal_risk followed by a valid JSON object",
)

legal_crew = Crew(
    agents=[legal_agent],
    tasks=[legal_task],
    verbose=True,
)

# -------------------------------------------------------------------
# Agent setup
# -------------------------------------------------------------------

def create_legal_risk() -> Agent:
    agent_id, api_key = load_agent_config("legal_risk")

    adapter = CrewAIAdapter(crew=legal_crew)

    return Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
    )


async def run_legal_risk():
    logger.info("Legal Risk Agent starting (Llama-3.1-70B via Featherless, CrewAI)...")
    agent = create_legal_risk()
    await agent.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_legal_risk())
