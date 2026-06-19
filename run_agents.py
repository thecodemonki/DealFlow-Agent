"""
run_agents.py — Start all DealFlow AI agents concurrently.

This script launches all 6 agents in a single process using asyncio.
Each agent connects to Band via WebSocket and listens for @mentions.

Usage:
    uv run python run_agents.py

To run agents individually (useful for debugging):
    uv run python -m agents.orchestrator.agent
    uv run python -m agents.document_parser.agent
    etc.
"""

import asyncio
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("dealflow")


async def main():
    # Import here so .env is loaded first
    from agents.orchestrator.agent import run_orchestrator
    from agents.document_parser.agent import run_document_parser
    from agents.financial_analyst.agent import run_financial_analyst
    from agents.legal_risk.agent import run_legal_risk
    from agents.web_research.agent import run_web_research
    from agents.synthesis.agent import run_synthesis

    print("Agent worker starting...", flush=True)
    while True:
        try:
            print("Starting all 6 agents...", flush=True)
            await asyncio.gather(
                run_orchestrator(),
                run_document_parser(),
                run_financial_analyst(),
                run_legal_risk(),
                run_web_research(),
                run_synthesis(),
            )
        except KeyboardInterrupt:
            print("Shutting down agents...")
            break
        except Exception as e:
            print(f"Agent crash: {e} — restarting in 5s...", flush=True)
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
