"""
Document Parser Agent — DealFlow AI
Framework: Pydantic AI
Model: Mistral-7B-Instruct via Featherless

Specialist in extracting structured data from company documents.
Fast and precise — doesn't need deep reasoning, needs structured extraction.
"""

import asyncio
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic_ai import Agent as PAAgent
from pydantic_ai.models.openai import OpenAIModel
from thenvoi import Agent
from thenvoi.adapters import PydanticAIAdapter
from thenvoi.config import load_agent_config

from shared.models import ParsedDocuments
from shared.prompts import DOCUMENT_PARSER_PROMPT

load_dotenv()
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Model — Mistral-7B via Featherless (fast structured extraction)
# -------------------------------------------------------------------
featherless_client = AsyncOpenAI(
    base_url="https://api.featherless.ai/v1",
    api_key=os.environ["FEATHERLESS_API_KEY"],
)

model = OpenAIModel(
    "mistralai/Mistral-7B-Instruct-v0.3",
    openai_client=featherless_client,
)

# -------------------------------------------------------------------
# Pydantic AI agent with file reading tools
# -------------------------------------------------------------------

pa_agent = PAAgent(
    model=model,
    system_prompt=DOCUMENT_PARSER_PROMPT,
    result_type=str,  # Returns structured JSON string posted back to Band
)


@pa_agent.tool_plain
def read_pdf_file(file_path: str) -> str:
    """Read and extract text from a PDF file."""
    try:
        import pypdf
        path = Path(file_path)
        if not path.exists():
            return f"ERROR: File not found at {file_path}"
        reader = pypdf.PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text[:15000]  # Cap at 15k chars to stay within context
    except Exception as e:
        return f"ERROR reading {file_path}: {e}"


@pa_agent.tool_plain
def read_text_file(file_path: str) -> str:
    """Read a plain text or CSV file."""
    try:
        return Path(file_path).read_text(encoding="utf-8")[:10000]
    except Exception as e:
        return f"ERROR reading {file_path}: {e}"


@pa_agent.tool_plain
def format_signal(parsed_data: dict) -> str:
    """
    Format the ParsedDocuments dict as a SIGNAL message for the Band room.
    This is what gets posted back via Band after parsing is complete.
    """
    try:
        validated = ParsedDocuments(**parsed_data)
        return f"SIGNAL:parsed_documents\n{validated.model_dump_json(indent=2)}"
    except Exception as e:
        return f"SIGNAL:parsed_documents\n{json.dumps(parsed_data, indent=2)}"


# -------------------------------------------------------------------
# Agent setup
# -------------------------------------------------------------------

def create_document_parser() -> Agent:
    agent_id, api_key = load_agent_config("document_parser")

    adapter = PydanticAIAdapter(agent=pa_agent)

    return Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
    )


async def run_document_parser():
    logger.info("Document Parser starting (Mistral-7B via Featherless)...")
    agent = create_document_parser()
    await agent.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_document_parser())
