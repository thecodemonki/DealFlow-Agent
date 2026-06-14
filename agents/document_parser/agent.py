"""
Document Parser Agent — DealFlow AI
Framework: LangGraph
Model: GPT-4o-mini via AI/ML API

Specialist in extracting structured data from company documents.
Fast and precise — doesn't need deep reasoning, needs structured extraction.
"""

import asyncio
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from thenvoi import Agent
from thenvoi.adapters import LangGraphAdapter
from thenvoi.config import load_agent_config

from shared.models import ParsedDocuments
from shared.prompts import DOCUMENT_PARSER_PROMPT

load_dotenv()
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Model — GPT-4o-mini via AI/ML API (reliable tool calling, no Featherless 429s)
# -------------------------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    base_url="https://api.aimlapi.com/v1",
    api_key=os.environ["AIML_API_KEY"],
    temperature=0.1,
)

# -------------------------------------------------------------------
# Tools
# -------------------------------------------------------------------

@tool
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


@tool
def read_text_file(file_path: str) -> str:
    """Read a plain text or CSV file."""
    try:
        return Path(file_path).read_text(encoding="utf-8")[:10000]
    except Exception as e:
        return f"ERROR reading {file_path}: {e}"


@tool
def format_signal(parsed_json: str) -> str:
    """
    Format parsed document data as a SIGNAL message for the Band room.
    Input: JSON string of the parsed document fields.
    Output: SIGNAL:parsed_documents message to post back to Band.
    Keeps content under 2000 chars to comply with Band API limits.
    """
    try:
        data = json.loads(parsed_json)
        validated = ParsedDocuments(**data)
        payload = validated.model_dump_json(indent=2)
    except Exception:
        payload = parsed_json

    signal = f"SIGNAL:parsed_documents\n{payload}"
    if len(signal) > 2000:
        try:
            compact = json.dumps(json.loads(payload), separators=(',', ':'))
            signal = f"SIGNAL:parsed_documents\n{compact}"
        except Exception:
            pass
    if len(signal) > 2000:
        signal = signal[:1990] + "\n...}"
    return signal


# -------------------------------------------------------------------
# Agent setup
# -------------------------------------------------------------------

def create_document_parser() -> Agent:
    agent_id, api_key = load_agent_config("document_parser")

    adapter = LangGraphAdapter(
        llm=llm.bind(system=DOCUMENT_PARSER_PROMPT),
        checkpointer=InMemorySaver(),
        additional_tools=[
            read_pdf_file,
            read_text_file,
            format_signal,
        ],
    )

    return Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
    )


async def run_document_parser():
    logger.info("Document Parser starting (GPT-4o-mini via AI/ML API)...")
    agent = create_document_parser()
    await agent.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_document_parser())
