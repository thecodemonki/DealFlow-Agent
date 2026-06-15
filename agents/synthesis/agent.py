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
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from thenvoi import Agent
from thenvoi.adapters import LangGraphAdapter
from thenvoi.config import load_agent_config

from shared.models import FinancialAnalysis, LegalRiskAnalysis, MarketResearch, InvestmentMemo, verdict_from_deal_score
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

def normalize_deal_score_fields(memo: dict) -> dict:
    """Clamp deal_score to 0–100, align deal_verdict to score bands, coerce risks_flagged_count."""
    out = dict(memo)
    try:
        s = int(out.get("deal_score", 50))
    except (TypeError, ValueError):
        s = 50
    out["deal_score"] = max(0, min(100, s))
    out["deal_verdict"] = verdict_from_deal_score(out["deal_score"])
    try:
        out["risks_flagged_count"] = max(0, int(out.get("risks_flagged_count", 0)))
    except (TypeError, ValueError):
        out["risks_flagged_count"] = 0
    return out


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
    Generate a formatted PDF investment memo from the InvestmentMemo data.
    Returns the file path of the saved PDF.
    """
    memo_data = normalize_deal_score_fields(dict(memo_data))
    # Guardrail: if the LLM hallucinated a placeholder valuation despite no financial docs,
    # replace it with the correct no-data message
    financial_highlights = memo_data.get("financial_highlights", "")
    placeholder_patterns = [
        "$50",
        "$95B",
        "illustrative",
        "provisional valuation band",
        "large-cap comparables",
        "in this demo",
    ]
    if any(p.lower() in financial_highlights.lower() for p in placeholder_patterns):
        memo_data["financial_highlights"] = (
            "No financial documents were provided. Valuation cannot be modelled "
            "from public data alone. Recommend requesting audited financials before proceeding."
        )
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
        from reportlab.lib.colors import HexColor

        output_dir = Path(os.environ.get("UPLOAD_DIR", "./uploads"))
        output_dir.mkdir(exist_ok=True)

        company = memo_data.get("company_name", "Unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = output_dir / f"investment_memo_{company.replace(' ', '_')}_{timestamp}.pdf"

        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter,
                                rightMargin=inch, leftMargin=inch,
                                topMargin=inch, bottomMargin=inch)

        styles = getSampleStyleSheet()
        accent = HexColor("#1a365d")

        title_style = ParagraphStyle("Title", parent=styles["Heading1"],
                                     fontSize=20, textColor=accent, spaceAfter=6)
        h2_style = ParagraphStyle("H2", parent=styles["Heading2"],
                                  fontSize=13, textColor=accent, spaceAfter=4)
        body_style = styles["BodyText"]
        body_style.spaceAfter = 6

        recommendation = memo_data.get("recommendation", "").upper()
        confidence = memo_data.get("confidence", "").upper()
        rec_color = {"INVEST": "#276749", "CONDITIONAL": "#975a16", "PASS": "#9b2c2c"}.get(recommendation, "#2d3748")

        verdict_style = ParagraphStyle("Verdict", parent=styles["Heading1"],
                                       fontSize=16, textColor=HexColor(rec_color))

        content = []
        content.append(Paragraph(f"Investment Memo: {company}", title_style))
        content.append(Paragraph(f"Generated by DealFlow AI — {datetime.now().strftime('%B %d, %Y')}", body_style))
        content.append(HRFlowable(width="100%", thickness=2, color=accent))
        content.append(Spacer(1, 12))

        content.append(Paragraph(f"Recommendation: {recommendation} ({confidence} confidence)", verdict_style))
        ds = memo_data.get("deal_score", 50)
        dv = memo_data.get("deal_verdict", verdict_from_deal_score(ds))
        rc = memo_data.get("risks_flagged_count", 0)
        content.append(
            Paragraph(
                f"Deal score: {ds}/100 — {dv} — {rc} risk(s) flagged (Judge)",
                body_style,
            )
        )
        content.append(Spacer(1, 12))

        sections = [
            ("Executive Summary", "executive_summary"),
            ("Financial Highlights", "financial_highlights"),
            ("Legal Risk Assessment", "legal_risks"),
            ("Market Position", "market_position"),
            ("Suggested Deal Terms", "deal_terms_suggested"),
        ]

        for title, key in sections:
            content.append(Paragraph(title, h2_style))
            content.append(Paragraph(memo_data.get(key, "N/A"), body_style))
            content.append(Spacer(1, 8))

        red_flags = memo_data.get("red_flags", [])
        if red_flags:
            content.append(Paragraph("Red Flags", h2_style))
            for flag in red_flags:
                content.append(Paragraph(f"• {flag}", body_style))
            content.append(Spacer(1, 8))

        content.append(HRFlowable(width="100%", thickness=1, color=HexColor("#e2e8f0")))
        content.append(Paragraph("Confidential — Generated by DealFlow AI multi-agent system", body_style))

        doc.build(content)
        logger.info(f"PDF memo saved to {pdf_path}")
        return str(pdf_path)

    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        return f"ERROR: {e}"


@tool
def format_final_signal(memo: dict) -> str:
    """Format InvestmentMemo as a SIGNAL message for the Band room.
    Posts a concise summary — full memo is saved to PDF separately.
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
