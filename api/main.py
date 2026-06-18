"""
FastAPI Gateway — DealFlow AI

Thin REST layer that:
1. Accepts deal submissions (company name + document uploads)
2. Saves files locally for agents to access
3. Triggers the Orchestrator via Band by sending it a message
4. Returns deal status

Run with: uvicorn api.main:app --reload --port 8000
"""

import json
import logging
import os
import uuid
from pathlib import Path
from datetime import datetime

import httpx
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from shared.models import verdict_from_deal_score
from shared.pdf_memo import build_memo_pdf_bytes

load_dotenv()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DealFlow AI",
    description="Multi-agent M&A due diligence platform powered by Band",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "./uploads"))
UPLOAD_DIR.mkdir(exist_ok=True)


def _relative_upload_path(deal_id: str, safe_name: str) -> str:
    """Repo-relative path for Band messages and Librarian (matches files under UPLOAD_DIR)."""
    return f"uploads/{deal_id}/{safe_name}"


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
INDEX_HTML = FRONTEND_DIR / "index.html"
CHARACTERS_DIR = FRONTEND_DIR / "characters"

app.mount("/characters", StaticFiles(directory=str(CHARACTERS_DIR)), name="characters")
# Serves frontend/ at /static (e.g. agentmax-cover.png); use absolute path so cwd does not matter.
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# In-memory deal status store (replace with DB in production)
deals: dict[str, dict] = {}


class DealCompleteBody(BaseModel):
    """Webhook body when Synthesis finishes: memo summary for the UI and on-demand PDF."""

    memo_path: Optional[str] = Field(
        default=None,
        description="Deprecated — PDFs are generated on download from memo_summary",
    )
    memo_summary: Optional[dict[str, Any]] = Field(
        default=None,
        description="SIGNAL:investment_memo-style fields: deal_score, deal_verdict, risks_flagged_count, company_name, recommendation, confidence, etc.",
    )


class DealChatBody(BaseModel):
    """Persist agent chat replay and optional metadata from the UI."""

    messages: list[dict[str, Any]] = Field(default_factory=list)
    industry: Optional[str] = None
    company_name: Optional[str] = None


class DealMessageBody(BaseModel):
    """Follow-up question from the UI after analysis completes."""

    message: str


DEALS_INDEX_PATH = UPLOAD_DIR / "deals_index.json"


def _deal_has_memo_pdf(d: dict[str, Any]) -> bool:
    return d.get("status") == "complete" and bool(d.get("memo_summary"))


def _memo_data_for_deal(d: dict[str, Any]) -> dict[str, Any]:
    summary = dict(d.get("memo_summary") or {})
    summary.setdefault("company_name", d.get("company_name") or "Unknown")
    return summary


def _build_deal_memo_response(deal_id: str) -> Response:
    d = deals[deal_id]
    summary = d.get("memo_summary")
    if not summary:
        raise HTTPException(status_code=404, detail="Memo not yet generated")
    try:
        pdf_bytes = build_memo_pdf_bytes(_memo_data_for_deal(d))
    except Exception as e:
        logger.error("On-demand PDF generation failed for deal %s: %s", deal_id, e)
        raise HTTPException(status_code=500, detail="Failed to generate memo PDF") from e
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": 'attachment; filename="agentmax-memo.pdf"'
        },
    )


def _scores_from_memo_summary(raw: Optional[dict[str, Any]]) -> dict[str, Any]:
    raw = dict(raw or {})
    try:
        score = int(raw.get("deal_score", 50))
    except (TypeError, ValueError):
        score = 50
    score = max(0, min(100, score))
    verdict = raw.get("deal_verdict") or verdict_from_deal_score(score)
    try:
        risks = max(0, int(raw.get("risks_flagged_count", 0)))
    except (TypeError, ValueError):
        risks = 0
    return {"deal_score": score, "deal_verdict": str(verdict).upper(), "risks_flagged_count": risks}


def _parse_iso_timestamp(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _duration_seconds_between(created: Optional[str], completed: Optional[str]) -> Optional[int]:
    start = _parse_iso_timestamp(created or "")
    end = _parse_iso_timestamp(completed or "")
    if not start or not end:
        return None
    return max(0, round((end - start).total_seconds()))


def _analysis_duration_seconds(d: dict[str, Any]) -> Optional[int]:
    stored = d.get("analysis_duration_seconds")
    if stored is not None:
        try:
            return max(0, int(stored))
        except (TypeError, ValueError):
            pass
    return _duration_seconds_between(d.get("created_at"), d.get("completed_at"))


def _apply_memo_summary_to_deal(deal: dict[str, Any], memo_summary: Optional[dict[str, Any]]) -> None:
    if memo_summary is None:
        return
    deal["memo_summary"] = memo_summary
    scores = _scores_from_memo_summary(memo_summary)
    deal.update(scores)
    if memo_summary.get("company_name"):
        deal["company_name"] = memo_summary["company_name"]


def _scores_from_deal(d: dict[str, Any]) -> dict[str, Any]:
    ms = d.get("memo_summary")
    if isinstance(ms, dict) and ms.get("deal_score") is not None:
        return _scores_from_memo_summary(ms)
    if d.get("deal_score") is not None:
        try:
            score = int(d["deal_score"])
        except (TypeError, ValueError):
            score = 50
        score = max(0, min(100, score))
        verdict = d.get("deal_verdict") or verdict_from_deal_score(score)
        try:
            risks = max(0, int(d.get("risks_flagged_count", 0)))
        except (TypeError, ValueError):
            risks = 0
        return {"deal_score": score, "deal_verdict": str(verdict).upper(), "risks_flagged_count": risks}
    return _scores_from_memo_summary(ms if isinstance(ms, dict) else {})


def _public_deal_row(deal_id: str, d: dict[str, Any]) -> dict[str, Any]:
    """Deal list/detail shape with top-level score fields for the sidebar."""
    scores = _scores_from_deal(d)
    row = dict(d)
    row["id"] = deal_id
    row["company_name"] = d.get("company_name") or ""
    row["industry"] = d.get("industry")
    row["deal_score"] = scores["deal_score"]
    row["deal_verdict"] = scores["deal_verdict"]
    row["risks_flagged_count"] = scores["risks_flagged_count"]
    row["completed_at"] = d.get("completed_at")
    row["analysis_duration_seconds"] = _analysis_duration_seconds(d)
    row["has_memo_pdf"] = _deal_has_memo_pdf(d)
    return row


def _save_deals_index() -> None:
    try:
        DEALS_INDEX_PATH.write_text(json.dumps(deals, indent=2, default=str))
    except OSError as e:
        logger.error("Failed to persist deals index: %s", e)


def _load_deals_index() -> None:
    if not DEALS_INDEX_PATH.is_file():
        return
    try:
        loaded = json.loads(DEALS_INDEX_PATH.read_text())
        if isinstance(loaded, dict):
            deals.update(loaded)
            logger.info("Loaded %d deals from %s", len(loaded), DEALS_INDEX_PATH)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not load deals index: %s", e)


def _latest_complete_deal_id() -> Optional[str]:
    """Most recently completed deal that has memo_summary for on-demand PDF."""
    best_id: Optional[str] = None
    best_ts = ""
    for did, d in deals.items():
        if not _deal_has_memo_pdf(d):
            continue
        ts = str(d.get("completed_at") or d.get("created_at") or "")
        if ts >= best_ts:
            best_ts = ts
            best_id = did
    return best_id


def _public_memo_summary(deal_id: str) -> dict[str, Any]:
    d = deals[deal_id]
    raw = dict(d.get("memo_summary") or {})
    try:
        score = int(raw.get("deal_score", 50))
    except (TypeError, ValueError):
        score = 50
    score = max(0, min(100, score))
    verdict = verdict_from_deal_score(score)
    try:
        risks = max(0, int(raw.get("risks_flagged_count", 0)))
    except (TypeError, ValueError):
        risks = 0
    return {
        "deal_id": deal_id,
        "company_name": raw.get("company_name") or d.get("company_name"),
        "deal_score": score,
        "deal_verdict": verdict,
        "risks_flagged_count": risks,
        "recommendation": raw.get("recommendation"),
        "confidence": raw.get("confidence"),
    }


def _full_deal_summary(deal_id: str) -> dict[str, Any]:
    d = deals[deal_id]
    raw = dict(d.get("memo_summary") or {})
    has_memo_pdf = _deal_has_memo_pdf(d)
    out = _public_memo_summary(deal_id)
    out.update(
        {
            "status": d.get("status"),
            "completed_at": d.get("completed_at"),
            "band_room_id": d.get("band_room_id"),
            "has_memo_pdf": has_memo_pdf,
            "executive_summary": raw.get("executive_summary"),
            "financial_highlights": raw.get("financial_highlights"),
            "market_position": raw.get("market_position"),
            "legal_risks": raw.get("legal_risks"),
            "red_flags": raw.get("red_flags") or [],
            "investment_thesis": raw.get("investment_thesis")
            or raw.get("deal_terms_suggested")
            or raw.get("recommendation"),
            "recommendation": raw.get("recommendation"),
            "confidence": raw.get("confidence"),
        }
    )
    return out


_load_deals_index()


# -------------------------------------------------------------------
# Band API helper — sends a message to the Orchestrator's inbox room.
# The Orchestrator agent is always running and listening for incoming messages.
# -------------------------------------------------------------------

BAND_API_BASE = "https://api.band.ai/api/v1/agent"

async def trigger_orchestrator(deal_id: str, company_name: str, file_paths: list[str], notes: str) -> str:
    """
    Sends the deal request to the Orchestrator agent via Band REST API.
    The Orchestrator's Band agent_id is used to create a direct room.
    Returns the Band room ID for tracking.
    """
    import yaml
    with open("agent_config.yaml") as f:
        config = yaml.safe_load(f)

    orchestrator_config = config["orchestrator"]
    orchestrator_agent_id = orchestrator_config["agent_id"]
    orchestrator_api_key = orchestrator_config["api_key"]

    headers = {"X-API-Key": orchestrator_api_key, "Content-Type": "application/json"}

    async with httpx.AsyncClient() as client:
        # Create a new Band chat room for this deal
        room_name = f"Deal: {company_name} [{deal_id[:8]}]"
        create_room_resp = await client.post(
            f"{BAND_API_BASE}/chats",
            headers=headers,
            json={"name": room_name},
        )
        if create_room_resp.status_code not in (200, 201):
            raise HTTPException(status_code=500, detail=f"Failed to create Band room: {create_room_resp.text}")

        room_data = create_room_resp.json()
        room_id = room_data.get("id") or room_data.get("chat_id")

        # Send the deal request as the first message in the room
        message_body = {
            "deal_id": deal_id,
            "company_name": company_name,
            "file_paths": file_paths,
            "notes": notes,
            "instructions": (
                f"New deal request received. Please begin analysis for {company_name}. "
                f"Files are available at: {', '.join(file_paths)}. "
                f"Create a deal room, recruit all specialist agents, and kick off the pipeline."
            ),
        }

        send_resp = await client.post(
            f"{BAND_API_BASE}/chats/{room_id}/messages",
            headers=headers,
            json={
                "content": f"@Orchestrator {json.dumps(message_body)}",
                "mentions": [orchestrator_agent_id],
            },
        )

        if send_resp.status_code not in (200, 201):
            logger.warning(f"Message send returned {send_resp.status_code}: {send_resp.text}")

        return room_id


async def _post_band_document_uploaded(room_id: str, file_path: str) -> bool:
    """
    Publish DOCUMENT_UPLOADED to the deal Band room so the Orchestrator / Librarian pipeline can pick it up.
    """
    import yaml

    try:
        with open("agent_config.yaml") as f:
            config = yaml.safe_load(f)
        orchestrator_config = config["orchestrator"]
        orchestrator_agent_id = orchestrator_config["agent_id"]
        orchestrator_api_key = orchestrator_config["api_key"]
        headers = {"X-API-Key": orchestrator_api_key, "Content-Type": "application/json"}
        content = f"DOCUMENT_UPLOADED: {file_path}"
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{BAND_API_BASE}/chats/{room_id}/messages",
                headers=headers,
                json={
                    "content": content,
                    "mentions": [orchestrator_agent_id],
                },
            )
            if resp.status_code not in (200, 201):
                logger.warning(
                    "DOCUMENT_UPLOADED post failed %s: %s",
                    resp.status_code,
                    resp.text[:500],
                )
                return False
        return True
    except Exception as e:
        logger.error("Failed to post DOCUMENT_UPLOADED to Band: %s", e)
        return False


async def _post_band_user_message(room_id: str, message: str) -> bool:
    """Publish a user follow-up question to the deal Band room for the Orchestrator."""
    import yaml

    try:
        with open("agent_config.yaml") as f:
            config = yaml.safe_load(f)
        orchestrator_config = config["orchestrator"]
        orchestrator_agent_id = orchestrator_config["agent_id"]
        orchestrator_api_key = orchestrator_config["api_key"]
        headers = {"X-API-Key": orchestrator_api_key, "Content-Type": "application/json"}
        content = f"USER: {message}"
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{BAND_API_BASE}/chats/{room_id}/messages",
                headers=headers,
                json={
                    "content": content,
                    "mentions": [orchestrator_agent_id],
                },
            )
            if resp.status_code not in (200, 201):
                logger.warning(
                    "USER message post failed %s: %s",
                    resp.status_code,
                    resp.text[:500],
                )
                return False
        return True
    except Exception as e:
        logger.error("Failed to post USER message to Band: %s", e)
        return False


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.get("/")
async def root():
    """Serve the web UI when built frontend is present; otherwise JSON health stub."""
    if INDEX_HTML.is_file():
        return FileResponse(INDEX_HTML, media_type="text/html")
    return {"service": "DealFlow AI", "status": "running", "agents": 6}


@app.post("/deals")
async def create_draft_deal():
    """
    Create a deal and uploads directory so the UI can POST /deals/{deal_id}/upload
    before Run Analysis (Band room is created when /analyze runs).
    """
    deal_id = str(uuid.uuid4())
    deal_dir = UPLOAD_DIR / deal_id
    deal_dir.mkdir(parents=True, exist_ok=True)
    deals[deal_id] = {
        "id": deal_id,
        "company_name": "",
        "status": "draft",
        "file_paths": [],
        "band_room_id": None,
        "created_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "memo_path": None,
        "memo_summary": None,
        "deal_score": None,
        "deal_verdict": None,
        "risks_flagged_count": None,
        "analysis_duration_seconds": None,
        "chat_messages": [],
        "industry": None,
    }
    _save_deals_index()
    return JSONResponse({"deal_id": deal_id, "status": "draft"})


@app.post("/analyze")
async def submit_deal(
    company_name: str = Form(...),
    industry: str = Form(default=""),
    notes: str = Form(default=""),
    deal_id: str = Form(default=""),
    files: list[UploadFile] = File(default=[]),
):
    """
    Submit a company for M&A due diligence analysis.
    Optional deal_id continues a draft created via POST /deals with files from POST /deals/{id}/upload.
    """
    existing_id = (deal_id or "").strip()
    reuse = bool(existing_id and existing_id in deals)
    if reuse:
        prev = deals[existing_id]
        if prev.get("status") == "complete":
            raise HTTPException(status_code=400, detail="Deal already complete")
        if prev.get("band_room_id"):
            raise HTTPException(
                status_code=400,
                detail="Analysis already triggered for this deal. Start a new analysis from the sidebar.",
            )
        deal_id = existing_id
        file_paths = list(prev.get("file_paths") or [])
        created_at = prev.get("created_at") or datetime.utcnow().isoformat()
    else:
        deal_id = str(uuid.uuid4())
        file_paths = []
        created_at = datetime.utcnow().isoformat()

    deal_dir = UPLOAD_DIR / deal_id
    deal_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        if file.filename:
            safe_name = Path(file.filename).name
            dest = deal_dir / safe_name
            dest.write_bytes(await file.read())
            rel = _relative_upload_path(deal_id, safe_name)
            if rel not in file_paths:
                file_paths.append(rel)
            logger.info("Saved %s for deal %s (%s)", safe_name, deal_id, rel)

    industry_value = industry.strip() or None

    deals[deal_id] = {
        "id": deal_id,
        "company_name": company_name,
        "status": "triggered",
        "file_paths": file_paths,
        "band_room_id": None,
        "created_at": created_at,
        "completed_at": None,
        "memo_path": None,
        "memo_summary": None,
        "deal_score": None,
        "deal_verdict": None,
        "risks_flagged_count": None,
        "analysis_duration_seconds": None,
        "chat_messages": list((prev.get("chat_messages") or []) if reuse else []),
        "industry": industry_value,
    }

    try:
        room_id = await trigger_orchestrator(deal_id, company_name, file_paths, notes)
        deals[deal_id]["band_room_id"] = room_id
        deals[deal_id]["status"] = "in_progress"
        for fp in file_paths:
            await _post_band_document_uploaded(room_id, fp)
    except Exception as e:
        logger.error(f"Failed to trigger orchestrator: {e}")
        deals[deal_id]["status"] = "error"
        deals[deal_id]["error"] = str(e)

    _save_deals_index()
    return JSONResponse({
        "deal_id": deal_id,
        "company_name": company_name,
        "status": deals[deal_id]["status"],
        "band_room_id": deals[deal_id].get("band_room_id"),
        "files_uploaded": len(file_paths),
        "message": "Analysis triggered. Agents are collaborating in your Band room.",
    })


@app.post("/deals/{deal_id}/upload")
async def upload_deal_document(deal_id: str, file: UploadFile = File(...)):
    """
    Save an additional document for an existing deal and notify the Band room (DOCUMENT_UPLOADED).
    """
    if deal_id not in deals:
        raise HTTPException(status_code=404, detail="Deal not found")
    if not file.filename:
        raise HTTPException(status_code=400, detail="File name is required")

    deal_dir = UPLOAD_DIR / deal_id
    deal_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename).name
    dest = deal_dir / safe_name
    dest.write_bytes(await file.read())
    rel = _relative_upload_path(deal_id, safe_name)

    paths = deals[deal_id].setdefault("file_paths", [])
    if rel not in paths:
        paths.append(rel)

    room_id = deals[deal_id].get("band_room_id")
    if room_id:
        await _post_band_document_uploaded(room_id, rel)
    else:
        logger.warning("upload: deal %s has no band_room_id; file saved but not broadcast", deal_id)

    _save_deals_index()
    return JSONResponse({"file_path": rel, "status": "uploaded"})


@app.get("/deals/{deal_id}")
async def get_deal_status(deal_id: str):
    """Check the status of a deal analysis."""
    if deal_id not in deals:
        raise HTTPException(status_code=404, detail="Deal not found")
    return _public_deal_row(deal_id, deals[deal_id])


@app.get("/deals")
async def list_deals():
    """List all deal analyses."""
    return [_public_deal_row(did, d) for did, d in deals.items()]


@app.get("/deals/{deal_id}/memo")
async def download_memo(deal_id: str):
    """Generate and download the investment memo PDF from stored memo_summary."""
    if deal_id not in deals:
        raise HTTPException(status_code=404, detail="Deal not found")
    return _build_deal_memo_response(deal_id)


@app.get("/deals/{deal_id}/summary")
async def deal_memo_summary(deal_id: str):
    """JSON summary for a specific deal: score, verdict, memo sections, PDF availability."""
    if deal_id not in deals:
        raise HTTPException(status_code=404, detail="Deal not found")
    return _full_deal_summary(deal_id)


@app.get("/memo/latest")
async def download_latest_memo():
    """Download the PDF for the most recently completed deal (generated on demand)."""
    deal_id = _latest_complete_deal_id()
    if not deal_id:
        raise HTTPException(status_code=404, detail="No completed memo available")
    return _build_deal_memo_response(deal_id)


@app.get("/memo/latest/summary")
async def latest_memo_summary():
    """JSON for the UI: Deal Score, verdict, and Judge risk count."""
    deal_id = _latest_complete_deal_id()
    if not deal_id:
        raise HTTPException(status_code=404, detail="No completed memo available")
    return _public_memo_summary(deal_id)


@app.post("/deals/{deal_id}/complete")
async def mark_deal_complete(deal_id: str, body: DealCompleteBody):
    """
    Called by the Synthesis agent (via webhook) when the memo is ready.

    Body JSON: {"memo_summary": { deal_score, deal_verdict, risks_flagged_count,
    company_name, recommendation, confidence, executive_summary, ... }}
    """
    if deal_id not in deals:
        raise HTTPException(status_code=404, detail="Deal not found")
    if not body.memo_summary:
        raise HTTPException(status_code=400, detail="memo_summary is required")
    deals[deal_id]["status"] = "complete"
    deals[deal_id]["memo_path"] = None
    deals[deal_id]["completed_at"] = datetime.utcnow().isoformat()
    deals[deal_id]["analysis_duration_seconds"] = _duration_seconds_between(
        deals[deal_id].get("created_at"),
        deals[deal_id]["completed_at"],
    )
    _apply_memo_summary_to_deal(deals[deal_id], body.memo_summary)
    _save_deals_index()
    return {"status": "updated", "deal_id": deal_id}


@app.post("/deals/{deal_id}/chat")
async def save_deal_chat(deal_id: str, body: DealChatBody):
    """Persist agent chat replay from the UI for restoring past deal views."""
    if deal_id not in deals:
        raise HTTPException(status_code=404, detail="Deal not found")
    deals[deal_id]["chat_messages"] = body.messages
    if body.industry:
        deals[deal_id]["industry"] = body.industry
    if body.company_name:
        deals[deal_id]["company_name"] = body.company_name
    _save_deals_index()
    return {"status": "saved", "deal_id": deal_id, "message_count": len(body.messages)}


@app.post("/deals/{deal_id}/message")
async def post_deal_message(deal_id: str, body: DealMessageBody):
    """Post a user follow-up question to the deal's Band room."""
    message = (body.message or "").strip()
    deal_found = deal_id in deals
    room_id = deals[deal_id].get("band_room_id") if deal_found else None
    logger.info(
        "POST /deals/%s/message: deal_found=%s band_room_id=%s message_len=%d",
        deal_id,
        deal_found,
        room_id,
        len(message),
    )
    if not deal_found:
        logger.warning("POST /deals/%s/message: deal not found", deal_id)
        raise HTTPException(status_code=404, detail="Deal not found")
    if not message:
        logger.warning("POST /deals/%s/message: empty message", deal_id)
        raise HTTPException(status_code=400, detail="Message is required")
    if not room_id:
        logger.warning("POST /deals/%s/message: no band_room_id on deal", deal_id)
        raise HTTPException(status_code=400, detail="No Band room for this deal")
    ok = await _post_band_user_message(room_id, message)
    if not ok:
        logger.warning("POST /deals/%s/message: Band post failed for room %s", deal_id, room_id)
        raise HTTPException(status_code=502, detail="Failed to send message")
    return {"status": "sent"}
