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
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from shared.models import verdict_from_deal_score

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
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# In-memory deal status store (replace with DB in production)
deals: dict[str, dict] = {}


class DealCompleteBody(BaseModel):
    """Webhook body when Synthesis finishes: PDF path plus optional memo summary for the UI."""

    memo_path: str = Field(..., description="Absolute or project-relative path to the generated PDF")
    memo_summary: Optional[dict[str, Any]] = Field(
        default=None,
        description="SIGNAL:investment_memo-style fields: deal_score, deal_verdict, risks_flagged_count, company_name, recommendation, confidence, etc.",
    )


def _latest_complete_deal_id() -> Optional[str]:
    """Most recently completed deal that has a memo PDF on disk."""
    best_id: Optional[str] = None
    best_ts = ""
    for did, d in deals.items():
        if d.get("status") != "complete":
            continue
        mp = d.get("memo_path")
        if not mp or not Path(mp).exists():
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
    }
    return JSONResponse({"deal_id": deal_id, "status": "draft"})


@app.post("/analyze")
async def submit_deal(
    company_name: str = Form(...),
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

    return JSONResponse({"file_path": rel, "status": "uploaded"})


@app.get("/deals/{deal_id}")
async def get_deal_status(deal_id: str):
    """Check the status of a deal analysis."""
    if deal_id not in deals:
        raise HTTPException(status_code=404, detail="Deal not found")
    return deals[deal_id]


@app.get("/deals")
async def list_deals():
    """List all deal analyses."""
    return list(deals.values())


@app.get("/deals/{deal_id}/memo")
async def download_memo(deal_id: str):
    """Download the final investment memo PDF for a completed deal."""
    if deal_id not in deals:
        raise HTTPException(status_code=404, detail="Deal not found")
    memo_path = deals[deal_id].get("memo_path")
    if not memo_path or not Path(memo_path).exists():
        raise HTTPException(status_code=404, detail="Memo not yet generated")
    return FileResponse(memo_path, media_type="application/pdf", filename=Path(memo_path).name)


@app.get("/memo/latest")
async def download_latest_memo():
    """Download the PDF for the most recently completed deal (same as README `curl /memo/latest`)."""
    deal_id = _latest_complete_deal_id()
    if not deal_id:
        raise HTTPException(status_code=404, detail="No completed memo available")
    memo_path = deals[deal_id]["memo_path"]
    return FileResponse(memo_path, media_type="application/pdf", filename=Path(memo_path).name)


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

    Body JSON: {"memo_path": "/path/to.pdf", "memo_summary": { ... optional fields including
    deal_score, deal_verdict, risks_flagged_count, company_name, recommendation, confidence }}
    """
    if deal_id not in deals:
        raise HTTPException(status_code=404, detail="Deal not found")
    deals[deal_id]["status"] = "complete"
    deals[deal_id]["memo_path"] = body.memo_path
    deals[deal_id]["completed_at"] = datetime.utcnow().isoformat()
    if body.memo_summary is not None:
        deals[deal_id]["memo_summary"] = body.memo_summary
    return {"status": "updated", "deal_id": deal_id}
