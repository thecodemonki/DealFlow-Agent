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
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

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

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
INDEX_HTML = FRONTEND_DIR / "index.html"
CHARACTERS_DIR = FRONTEND_DIR / "characters"

app.mount("/characters", StaticFiles(directory=str(CHARACTERS_DIR)), name="characters")

# In-memory deal status store (replace with DB in production)
deals: dict[str, dict] = {}

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


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.get("/")
async def root():
    """Serve the web UI when built frontend is present; otherwise JSON health stub."""
    if INDEX_HTML.is_file():
        return FileResponse(INDEX_HTML, media_type="text/html")
    return {"service": "DealFlow AI", "status": "running", "agents": 6}


@app.post("/analyze")
async def submit_deal(
    company_name: str = Form(...),
    notes: str = Form(default=""),
    files: list[UploadFile] = File(default=[]),
):
    """
    Submit a company for M&A due diligence analysis.
    Accepts company name, optional notes, and up to 10 document files.
    """
    deal_id = str(uuid.uuid4())
    deal_dir = UPLOAD_DIR / deal_id
    deal_dir.mkdir(exist_ok=True)

    # Save uploaded files
    file_paths = []
    for file in files:
        if file.filename:
            safe_name = Path(file.filename).name
            dest = deal_dir / safe_name
            content = await file.read()
            dest.write_bytes(content)
            file_paths.append(str(dest.absolute()))
            logger.info(f"Saved {safe_name} for deal {deal_id}")

    # Record deal in local store
    deals[deal_id] = {
        "id": deal_id,
        "company_name": company_name,
        "status": "triggered",
        "file_paths": file_paths,
        "band_room_id": None,
        "created_at": datetime.utcnow().isoformat(),
        "memo_path": None,
    }

    # Trigger Orchestrator via Band
    try:
        room_id = await trigger_orchestrator(deal_id, company_name, file_paths, notes)
        deals[deal_id]["band_room_id"] = room_id
        deals[deal_id]["status"] = "in_progress"
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


@app.post("/deals/{deal_id}/complete")
async def mark_deal_complete(deal_id: str, memo_path: str):
    """
    Called by the Synthesis agent (via webhook or Band message) when the memo is ready.
    In production, the Synthesis agent posts a webhook to this endpoint.
    """
    if deal_id not in deals:
        raise HTTPException(status_code=404, detail="Deal not found")
    deals[deal_id]["status"] = "complete"
    deals[deal_id]["memo_path"] = memo_path
    return {"status": "updated"}
