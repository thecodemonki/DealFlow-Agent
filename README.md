# DealFlow AI 🎯

> **Multi-agent M&A due diligence — powered by [Band](https://band.ai)**

DealFlow AI is a cross-framework multi-agent system that performs complete M&A due diligence in minutes. Six specialized AI agents collaborate through Band's shared environment — passing structured signals, coordinating tasks, and producing a final investment memo without any human in the loop.

**Live demo:** [dealflow-agent-production-4bf4.up.railway.app](https://dealflow-agent-production-4bf4.up.railway.app)

---

## How it works

Submit a company name and any available documents. Six agents immediately spring into action inside a shared Band room, routing work to each other, posting structured signals, and handing off outputs until the investment memo is generated.

```
User submits company → Orchestrator kicks off pipeline
    ├── WebResearch    → scrapes market data → posts SIGNAL:market_research
    ├── DocumentParser → extracts financials from uploaded files → posts SIGNAL:parsed_documents
    │       ├── FinancialAnalyst → CAGR, burn rate, valuation range → SIGNAL:financial_analysis
    │       └── LegalRisk        → CoC clauses, IP risks, liability → SIGNAL:legal_risk
    └── Synthesis → reads all signals → writes investment memo PDF → SIGNAL:investment_memo
```

All coordination happens **through Band** — agents @mention each other, post structured JSON signals, and the Band room acts as the shared memory and message bus for the entire workflow.

---

## The 6 agents

| Agent | Character | Framework | Model | Role |
|-------|-----------|-----------|-------|------|
| **Orchestrator** | 🎩 Conductor | OpenAI Agents SDK | GPT-4o | Kicks off the pipeline, routes tasks, escalates human review |
| **WebResearch** | 🔍 Detective | LangGraph | GPT-4o | Wikipedia + DuckDuckGo market research, posts to Band directly |
| **DocumentParser** | 📚 Librarian | OpenAI Agents SDK | GPT-4o-mini | Extracts financials, contracts, and cap table from uploaded PDFs |
| **FinancialAnalyst** | 💼 Banker | OpenAI Agents SDK | GPT-4o-mini | Calculates CAGR, burn rate, runway, and valuation range |
| **LegalRisk** | ⚖️ Judge | OpenAI Agents SDK | GPT-4o-mini | Flags change-of-control clauses, IP risks, and liability exposure |
| **Synthesis** | 🔮 Wizard | OpenAI Agents SDK | GPT-4o | Synthesizes all signals into a final PDF investment memo |

---

## Band integration

Band is the **central nervous system** of DealFlow AI — not just a notification layer.

- Every agent listens on a shared Band room via the `thenvoi` SDK
- Agents post **structured JSON signals** (e.g. `SIGNAL:financial_analysis`) that downstream agents parse
- @mentions route work: `@FinancialAnalyst` only activates when it receives a message with its handle
- The Orchestrator reads incoming Band messages to decide what to trigger next
- WebResearch posts its signal **directly via the Band REST API** (bypassing the LLM) for reliability
- The Band room acts as a persistent shared log — Synthesis reads the full message history to compile its memo
- If LegalRisk flags a deal-breaker, the Orchestrator posts an escalation message to loop in a human reviewer

---

## Tech stack

- **Agent frameworks:** OpenAI Agents SDK + LangGraph (cross-framework)
- **Models:** GPT-4o and GPT-4o-mini via [AI/ML API](https://aimlapi.com)
- **Coordination:** [Band (thenvoi)](https://band.ai) — shared room, @mentions, structured signals
- **API:** FastAPI (Python)
- **Frontend:** Vanilla JS + CSS animations
- **Deployment:** Railway (Docker)

---

## Running locally

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv)
- A Band account — get free Pro with code `BANDHACK26` at [band.ai/manage-billing](https://band.ai/manage-billing)
- AI/ML API key for GPT-4o access

### Setup

```bash
git clone https://github.com/<your-username>/dealflow-agent
cd dealflow-agent

uv sync

cp .env.example .env
# Add your AIML_API_KEY and ORCHESTRATOR_API_KEY

cp agent_config.yaml.example agent_config.yaml
# Add your Band agent IDs and API keys for all 6 agents
```

### agent_config.yaml

```yaml
orchestrator:
  agent_id: "..."
  api_key: "..."
web_research:
  agent_id: "..."
  api_key: "..."
document_parser:
  agent_id: "..."
  api_key: "..."
financial_analyst:
  agent_id: "..."
  api_key: "..."
legal_risk:
  agent_id: "..."
  api_key: "..."
synthesis:
  agent_id: "..."
  api_key: "..."
```

### Run

```bash
# Terminal 1 — start all 6 agents
python run_agents.py

# Terminal 2 — start the API + frontend
uvicorn api.main:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000) and submit a company.

---

## API

```bash
# Submit a company for analysis
curl -X POST http://localhost:8000/analyze \
  -F "company_name=Stripe" \
  -F "notes=Series B SaaS, considering acquisition" \
  -F "files=@financials.pdf"

# Check status
curl http://localhost:8000/deals/{deal_id}

# Download the investment memo PDF (most recently completed deal)
curl http://localhost:8000/memo/latest --output memo.pdf

# Deal Score + verdict JSON for the UI (same “latest” deal)
curl http://localhost:8000/memo/latest/summary

# When synthesis finishes, register the PDF and optional memo summary (JSON body)
curl -X POST "http://localhost:8000/deals/{deal_id}/complete" \
  -H "Content-Type: application/json" \
  -d '{"memo_path":"/abs/path/to/memo.pdf","memo_summary":{"deal_score":72,"risks_flagged_count":3,"company_name":"Acme","recommendation":"conditional","confidence":"medium"}}'
```

---

## Project structure

```
dealflow-agent/
├── agents/
│   ├── orchestrator/      # GPT-4o, OpenAI Agents SDK
│   ├── web_research/      # GPT-4o, LangGraph
│   ├── document_parser/   # GPT-4o-mini, OpenAI Agents SDK
│   ├── financial_analyst/ # GPT-4o-mini, OpenAI Agents SDK
│   ├── legal_risk/        # GPT-4o-mini, OpenAI Agents SDK
│   └── synthesis/         # GPT-4o, OpenAI Agents SDK
├── api/
│   └── main.py            # FastAPI gateway
├── shared/
│   ├── prompts.py         # System prompts for all agents
│   └── models.py          # Shared Pydantic models
├── frontend/
│   ├── index.html         # Single-page UI
│   └── characters/        # Agent character images
├── Dockerfile
└── railway.json
```

---

## Why multi-agent?

M&A due diligence is inherently parallel and multi-disciplinary. A single agent can't simultaneously hold deep financial modeling, legal contract review, market research, and narrative synthesis in context without degrading on all of them.

By splitting into specialists that communicate through Band:
- **WebResearch and DocumentParser run concurrently** — no waiting
- **Each agent's context window is focused** on one domain
- **Outputs are structured and auditable** — every signal is a JSON blob in the Band room log
- **Human escalation is built in** — LegalRisk can trigger a human review without breaking the pipeline

---

Built for the **[Band of Agents Hackathon](https://lablab.ai)** · June 2026
