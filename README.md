# DealFlow AI

**Autonomous M&A due diligence platform — 6 agents collaborating through Band.**

6 specialized agents across 3 frameworks (LangGraph, CrewAI, Pydantic AI) coordinate through Band to analyze acquisition targets, passing structured context at each handoff. A deal that takes analysts weeks completes in ~90 seconds.

---

## Agent Architecture

| Agent | Framework | Model | Role |
|---|---|---|---|
| Orchestrator | LangGraph | Claude Sonnet (AI/ML API) | Creates Band room, recruits agents, coordinates flow |
| Document Parser | Pydantic AI | Mistral-7B (Featherless) | Extracts structured data from contracts and financials |
| Financial Analyst | LangGraph | Qwen2.5-72B (Featherless) | Revenue modeling, valuation, burn rate analysis |
| Legal Risk | CrewAI | Llama-3.1-70B (Featherless) | Contract risks, IP issues, change-of-control clauses |
| Web Research | LangGraph | Claude Haiku (AI/ML API) | Market sizing, competitors, news, founder background |
| Synthesis | LangGraph | Claude Sonnet (AI/ML API) | Aggregates all findings → investment memo PDF |

---

## Setup

### 1. Install dependencies

```bash
uv install
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in FEATHERLESS_API_KEY and AIML_API_KEY
```

### 3. Register agents in Band

Go to [band.ai/agents](https://app.band.ai/agents) and create 6 External Agents:
- Orchestrator
- DocumentParser
- FinancialAnalyst
- LegalRisk
- WebResearch
- Synthesis

```bash
cp agent_config.yaml.example agent_config.yaml
# Fill in agent_id and api_key for each agent
```

### 4. Run all agents

```bash
uv run python run_agents.py
```

### 5. Start the API server (separate terminal)

```bash
uv run uvicorn api.main:app --reload --port 8000
```

---

## Usage

### Submit a deal via API

```bash
curl -X POST http://localhost:8000/analyze \
  -F "company_name=Acme Corp" \
  -F "notes=Series B SaaS company, considering acquisition" \
  -F "files=@contracts/msa.pdf" \
  -F "files=@financials/p_and_l.pdf"
```

### Check deal status

```bash
curl http://localhost:8000/deals/{deal_id}
```

### Download investment memo PDF

```bash
curl http://localhost:8000/deals/{deal_id}/memo --output memo.pdf
```

---

## How Band Coordination Works

1. **Room creation** — Orchestrator calls `thenvoi_create_chatroom` → "Deal: Acme Corp [abc12345]"
2. **Agent recruitment** — Orchestrator calls `thenvoi_lookup_peers` + `thenvoi_add_participant` for each specialist
3. **Parallel dispatch** — Document Parser posts `SIGNAL:parsed_documents` + `@FinancialAnalyst @LegalRisk` simultaneously
4. **Context passing** — Each agent serializes its findings as structured JSON in Band messages
5. **Human escalation** — If Legal Risk flags a deal-breaker, Orchestrator calls `thenvoi_add_participant` to add the human analyst
6. **Completion** — Synthesis posts `SIGNAL:investment_memo` + generates PDF

---

## Tech Stack

- **Band SDK** (`band-sdk`) — agent coordination layer
- **Featherless AI** — serverless open-source model inference (Mistral, Qwen, Llama)
- **AI/ML API** — Claude Sonnet + Haiku access
- **FastAPI** — REST gateway for deal submissions
- **ReportLab** — PDF memo generation
- **pypdf** — document text extraction

---

## Project Structure

```
dealflow-ai/
├── agents/
│   ├── orchestrator/agent.py      # LangGraph + Claude Sonnet
│   ├── document_parser/agent.py   # Pydantic AI + Mistral-7B
│   ├── financial_analyst/agent.py # LangGraph + Qwen2.5-72B
│   ├── legal_risk/agent.py        # CrewAI + Llama-3.1-70B
│   ├── web_research/agent.py      # LangGraph + Claude Haiku
│   └── synthesis/agent.py         # LangGraph + Claude Sonnet
├── shared/
│   ├── models.py                  # Pydantic schemas for agent outputs
│   └── prompts.py                 # System prompts for each agent
├── api/main.py                    # FastAPI gateway
├── run_agents.py                  # Start all 6 agents
├── pyproject.toml
└── agent_config.yaml.example
```
