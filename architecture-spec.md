# DealFlow AI — Architecture & Tech Stack Spec
### Band of Agents Hackathon | Track 1 (Enterprise) + Partner Prizes

---

## What It Is

**DealFlow AI** is an autonomous M&A due diligence platform where specialized agents collaborate through Band to analyze a target company and produce a comprehensive investment memo. A deal analyst submits a company name + documents, and 5 agents — each running a different framework and a different model — coordinate through a Band deal room to complete the analysis.

The key differentiator from a single-agent system: the findings from each agent flow into the others. Legal findings reshape the financial model. Financial anomalies trigger deeper legal research. The synthesis agent only fires when all specialists have completed and posted their structured outputs.

---

## Agent Roster

| Agent | Framework | Model (via) | Responsibility |
|---|---|---|---|
| **Orchestrator** | LangGraph | Claude Sonnet (AI/ML API) | Receives deal request, creates Band room, recruits agents, coordinates flow |
| **Document Parser** | Pydantic AI | Mistral-7B-Instruct (Featherless) | Extracts structured data from PDFs — contracts, financials, cap tables |
| **Financial Analyst** | LangGraph | Qwen2.5-72B-Instruct (Featherless) | Revenue modeling, burn rate, margin analysis, cap table math |
| **Legal Risk Agent** | CrewAI | Llama-3.1-70B (Featherless) | Flags liability clauses, IP ownership issues, change-of-control triggers |
| **Web Research Agent** | Anthropic Adapter | Claude Haiku (AI/ML API) | Live market research, competitor landscape, news sentiment |
| **Synthesis Agent** | LangGraph | Claude Sonnet (AI/ML API) | Waits for all signals, writes final investment memo |

**Why this lineup impresses the judges:**
- 3 different frameworks (LangGraph, Pydantic AI, CrewAI) coordinating through Band — the cross-framework pitch in action
- 3 different Featherless open-source models, each chosen for task fit
- Band is genuinely load-bearing: without it, cross-framework agent coordination is impossible

---

## System Architecture

```
User (deal analyst)
    │
    │  POST /analyze { company, documents[] }
    ▼
┌─────────────────────────────────┐
│         FastAPI Gateway         │  ← thin REST layer, handles uploads
└─────────────────────────────────┘
    │
    │  triggers
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR AGENT                           │
│                (LangGraph + Claude via AI/ML API)               │
│                                                                 │
│  1. Creates Band chat room: "Deal Room: {company}"              │
│  2. Calls thenvoi_lookup_peers → discovers available agents     │
│  3. Calls thenvoi_add_participant for each specialist           │
│  4. @mentions Document Parser with raw files                    │
│  5. Monitors room, coordinates handoffs via @mentions           │
└─────────────────────────────────────────────────────────────────┘
    │
    │  Band Chat Room (WebSocket, real-time)
    │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    │
    ├──▶ @DocumentParser: "Parse these files and post structured JSON"
    │        │
    │        ▼
    │   ┌──────────────────────────────────────────┐
    │   │  DOCUMENT PARSER AGENT                   │
    │   │  (Pydantic AI + Mistral-7B via Featherless) │
    │   │  Outputs: structured financials, clauses, │
    │   │           cap table, key dates            │
    │   │  Posts: @FinancialAnalyst @LegalRisk      │
    │   └──────────────────────────────────────────┘
    │
    ├──▶ @FinancialAnalyst: (receives parsed financials)
    │        │
    │        ▼
    │   ┌──────────────────────────────────────────┐
    │   │  FINANCIAL ANALYST AGENT                 │
    │   │  (LangGraph + Qwen2.5-72B via Featherless) │
    │   │  Outputs: revenue model, burn rate,      │
    │   │           valuation range, red flags      │
    │   │  Posts: @LegalRisk @Synthesis            │
    │   └──────────────────────────────────────────┘
    │
    ├──▶ @LegalRisk: (receives parsed clauses + financial flags)
    │        │
    │        ▼
    │   ┌──────────────────────────────────────────┐
    │   │  LEGAL RISK AGENT                        │
    │   │  (CrewAI + Llama-3.1-70B via Featherless) │
    │   │  Outputs: liability flags, IP risks,      │
    │   │           deal-breaker clauses            │
    │   │  Posts: @WebResearch @Synthesis           │
    │   └──────────────────────────────────────────┘
    │
    ├──▶ @WebResearch: (receives company name + legal/financial context)
    │        │
    │        ▼
    │   ┌──────────────────────────────────────────┐
    │   │  WEB RESEARCH AGENT                      │
    │   │  (Anthropic Adapter + Claude Haiku)       │
    │   │  Outputs: market size, competitors,       │
    │   │           news sentiment, founder history │
    │   │  Posts: @Synthesis                        │
    │   └──────────────────────────────────────────┘
    │
    └──▶ @Synthesis: (waits for all 4 agents, then fires)
             │
             ▼
        ┌──────────────────────────────────────────┐
        │  SYNTHESIS AGENT                         │
        │  (LangGraph + Claude Sonnet)              │
        │  Outputs: full investment memo as PDF     │
        │  Posts result to Band room + webhook      │
        └──────────────────────────────────────────┘
```

---

## Band Integration Details

### How Band is used (non-trivially)

1. **Dynamic room creation** — Orchestrator calls `thenvoi_create_chatroom` with the deal name. Every analysis gets its own isolated room with full audit trail.

2. **Agent discovery** — Orchestrator calls `thenvoi_lookup_peers` to find available specialist agents. If a specialist is busy/offline, orchestrator can recruit an alternate.

3. **Mention-based routing** — Each agent only sees messages it's @mentioned in. This is the core coordination primitive: Document Parser posts `@FinancialAnalyst @LegalRisk here's the structured data: {...}` and both agents receive it simultaneously.

4. **Structured event posting** — Agents post intermediate thoughts and tool call results via `thenvoi_send_event` (not messages). This keeps the main chat clean while full reasoning traces are logged.

5. **Human-in-the-loop escalation** — If Legal Risk agent finds a potential deal-breaker, it calls `thenvoi_add_participant` to add the human analyst to the room with a @mention, pausing the workflow until the human responds.

6. **Context rehydration** — If any agent crashes and restarts, it calls `/agent/context` to rebuild its conversation state and resume without data loss.

### agent_config.yaml structure
```yaml
orchestrator:
  agent_id: "<uuid>"
  api_key: "<key>"
document_parser:
  agent_id: "<uuid>"
  api_key: "<key>"
financial_analyst:
  agent_id: "<uuid>"
  api_key: "<key>"
legal_risk:
  agent_id: "<uuid>"
  api_key: "<key>"
web_research:
  agent_id: "<uuid>"
  api_key: "<key>"
synthesis:
  agent_id: "<uuid>"
  api_key: "<key>"
```

---

## Featherless Integration Details

All three Featherless models use the OpenAI-compatible API endpoint:

```python
from openai import OpenAI

featherless_client = OpenAI(
    base_url="https://api.featherless.ai/v1",
    api_key=os.environ["FEATHERLESS_API_KEY"]
)
```

### Model selection rationale (important for the demo/pitch)

| Model | Why this model for this task |
|---|---|
| `mistralai/Mistral-7B-Instruct-v0.3` | Fast, cheap, excellent at structured extraction. Perfect for parsing PDFs into JSON — doesn't need reasoning depth, needs precision. |
| `Qwen/Qwen2.5-72B-Instruct` | Best open-source model for quantitative reasoning. Outperforms Llama on math benchmarks. Ideal for financial modeling. |
| `meta-llama/Meta-Llama-3.1-70B-Instruct` | Strong instruction-following, good at legal document comprehension. Widely trusted for long-context legal tasks. |

This model selection story is your Featherless prize argument: you're not just using Featherless as a generic API — you're demonstrating that open-source model specialization is real and measurable.

---

## AI/ML API Integration

Used for the three Claude-powered agents (Orchestrator, Web Research, Synthesis):

```python
from openai import OpenAI

aiml_client = OpenAI(
    base_url="https://api.aimlapi.com/v1",
    api_key=os.environ["AIML_API_KEY"]
)
```

Models: `claude-sonnet-4-5` (Orchestrator + Synthesis), `claude-haiku-4-5-20251001` (Web Research — fast + cheap for search tasks)

---

## Tech Stack Summary

| Layer | Technology |
|---|---|
| **Agent framework** | Band SDK (`band-sdk`) with LangGraph, CrewAI, Pydantic AI, Anthropic adapters |
| **Open-source models** | Featherless AI (Mistral-7B, Qwen2.5-72B, Llama-3.1-70B) |
| **Frontier models** | AI/ML API (Claude Sonnet + Haiku) |
| **Coordination layer** | Band (chat rooms, @mentions, agent discovery, WebSocket events) |
| **Backend API** | FastAPI (Python) |
| **Frontend** | React + simple dashboard showing live Band room activity |
| **Document parsing** | pypdf + pandas (within Document Parser agent) |
| **Report generation** | ReportLab (PDF memo output) |
| **Environment** | Python 3.11+, uv package manager, `.env` for secrets |

---

## Project Structure

```
dealflow-ai/
├── agents/
│   ├── orchestrator/
│   │   └── agent.py          # LangGraph + Claude
│   ├── document_parser/
│   │   └── agent.py          # Pydantic AI + Mistral via Featherless
│   ├── financial_analyst/
│   │   └── agent.py          # LangGraph + Qwen via Featherless
│   ├── legal_risk/
│   │   └── agent.py          # CrewAI + Llama via Featherless
│   ├── web_research/
│   │   └── agent.py          # Anthropic Adapter + Claude Haiku
│   └── synthesis/
│       └── agent.py          # LangGraph + Claude Sonnet
├── api/
│   └── main.py               # FastAPI gateway
├── frontend/
│   └── dashboard/            # React live deal room view
├── shared/
│   ├── models.py             # Pydantic schemas for structured outputs
│   └── prompts.py            # System prompts for each agent
├── agent_config.yaml
├── .env
└── pyproject.toml
```

---

## Demo Script (for judges)

1. Open dashboard → submit "Analyze Acme Corp" + upload sample contracts + financials PDF
2. Watch Band room appear live: "Deal Room: Acme Corp"
3. Watch agents join the room one by one (Orchestrator recruits them)
4. Watch @mentions flow between agents in real-time
5. Legal agent flags a change-of-control clause → adds human to room → pauses workflow
6. Human clears it → workflow resumes
7. Synthesis agent fires → investment memo PDF downloads
8. Show the Band audit trail: every agent decision, every handoff, full traceability

Total runtime: ~90 seconds for a full deal analysis.

---

## Why This Wins

- **Band judging criterion (Technology):** 6 agents, 3 frameworks, genuine cross-agent coordination through Band — not a wrapper
- **Business Value:** Replaces $50K–$500K in advisor fees per deal. Clear enterprise SaaS model.
- **Originality:** Human-in-the-loop escalation mid-workflow + agent discovery are novel demos
- **Featherless prize:** 3 different specialized open-source models with explicit rationale for each
- **AI/ML API prize:** Claude powers the highest-stakes agents (orchestration + synthesis)
- **Recruitment signal:** Shows mastery of multi-framework agent orchestration, Band's API, and model selection judgment
