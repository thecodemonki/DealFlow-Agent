"""
Shared Pydantic models for structured data passed between agents via Band messages.
Each agent serializes its output as JSON matching one of these models,
posts it to the Band room, and the receiving agent deserializes it.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class DealStatus(str, Enum):
    PENDING = "pending"
    PARSING = "parsing"
    ANALYZING = "analyzing"
    REVIEWING = "reviewing"
    RESEARCHING = "researching"
    SYNTHESIZING = "synthesizing"
    ESCALATED = "escalated"
    COMPLETE = "complete"


class DealRequest(BaseModel):
    """Submitted by the FastAPI gateway to kick off a deal analysis."""
    company_name: str
    file_paths: list[str] = Field(default_factory=list)
    notes: Optional[str] = None


class ParsedDocuments(BaseModel):
    """Output of the Document Parser agent."""
    company_name: str
    financials: dict = Field(
        description="Revenue, COGS, expenses, burn rate, cash, headcount"
    )
    key_contracts: list[dict] = Field(
        description="List of {type, parties, key_clauses[], expiry, risk_notes}"
    )
    cap_table: list[dict] = Field(
        description="List of {investor, shares, percentage, investment_round}"
    )
    key_dates: list[dict] = Field(
        description="List of {event, date, significance}"
    )
    raw_text_summary: str


class FinancialAnalysis(BaseModel):
    """Output of the Financial Analyst agent."""
    revenue_cagr_pct: Optional[float] = None
    burn_rate_monthly_usd: Optional[float] = None
    runway_months: Optional[float] = None
    gross_margin_pct: Optional[float] = None
    arr_usd: Optional[float] = None
    valuation_range: dict = Field(
        description="{low_usd, mid_usd, high_usd, methodology}"
    )
    red_flags: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    summary: str


class LegalRiskAnalysis(BaseModel):
    """Output of the Legal Risk agent."""
    risk_level: str = Field(description="'low', 'medium', 'high', or 'deal_breaker'")
    ip_issues: list[str] = Field(default_factory=list)
    liability_flags: list[str] = Field(default_factory=list)
    change_of_control_clauses: list[str] = Field(default_factory=list)
    deal_breakers: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    requires_human_review: bool = False
    human_review_reason: Optional[str] = None
    summary: str


class MarketResearch(BaseModel):
    """Output of the Web Research agent."""
    market_size_usd: Optional[float] = None
    market_growth_rate_pct: Optional[float] = None
    top_competitors: list[dict] = Field(
        description="List of {name, description, estimated_funding_usd, threat_level}"
    )
    news_sentiment: str = Field(description="'positive', 'neutral', or 'negative'")
    recent_news: list[str] = Field(default_factory=list)
    founder_background: str = ""
    moat_assessment: str = ""
    summary: str


class InvestmentMemo(BaseModel):
    """Final output of the Synthesis agent."""
    company_name: str
    recommendation: str = Field(description="'invest', 'pass', or 'conditional'")
    confidence: str = Field(description="'high', 'medium', or 'low'")
    executive_summary: str
    financial_highlights: str
    legal_risks: str
    market_position: str
    red_flags: list[str] = Field(default_factory=list)
    deal_terms_suggested: str
    pdf_path: Optional[str] = None


class AgentSignal(BaseModel):
    """
    Wrapper agents use when posting results to the Band room.
    The `payload` field contains one of the above models serialized as a dict.
    """
    agent_name: str
    signal_type: str  # "parsed_documents", "financial_analysis", "legal_risk", "market_research", "investment_memo"
    payload: dict
    next_agents: list[str] = Field(
        description="Which agents should be @mentioned next"
    )
    status: str = "complete"
    message: Optional[str] = None
