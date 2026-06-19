"""
Shared Pydantic models for structured data passed between agents via Band messages.
Each agent serializes its output as JSON matching one of these models,
posts it to the Band room, and the receiving agent deserializes it.
"""

from __future__ import annotations
from pydantic import BaseModel, Field, model_validator
from typing import Optional
from enum import Enum


def verdict_from_deal_score(score: int) -> str:
    """UI bands: 70–100 PASS, 40–69 CONDITIONAL, 0–39 FAIL."""
    try:
        s = int(score)
    except (TypeError, ValueError):
        s = 50
    s = max(0, min(100, s))
    if s >= 70:
        return "PASS"
    if s >= 40:
        return "CONDITIONAL"
    return "FAIL"


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
    """Output of the Document Parser agent (Librarian)."""
    company_name: str = Field(default="Unknown")
    revenue_ttm: Optional[str] = Field(default=None, description="Trailing twelve months revenue as stated in docs")
    burn_rate_monthly: Optional[str] = Field(default=None, description="Monthly burn / cash consumption if stated")
    runway_months: Optional[int] = Field(default=None, description="Cash runway in months if derivable")
    valuation_ask: Optional[str] = Field(default=None, description="Ask or implied valuation from deck")
    total_raised: Optional[str] = Field(default=None, description="Funding raised to date if stated")
    key_metrics: list[str] = Field(
        default_factory=list,
        description="Short bullet metrics extracted verbatim or paraphrased from the file",
    )
    raw_text_excerpt: str = Field(
        default="",
        max_length=500,
        description="Up to 500 chars of representative text from the source document",
    )
    financials: dict = Field(
        default_factory=dict,
        description="Revenue, COGS, expenses, burn rate, cash, headcount (structured)",
    )
    key_contracts: list[dict] = Field(
        default_factory=list,
        description="List of {type, parties, key_clauses[], expiry, risk_notes}",
    )
    cap_table: list[dict] = Field(
        default_factory=list,
        description="List of {investor, shares, percentage, investment_round}",
    )
    key_dates: list[dict] = Field(
        default_factory=list,
        description="List of {event, date, significance}",
    )
    raw_text_summary: str = Field(
        default="",
        description="Legacy longer summary; prefer raw_text_excerpt for new signals",
    )

    @model_validator(mode="after")
    def _clip_raw_text_excerpt(self):
        ex = (self.raw_text_excerpt or "")[:500]
        object.__setattr__(self, "raw_text_excerpt", ex)
        return self


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
    risk_flag_count: int = Field(
        default=0,
        ge=0,
        description="Total discrete legal risks flagged across ip_issues, liability_flags, CoC clauses, and deal_breakers",
    )
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
    deal_score: int = Field(default=50, ge=0, le=100, description="0–100 holistic deal quality score")
    deal_verdict: str = Field(
        default="CONDITIONAL",
        description="PASS, CONDITIONAL, or FAIL — must match deal_score bands (normalized on validate)",
    )
    overall_score: float = Field(description="Overall investment score 1-10")
    legal_score: float = Field(description="Legal risk score 1-10")
    financial_score: float = Field(description="Financial health score 1-10")
    market_score: float = Field(description="Market position score 1-10")
    regulatory_score: float = Field(description="Regulatory exposure score 1-10")
    team_score: float = Field(description="Team and IP quality score 1-10")
    risks_flagged_count: int = Field(
        default=0,
        ge=0,
        description="Count from Judge legal_risk.risk_flag_count when available",
    )
    executive_summary: str
    financial_highlights: str
    legal_risks: str
    market_position: str
    red_flags: list[str] = Field(default_factory=list)
    deal_terms_suggested: str
    pdf_path: Optional[str] = None

    @model_validator(mode="after")
    def align_deal_verdict_to_score(self):
        object.__setattr__(self, "deal_verdict", verdict_from_deal_score(self.deal_score))
        return self


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
