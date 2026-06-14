"""
Standalone script to generate the Stripe investment memo PDF directly.
Run from ~/DealFlow-Agent/: uv run python generate_memo.py
"""
import sys
sys.path.insert(0, ".")

from agents.synthesis.agent import generate_pdf_memo

memo_data = {
    "company_name": "Stripe",
    "recommendation": "CONDITIONAL",
    "confidence": "medium",
    "executive_summary": (
        "Stripe, Inc. is an Irish-American multinational financial services and SaaS company "
        "dual-headquartered in South San Francisco, CA and Dublin, Ireland. Founded in 2010 by "
        "Patrick and John Collison, Stripe provides payment-processing software and APIs for "
        "e-commerce. The company was last valued at approximately $50–65B (2023) after a down "
        "round from its 2021 peak of $95B. This web-only analysis finds Stripe to be a market "
        "leader with strong growth trajectory, subject to regulatory and competitive risks."
    ),
    "financial_highlights": (
        "Estimated valuation range: $50B–$95B (ARR multiple methodology). "
        "Revenue CAGR (5-year): ~58% based on publicly reported growth figures. "
        "Cash runway: estimated 100+ months given reported profitability in 2023. "
        "Stripe processed $817B in total payment volume in 2022. "
        "Note: financials are estimates based on public data; no company documents were provided."
    ),
    "legal_risks": (
        "Change-of-control clauses present in standard commercial agreements — review required for CoC triggers. "
        "IP ownership: proprietary payment technology and APIs carry standard IP ownership risks; "
        "open-source license exposure should be assessed. "
        "Liability exposure: transaction disputes, fraud losses, and regulatory penalties (PCI-DSS, GDPR, PSD2) "
        "represent material risk factors. "
        "Deal-breaker terms: non-compete clauses and exclusivity agreements in key merchant contracts "
        "may restrict post-acquisition operations. "
        "Requires human legal review before proceeding."
    ),
    "market_position": (
        "Stripe operates in the global financial services and fintech market. "
        "Primary competitors: PayPal/Braintree, Adyen, Square (Block), Worldpay, and Checkout.com. "
        "Stripe holds strong developer mindshare and is the default payment infrastructure for "
        "startups and mid-market SaaS companies. Key moat: API-first developer experience, "
        "extensive product suite (Stripe Atlas, Radar, Billing, Connect), and global coverage "
        "across 46+ countries. TAM: global digital payments market estimated at $8–10T by 2026."
    ),
    "deal_terms_suggested": (
        "Recommended deal structure: minority growth investment or strategic partnership. "
        "Suggested terms: standard anti-dilution provisions, information rights, pro-rata rights. "
        "Valuation anchor: $50–65B range given 2023 down round precedent. "
        "Contingencies: full legal review of change-of-control clauses, IP audit, "
        "and validated financials from data room before closing."
    ),
    "red_flags": [
        "Valuation declined ~45% from 2021 peak ($95B → ~$50B) — monitor further compression",
        "No company documents provided; all financials are public estimates only",
        "Regulatory risk: heavy exposure to evolving global payments regulation (PSD2, CFPB)",
        "Competitive pressure from Adyen and PayPal in enterprise segment",
    ],
}

path = generate_pdf_memo.invoke({"memo_data": memo_data})
print(f"✅ PDF saved to: {path}")
