"""Load agent credentials from env vars with agent_config.yaml fallback."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
AGENT_CONFIG_PATH = BASE_DIR / "agent_config.yaml"

_AGENT_ENV_KEYS: dict[str, tuple[str, str]] = {
    "orchestrator": ("ORCHESTRATOR_AGENT_ID", "ORCHESTRATOR_API_KEY"),
    "document_parser": ("DOCUMENT_PARSER_AGENT_ID", "DOCUMENT_PARSER_API_KEY"),
    "financial_analyst": ("FINANCIAL_ANALYST_AGENT_ID", "FINANCIAL_ANALYST_API_KEY"),
    "legal_risk": ("LEGAL_RISK_AGENT_ID", "LEGAL_RISK_API_KEY"),
    "web_research": ("WEB_RESEARCH_AGENT_ID", "WEB_RESEARCH_API_KEY"),
    "synthesis": ("SYNTHESIS_AGENT_ID", "SYNTHESIS_API_KEY"),
}


def load_agent_config(agent_key: str) -> tuple[str, str]:
    """Return (agent_id, api_key) from env vars, falling back to agent_config.yaml."""
    id_env, key_env = _AGENT_ENV_KEYS.get(agent_key, (None, None))
    agent_id = (os.environ.get(id_env) or "").strip() if id_env else ""
    api_key = (os.environ.get(key_env) or "").strip() if key_env else ""

    if agent_id and api_key:
        return agent_id, api_key

    if AGENT_CONFIG_PATH.is_file():
        try:
            import yaml

            with open(AGENT_CONFIG_PATH, encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            section = config.get(agent_key) or {}
            agent_id = agent_id or str(section.get("agent_id") or "").strip()
            api_key = api_key or str(section.get("api_key") or "").strip()
        except Exception as e:
            logger.warning("Could not load agent_config.yaml for %s: %s", agent_key, e)

    missing: list[str] = []
    if not agent_id:
        missing.append("agent_id")
    if not api_key:
        missing.append("api_key")
    if missing:
        env_hint = f"{id_env} and {key_env}" if id_env and key_env else "env vars"
        raise ValueError(
            f"Missing required fields for agent '{agent_key}': {', '.join(missing)}. "
            f"Set {env_hint} or add credentials to {AGENT_CONFIG_PATH}."
        )

    return agent_id, api_key
