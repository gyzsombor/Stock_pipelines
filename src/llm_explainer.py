from __future__ import annotations

import json
import os
from typing import Any

from modeling import build_analyst_memo


def llm_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def generate_analyst_memo_llm(symbol: str, latest_row, analyst: dict) -> dict[str, Any]:
    """
    Optional LLM explanation layer.
    Falls back cleanly if OpenAI is not configured.
    """
    fallback = {
        "memo": build_analyst_memo(symbol, latest_row, analyst),
        "risks": analyst.get("reasons", [])[:2],
        "catalysts": analyst.get("reasons", [])[2:4],
        "mode": "fallback",
    }

    if not llm_available():
        return fallback

    try:
        from openai import OpenAI

        client = OpenAI()

        payload = {
            "symbol": symbol,
            "recommendation": analyst.get("recommendation"),
            "confidence_score": round(float(analyst.get("confidence_score", 0.0)), 4),
            "confidence_band": analyst.get("confidence_band"),
            "model_agreement": analyst.get("model_agreement"),
            "risk_level": analyst.get("risk_level"),
            "news_support": analyst.get("news_support"),
            "regime": analyst.get("regime"),
            "reasons": analyst.get("reasons", []),
        }

        instructions = (
            "You are a professional equity research analyst. "
            "Write a concise analyst note in plain business English. "
            "Do not exaggerate. Do not make unsupported forecasts. "
            "Return strict JSON with keys: memo, risks, catalysts."
        )

        schema = {
            "type": "object",
            "properties": {
                "memo": {"type": "string"},
                "risks": {"type": "array", "items": {"type": "string"}},
                "catalysts": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["memo", "risks", "catalysts"],
            "additionalProperties": False,
        }

        response = client.responses.create(
            model="gpt-5.4",
            instructions=instructions,
            input=f"Create an analyst note from this structured payload:\n{json.dumps(payload, ensure_ascii=False)}",
            text={
                "format": {
                    "type": "json_schema",
                    "name": "analyst_note",
                    "schema": schema,
                    "strict": True,
                }
            },
            temperature=0.2,
        )

        parsed = json.loads(response.output_text)
        return {
            "memo": parsed["memo"],
            "risks": parsed.get("risks", []),
            "catalysts": parsed.get("catalysts", []),
            "mode": "llm",
        }

    except Exception:
        return fallback