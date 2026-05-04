"""
Shared decision engine logic for consistency between live and backtest.
"""

import numpy as np
import pandas as pd

# Default strategy configuration
DEFAULT_MIN_CONFIDENCE_THRESHOLD = 0.50
DEFAULT_MAX_LONG_POSITION_SIZE = 0.40
DEFAULT_MAX_SHORT_POSITION_SIZE = 0.25
DEFAULT_RISK_AVERSION = 1.0
DEFAULT_DEAD_ZONE = 0.05


def apply_calibration(raw_confidence: float, calibration_model=None) -> float:
    """Safely apply calibration to raw confidence score."""
    if calibration_model is None:
        return raw_confidence
    try:
        # For IsotonicRegression, use predict
        if hasattr(calibration_model, 'predict'):
            calibrated = calibration_model.predict([raw_confidence])[0]
            return float(calibrated)
        else:
            # Fallback to raw if not supported
            return raw_confidence
    except Exception:
        return raw_confidence


def calibrate_confidence(confidence_score: float, calibration_model) -> float:
    """Calibrate raw confidence score using the provided model."""
    return apply_calibration(confidence_score, calibration_model)


def compute_position_size(
    ensemble_prob: float,
    calibrated_confidence: float,
    shorting_allowed: bool = False,
    model_agreement: float = 0.0,
) -> float:
    """Compute position size with selective filters."""
    # Moderate conviction filter
    if calibrated_confidence < 0.55:
        return 0.0

    # Agreement filter
    if model_agreement < 0.50:
        return 0.0

    # Probability edge filter: avoid weak near-neutral signals
    edge = abs(ensemble_prob - 0.5)
    if edge < 0.05:
        return 0.0

    # Decision zone: only act on directional signals
    if ensemble_prob > 0.55:
        direction = 1.0
    elif ensemble_prob < 0.45:
        direction = -1.0 if shorting_allowed else 0.0
    else:
        return 0.0

    # Conviction scaling
    conviction = (calibrated_confidence - 0.55) / 0.45
    conviction = max(0.0, min(conviction, 1.0))

    position_size = conviction * direction

    # Minimum trade size filter
    if abs(position_size) < 0.03:
        return 0.0

    # Apply caps
    if direction > 0:
        position_size = min(position_size, 0.4)
    else:
        position_size = max(position_size, -0.25)

    return float(position_size)


def get_decision_label(recommendation_score: float, confidence_score: float, shorting_allowed: bool = False) -> str:
    """Get a clear label for the decision."""
    signed_score = (recommendation_score - 0.5) * 2.0
    if signed_score > DEFAULT_DEAD_ZONE and confidence_score >= DEFAULT_MIN_CONFIDENCE_THRESHOLD:
        return "Buy"
    elif signed_score < -DEFAULT_DEAD_ZONE and shorting_allowed and confidence_score >= DEFAULT_MIN_CONFIDENCE_THRESHOLD:
        return "Short / Sell"
    elif signed_score < -DEFAULT_DEAD_ZONE:
        return "Avoid / No Buy"
    else:
        return "Hold / Wait"


def get_decision_explanation(analyst_data: dict, features: dict, shorting_allowed: bool = False) -> dict:
    """Generate explanation for the decision."""
    decision = analyst_data.get("recommendation", "Hold / Wait")
    score = analyst_data.get("recommendation_score", 0.5)
    confidence = analyst_data.get("confidence_score", 0.5)
    signed_score = (score - 0.5) * 2.0

    explanation = {
        "decision": decision,
        "confidence_level": "High" if confidence > 0.7 else "Medium" if confidence > 0.4 else "Low",
        "position_size_reason": f"Position size {analyst_data.get('position_size', 0):.2f} based on calibrated confidence {confidence:.2f} and risk adjustments",
        "top_drivers": []
    }

    if abs(signed_score) > DEFAULT_DEAD_ZONE:
        if signed_score > 0:
            explanation["top_drivers"].append("Model ensemble favors upward movement")
        else:
            if shorting_allowed:
                explanation["top_drivers"].append("Model ensemble favors downward movement")
            else:
                explanation["top_drivers"].append("Model ensemble suggests avoiding long positions")
    else:
        explanation["top_drivers"].append("Model signals are mixed or weak")

    if confidence >= DEFAULT_MIN_CONFIDENCE_THRESHOLD:
        explanation["top_drivers"].append("High confidence in the signal")
    else:
        explanation["top_drivers"].append("Low confidence - signal may be noisy")

    if len(explanation["top_drivers"]) == 0:
        explanation["top_drivers"].append("Insufficient data for strong signal")

    return explanation