"""Deterministic Safety & Input Validation Gate for AegisOmni Equipment Safety Assistant.
Preserves core regex safety rules and adds strict image upload validation (MIME, size, base64).
"""
from dataclasses import dataclass, asdict
import base64
import re
from typing import Dict, Any, List, Optional


PROTOTYPE_DISCLAIMER = (
    "PROTOTYPE DECISION-SUPPORT TOOL: Not a certified industrial safety system. "
    "Cannot issue physical control commands. Requires human Lockout/Tagout (LOTO) verification."
)

ALLOWED_IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB


@dataclass
class SafetyEvaluation:
    allowed: bool
    action: str
    triggered_rule: str
    reason: str
    required_human_action: str
    http_status: int = 200

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


UNSAFE_PATTERNS = [
    (
        re.compile(r"\b(bypass|override|jumper|disable|short-circuit)\b.*\b(interlock|relay|relief|breaker|fuse|loto|lockout|guard)\b", re.I),
        "INTERLOCK_OR_RELAY_BYPASS_PROHIBITED",
        "Bypassing safety interlocks, pressure relief devices, or overload relays creates severe physical hazard.",
    ),
    (
        re.compile(r"\b(remotely open|force open|actuate|energize|start motor|close breaker|open valve)\b", re.I),
        "NO_PHYSICAL_ACTUATION_COMMANDS",
        "AegisOmni is a read-only advisory assistant and is prohibited from issuing physical actuation or control commands.",
    ),
    (
        re.compile(r"\b(while energized|while live|480v live|without loto|skip lockout)\b", re.I),
        "LIVE_ELECTRICAL_WORK_PROHIBITED",
        "Instructions for live electrical work without Lockout/Tagout (LOTO) violate electrical safety standards.",
    ),
]


def validate_image_upload(
    filename: str,
    mime_type: str,
    size_bytes: int,
    image_base64: Optional[str] = None,
) -> Dict[str, Any]:
    if not filename or size_bytes <= 0:
        return {"valid": False, "http_status": 400, "error": "EMPTY_IMAGE_FILE: Image file is empty or missing."}
    if size_bytes > MAX_IMAGE_BYTES:
        return {
            "valid": False,
            "http_status": 413,
            "error": f"IMAGE_TOO_LARGE: {size_bytes} bytes exceeds 10 MB limit.",
        }
    if (mime_type or "").lower() not in ALLOWED_IMAGE_MIMES:
        return {
            "valid": False,
            "http_status": 415,
            "error": f"UNSUPPORTED_MEDIA_TYPE: '{mime_type}'. Allowed formats: JPEG, PNG, WebP.",
        }
    if image_base64 is not None:
        raw_b64 = image_base64.split(",", 1)[-1].strip()
        if not raw_b64:
            return {"valid": False, "http_status": 400, "error": "EMPTY_IMAGE_DATA: Base64 payload is empty."}
        try:
            decoded = base64.b64decode(raw_b64, validate=True)
            if len(decoded) < 8:
                return {"valid": False, "http_status": 400, "error": "CORRUPTED_IMAGE_DATA: Payload too short."}
        except Exception:
            return {"valid": False, "http_status": 400, "error": "INVALID_BASE64_IMAGE: Could not decode image bytes."}
    return {"valid": True, "http_status": 200, "error": None}


def evaluate_safety_gate(query_text: Optional[str]) -> SafetyEvaluation:
    cleaned = (query_text or "").strip()
    if not cleaned:
        return SafetyEvaluation(
            allowed=False,
            action="REJECT_EMPTY_QUERY",
            triggered_rule="EMPTY_OR_WHITESPACE_QUERY",
            reason="Question input cannot be empty.",
            required_human_action="Provide a specific equipment inspection question.",
            http_status=400,
        )

    for pattern, rule_name, reason in UNSAFE_PATTERNS:
        if pattern.search(cleaned):
            return SafetyEvaluation(
                allowed=False,
                action="BLOCK_EXECUTION",
                triggered_rule=rule_name,
                reason=reason,
                required_human_action="Physical Lockout/Tagout (LOTO) verification and licensed supervisor sign-off required.",
                http_status=403,
            )
    return SafetyEvaluation(
        allowed=True,
        action="ALLOW_ADVISORY_INSPECTION",
        triggered_rule="NONE",
        reason="Query requests read-only visual inspection or checklist generation with no physical actuation or interlock bypass.",
        required_human_action="Technician must still verify physical LOTO before performing maintenance.",
        http_status=200,
    )


def build_inspection_report(
    sample_title: str,
    query_text: str,
    regions: Optional[List[Dict[str, Any]]],
    checklist: Optional[List[str]],
    min_confidence_threshold: float = 0.70,
    mode: str = "demo",
) -> Dict[str, Any]:
    gate = evaluate_safety_gate(query_text)
    if not gate.allowed:
        return {
            "status": "REJECTED_BY_SAFETY_GATE" if gate.http_status == 403 else "VALIDATION_ERROR",
            "mode": mode,
            "http_status": gate.http_status,
            "disclaimer": PROTOTYPE_DISCLAIMER,
            "safety_evaluation": gate.to_dict(),
            "findings": [],
            "checklist": [],
        }

    if regions is None or len(regions) == 0:
        return {
            "status": "EMPTY_MODEL_RESPONSE_ERROR",
            "mode": mode,
            "http_status": 502,
            "disclaimer": PROTOTYPE_DISCLAIMER,
            "safety_evaluation": gate.to_dict(),
            "abstained_due_to_low_confidence": True,
            "requires_human_loto_signoff": True,
            "findings": [],
            "checklist": ["Model returned an empty response. Perform manual LOTO visual inspection."],
        }

    low_conf_regions = [r for r in regions if float(r.get("confidence", 0.0)) < min_confidence_threshold]
    abstained = len(low_conf_regions) > 0

    for r in regions:
        for coord in ("x", "y", "w", "h"):
            val = float(r.get(coord, 0.0))
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"Bounding box coordinate {coord}={val} out of normalized [0, 1] range")

    return {
        "status": "INSUFFICIENT_VISUAL_EVIDENCE" if abstained else "ADVISORY_REPORT_READY",
        "mode": mode,
        "http_status": 200,
        "disclaimer": PROTOTYPE_DISCLAIMER,
        "safety_evaluation": gate.to_dict(),
        "abstained_due_to_low_confidence": abstained,
        "requires_human_loto_signoff": True,
        "sample_title": sample_title,
        "findings": regions,
        "checklist": checklist or [],
    }
