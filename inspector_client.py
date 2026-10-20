"""Server-Side Meta Model API Client for AegisOmni Equipment Safety Assistant.
Requirements enforced:
  1. Keeps MODEL_API_KEY strictly on the server (read from os.environ).
  2. Never pretends a live model call succeeded when it failed or when MODEL_API_KEY is missing.
  3. Uses a real base64 image data URI with Meta Model API (https://api.meta.ai/v1/chat/completions or /v1/responses).
  4. Treats component segmentation (sam-3.1) as optional so core vision Q&A works independently.
"""
import json
import os
from typing import Optional
import requests
from src.safety_gate import evaluate_safety_gate, PROTOTYPE_DISCLAIMER

META_API_BASE = os.environ.get("META_API_BASE", "https://api.meta.ai/v1")
VISION_MODEL = os.environ.get("META_VISION_MODEL", "muse-spark-1.3")
SAM_MODEL = os.environ.get("META_SAM_MODEL", "sam-3.1")


def get_server_api_status() -> dict:
    api_key = os.environ.get("MODEL_API_KEY", "").strip()
    return {
        "live_api_configured": bool(api_key),
        "vision_model": VISION_MODEL,
        "optional_segmentation_model": SAM_MODEL,
        "key_location": "server_environment_only",
    }


def inspect_equipment_with_meta_api(
    user_question: str,
    image_base64: Optional[str] = None,
    mime_type: str = "image/png",
    component_prompt: Optional[str] = None,
    enable_segmentation: bool = False,
    mode: str = "live",
    session=None,
) -> dict:
    # Step 1: Deterministic safety gate BEFORE any model call
    gate = evaluate_safety_gate(user_question)
    if not gate.allowed:
        return {
            "status": "REJECTED_BY_SAFETY_GATE" if gate.http_status == 403 else "VALIDATION_ERROR",
            "mode": mode,
            "http_status": gate.http_status,
            "disclaimer": PROTOTYPE_DISCLAIMER,
            "safety_evaluation": gate.to_dict(),
        }

    # Step 2: If explicitly running in labeled Demo Mode, never claim a live API call happened
    if mode == "demo":
        return {
            "status": "DEMO_MODE_MOCK_RESPONSE",
            "mode": "demo",
            "http_status": 200,
            "disclaimer": PROTOTYPE_DISCLAIMER,
            "message": "Explicitly labeled Demo Mode response (no live Meta Model API call made).",
            "safety_evaluation": gate.to_dict(),
        }

    # Step 3: Live mode requires MODEL_API_KEY on the server — fail honestly (503) if missing
    api_key = os.environ.get("MODEL_API_KEY", "").strip()
    if not api_key:
        return {
            "status": "LIVE_API_KEY_MISSING_ON_SERVER",
            "mode": "live",
            "http_status": 503,
            "error": (
                "Live Meta Model API call failed: MODEL_API_KEY is not set in the server environment. "
                "Refusing to fabricate live inspection results. Set MODEL_API_KEY on the server or switch to Demo Mode."
            ),
            "disclaimer": PROTOTYPE_DISCLAIMER,
            "safety_evaluation": gate.to_dict(),
        }

    if not image_base64:
        return {
            "status": "MISSING_IMAGE_PAYLOAD",
            "mode": "live",
            "http_status": 400,
            "error": "Live inspection requires a base64-encoded equipment image.",
            "disclaimer": PROTOTYPE_DISCLAIMER,
            "safety_evaluation": gate.to_dict(),
        }

    data_uri = image_base64 if image_base64.startswith("data:") else f"data:{mime_type};base64,{image_base64}"
    http = session or requests
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    system_instruction = (
        "You are AegisOmni, a read-only industrial equipment inspection prototype. "
        "Never issue physical control commands. Return strict JSON with keys: "
        "'visible_observations' (list of objects with component, observation, confidence 0-1), "
        "'uncertainty' (string explaining lighting/angle/2D limitations), and "
        "'next_steps' (list of de-energized LOTO inspection steps)."
    )

    try:
        qa_resp = http.post(
            f"{META_API_BASE}/chat/completions",
            headers=headers,
            json={
                "model": VISION_MODEL,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_question},
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    },
                ],
                "response_format": {"type": "json_object"},
            },
            timeout=30,
        )
        qa_resp.raise_for_status()
        qa_json = qa_resp.json()
        if not qa_json or "choices" not in qa_json or not qa_json["choices"]:
            return {
                "status": "EMPTY_UPSTREAM_RESPONSE",
                "mode": "live",
                "http_status": 502,
                "error": "Meta Model API returned an empty or malformed response payload.",
                "disclaimer": PROTOTYPE_DISCLAIMER,
                "safety_evaluation": gate.to_dict(),
            }

        raw_content = qa_json["choices"][0].get("message", {}).get("content", "")
        parsed_report = json.loads(raw_content) if isinstance(raw_content, str) and raw_content.strip().startswith("{") else {"raw_text": raw_content}

        # Optional Step 4: Component segmentation with sam-3.1 if requested
        segmentation_result = {"enabled": False, "status": "SKIPPED"}
        if enable_segmentation and component_prompt:
            try:
                sam_resp = http.post(
                    f"{META_API_BASE}/responses",
                    headers=headers,
                    json={
                        "model": SAM_MODEL,
                        "input": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_image", "image_url": data_uri},
                                    {"type": "input_text", "text": component_prompt},
                                ],
                            }
                        ],
                    },
                    timeout=25,
                )
                sam_resp.raise_for_status()
                segmentation_result = {"enabled": True, "status": "SUCCESS", "data": sam_resp.json()}
            except Exception as sam_exc:
                segmentation_result = {
                    "enabled": True,
                    "status": "SEGMENTATION_FAILED_NON_FATAL",
                    "error": f"Optional sam-3.1 call failed: {sam_exc}",
                }

        return {
            "status": "LIVE_API_SUCCESS",
            "mode": "live",
            "http_status": 200,
            "model_used": VISION_MODEL,
            "disclaimer": PROTOTYPE_DISCLAIMER,
            "safety_evaluation": gate.to_dict(),
            "inspection_report": parsed_report,
            "segmentation": segmentation_result,
        }
    except Exception as exc:
        return {
            "status": "UPSTREAM_API_ERROR",
            "mode": "live",
            "http_status": 502,
            "error": f"Meta Model API call failed: {exc}",
            "disclaimer": PROTOTYPE_DISCLAIMER,
            "safety_evaluation": gate.to_dict(),
        }
