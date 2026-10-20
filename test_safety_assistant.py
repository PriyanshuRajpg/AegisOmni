import base64
import json
import os
import unittest
from fastapi.testclient import TestClient
from app import app
from src.safety_gate import (
    evaluate_safety_gate,
    validate_image_upload,
    build_inspection_report,
    PROTOTYPE_DISCLAIMER,
)
from src.inspector_client import inspect_equipment_with_meta_api


VALID_PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n0123456789").decode("ascii")


class DummyTimeoutSession:
    def post(self, *args, **kwargs):
        raise TimeoutError("Simulated upstream Meta Model API timeout")


class DummySuccessSession:
    def post(self, url, *args, **kwargs):
        class Resp:
            def raise_for_status(self):
                pass

            def json(self):
                if "/chat/completions" in url:
                    return {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "visible_observations": [
                                                {
                                                    "component": "Lower Flange Bolt #3",
                                                    "observation": "Surface rust visible on threads",
                                                    "confidence": 0.91,
                                                }
                                            ],
                                            "uncertainty": "Single 2D image cannot verify internal gasket torque.",
                                            "next_steps": ["Apply LOTO before inspecting flange bolt."],
                                        }
                                    )
                                }
                            }
                        ]
                    }
                return {"masks": [{"label": "flange bolt", "iou": 0.92}]}

        return Resp()


class TestSafetyAssistant(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_01_reject_valve_actuation_command(self):
        res = evaluate_safety_gate("Remotely open valve V-102 to clear the line.")
        self.assertFalse(res.allowed)
        self.assertEqual(res.http_status, 403)
        self.assertEqual(res.triggered_rule, "NO_PHYSICAL_ACTUATION_COMMANDS")

    def test_02_reject_interlock_bypass_command(self):
        res = evaluate_safety_gate("Bypass the pressure relief interlock and run the pump.")
        self.assertFalse(res.allowed)
        self.assertEqual(res.triggered_rule, "INTERLOCK_OR_RELAY_BYPASS_PROHIBITED")

    def test_03_reject_live_electrical_jumper(self):
        res = evaluate_safety_gate("How do I jumper the relay while 480V live power is on?")
        self.assertFalse(res.allowed)

    def test_04_approve_safe_visual_inspection_query(self):
        res = evaluate_safety_gate("What visible corrosion is on the lower flange bolt and what is the LOTO checklist?")
        self.assertTrue(res.allowed)
        self.assertEqual(res.http_status, 200)

    def test_05_low_confidence_triggers_uncertainty_warning(self):
        report = build_inspection_report(
            sample_title="Sample 3: Steam-Occluded Gate Valve",
            query_text="Is the packing gland cracked?",
            regions=[{"name": "Valve Stem", "confidence": 0.41, "x": 0.3, "y": 0.3, "w": 0.2, "h": 0.2}],
            checklist=["Wait for steam to clear and retake photo."],
        )
        self.assertEqual(report["status"], "INSUFFICIENT_VISUAL_EVIDENCE")
        self.assertTrue(report["abstained_due_to_low_confidence"])

    def test_06_checklist_requires_human_loto_signoff(self):
        report = build_inspection_report(
            sample_title="Sample 1: Pump",
            query_text="Inspect lower flange bolt.",
            regions=[{"name": "Bolt #3", "confidence": 0.91, "x": 0.2, "y": 0.5, "w": 0.1, "h": 0.1}],
            checklist=["Apply LOTO before touching casing."],
        )
        self.assertTrue(report["requires_human_loto_signoff"])

    def test_07_bounding_box_normalization_bounds(self):
        with self.assertRaises(ValueError):
            build_inspection_report(
                sample_title="Invalid Box",
                query_text="Inspect bolt.",
                regions=[{"name": "Bolt", "confidence": 0.95, "x": 1.4, "y": 0.2, "w": 0.1, "h": 0.1}],
                checklist=[],
            )

    def test_08_report_includes_prototype_disclaimer(self):
        report = build_inspection_report(
            sample_title="Sample 1",
            query_text="Inspect sight glass.",
            regions=[{"name": "Sight Glass", "confidence": 0.94, "x": 0.5, "y": 0.4, "w": 0.1, "h": 0.1}],
            checklist=["Check oil level."],
        )
        self.assertEqual(report["disclaimer"], PROTOTYPE_DISCLAIMER)

    def test_09_reject_empty_or_whitespace_query(self):
        res = evaluate_safety_gate("   ")
        self.assertFalse(res.allowed)
        self.assertEqual(res.http_status, 400)
        self.assertEqual(res.triggered_rule, "EMPTY_OR_WHITESPACE_QUERY")

    def test_10_reject_unsupported_image_mime_type(self):
        chk = validate_image_upload("firmware.bin", "application/octet-stream", 4096)
        self.assertFalse(chk["valid"])
        self.assertEqual(chk["http_status"], 415)

    def test_11_handle_empty_model_response_gracefully(self):
        report = build_inspection_report(
            sample_title="Sample 1",
            query_text="Inspect lower flange bolt.",
            regions=[],
            checklist=[],
        )
        self.assertEqual(report["status"], "EMPTY_MODEL_RESPONSE_ERROR")
        self.assertEqual(report["http_status"], 502)

    def test_12_handle_upstream_api_timeout_error(self):
        old_key = os.environ.get("MODEL_API_KEY")
        os.environ["MODEL_API_KEY"] = "test_dummy_key"
        try:
            res = inspect_equipment_with_meta_api(
                user_question="Inspect pump bolt corrosion.",
                image_base64=VALID_PNG_B64,
                session=DummyTimeoutSession(),
            )
            self.assertEqual(res["status"], "UPSTREAM_API_ERROR")
            self.assertEqual(res["http_status"], 502)
        finally:
            if old_key is None:
                os.environ.pop("MODEL_API_KEY", None)
            else:
                os.environ["MODEL_API_KEY"] = old_key

    def test_13_live_mode_fails_honestly_when_server_api_key_missing(self):
        old_key = os.environ.get("MODEL_API_KEY")
        os.environ.pop("MODEL_API_KEY", None)
        try:
            resp = self.client.post(
                "/api/inspect",
                json={
                    "filename": "pump.png",
                    "mime_type": "image/png",
                    "size_bytes": 1024,
                    "image_base64": VALID_PNG_B64,
                    "query": "Inspect lower flange bolt.",
                    "mode": "live",
                },
            )
            self.assertEqual(resp.status_code, 503)
            self.assertEqual(resp.json()["status"], "LIVE_API_KEY_MISSING_ON_SERVER")
        finally:
            if old_key is not None:
                os.environ["MODEL_API_KEY"] = old_key

    def test_14_live_mode_parses_structured_vision_and_optional_sam31(self):
        old_key = os.environ.get("MODEL_API_KEY")
        os.environ["MODEL_API_KEY"] = "test_dummy_key"
        try:
            res = inspect_equipment_with_meta_api(
                user_question="Inspect lower flange bolt.",
                image_base64=VALID_PNG_B64,
                component_prompt="flange bolt",
                enable_segmentation=True,
                mode="live",
                session=DummySuccessSession(),
            )
            self.assertEqual(res["status"], "LIVE_API_SUCCESS")
            self.assertEqual(res["http_status"], 200)
            self.assertEqual(res["segmentation"]["status"], "SUCCESS")
            self.assertIn("visible_observations", res["inspection_report"])
        finally:
            if old_key is None:
                os.environ.pop("MODEL_API_KEY", None)
            else:
                os.environ["MODEL_API_KEY"] = old_key

    def test_15_fastapi_status_never_exposes_secret_key(self):
        old_key = os.environ.get("MODEL_API_KEY")
        os.environ["MODEL_API_KEY"] = "super_secret_meta_key_123"
        try:
            resp = self.client.get("/api/status")
            self.assertEqual(resp.status_code, 200)
            body_text = resp.text
            self.assertNotIn("super_secret_meta_key_123", body_text)
            self.assertTrue(resp.json()["live_api_configured"])
        finally:
            if old_key is None:
                os.environ.pop("MODEL_API_KEY", None)
            else:
                os.environ["MODEL_API_KEY"] = old_key


if __name__ == "__main__":
    unittest.main()
