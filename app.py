"""FastAPI Backend Server for AegisOmni — Multimodal Equipment Safety Assistant.
Keeps MODEL_API_KEY strictly on the server and exposes:
  - GET  /health
  - GET  /api/status
  - POST /api/safety-check
  - POST /api/inspect
"""
from typing import Optional
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from src.safety_gate import evaluate_safety_gate, validate_image_upload
from src.inspector_client import inspect_equipment_with_meta_api, get_server_api_status

app = FastAPI(title="AegisOmni Multimodal Equipment Safety Assistant")


class SafetyRequest(BaseModel):
    query: str


class InspectRequest(BaseModel):
    filename: str = "sample_pump.png"
    mime_type: str = "image/png"
    size_bytes: int = 10240
    image_base64: Optional[str] = None
    query: str
    component_prompt: Optional[str] = None
    enable_segmentation: bool = False
    mode: str = "live"


@app.get("/")
def serve_ui():
    return FileResponse("aegis_omni_meta_hackathon.html")


@app.get("/health")
def health():
    return {"status": "ok", "service": "AegisOmni Safety Assistant"}


@app.get("/api/status")
def api_status():
    return get_server_api_status()


@app.post("/api/safety-check")
def safety_check(req: SafetyRequest):
    res = evaluate_safety_gate(req.query)
    return JSONResponse(status_code=res.http_status, content=res.to_dict())


@app.post("/api/inspect")
def inspect_endpoint(req: InspectRequest):
    img_check = validate_image_upload(req.filename, req.mime_type, req.size_bytes, req.image_base64)
    if not img_check["valid"]:
        return JSONResponse(status_code=img_check["http_status"], content=img_check)
    result = inspect_equipment_with_meta_api(
        user_question=req.query,
        image_base64=req.image_base64,
        mime_type=req.mime_type,
        component_prompt=req.component_prompt,
        enable_segmentation=req.enable_segmentation,
        mode=req.mode,
    )
    return JSONResponse(status_code=result.get("http_status", 200), content=result)
