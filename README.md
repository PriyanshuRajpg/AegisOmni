# AegisOmni — Multimodal Equipment Safety Assistant (Prototype)

> **One-Sentence Problem Statement:** Maintenance technicians and trainees inspecting industrial pumps and valves often struggle to spot subtle visual wear or remember de-energized safety steps from a static photo, so AegisOmni turns an equipment image and a voice or text question into highlighted component findings, explicit uncertainty warnings, and a human-approved Lockout/Tagout (LOTO) checklist while blocking unsafe control commands.

## 1. Architecture Diagram

```text
[User Browser UI: Photo Upload + Voice/Text Question]
                        │
                        ▼
       [FastAPI Server: POST /api/inspect]
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
[Step 1: Input & MIME Check]   [Step 2: Safety Policy Gate]
 (Rejects empty queries 400     (Blocks valve/breaker actuation
  & non-image files 415)         & interlock bypasses with 403)
                        │
                        ▼ (If Safe & Valid)
       [Step 3: Meta Model API / Local Benchmark]
        • sam-3.1 (Component region highlighting)
        • muse-spark-1.3 (Visual Q&A + confidence check)
                        │
                        ▼
[Step 4: Evidence-Linked Report + Human LOTO Sign-off]
 (Abstains when visual confidence < 70%, e.g., steam occlusion)
```

## 2. How to Run (Laptop/PC OR Android Phone)

### Option A: Laptop / PC
```bash
# 1. Run the 12 automated safety & edge-case tests
python3 -m unittest discover -s tests -v

# 2. (Optional) Start the FastAPI backend server
export MODEL_API_KEY="your_meta_model_api_key"
uvicorn app:app --reload --port 8000
```

### Option B: Android Phone Only (Zero Local Install via GitHub Codespaces)
1. Upload this folder to a new GitHub repository from your mobile browser (`github.com/new`).
2. Tap **Code -> Codespaces -> Create codespace on main** (runs in a cloud Linux container right inside Chrome on Android).
3. In the Codespace terminal, run `python3 -m unittest discover -s tests -v` and `uvicorn app:app --host 0.0.0.0 --port 8000`.

## 3. Honest Limitations
- **Prototype Only:** Advisory decision-support tool, not a certified industrial safety system.
- **2D Visual Limits:** Cannot detect internal bearing fatigue, subsurface cracks, or live electrical voltage from an optical photo.
- **Abstention Policy:** Intentionally abstains when confidence is below 70% (such as heavy steam glare or poor lighting in Sample 3).
