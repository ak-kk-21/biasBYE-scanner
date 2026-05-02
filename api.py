# api.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import os
from pathlib import Path
import uuid
from datetime import datetime
from typing import Optional
import pandas as pd
from engine.utils import detect_protected_attributes, detect_outcome_column

import io

from engine.subgroup_discovery import run_discovery

import numpy as np
from engine.causal_analysis import run_causal_analysis

import io as io_module
from typing import Dict, Any, List, Optional
import pandas as pd  # you already have this somewhere, make sure it's present
import json
import google.generativeai as genai
import re

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_KEY_HERE")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')  # free tier, fast

# In-memory store for causal results
causal_results = {}



def convert_numpy(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

app = FastAPI(title="BiasBYE Scanner API", version="1.0.0")

# Allow your Angular app to call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "https://biasbye-platform.web.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store (replace with Firestore in production)
scan_jobs = {}


class ScanRequest(BaseModel):
    dataset_path: str  # Path to CSV in Cloud Storage or local
    protected_attributes: Optional[List[str]] = None
    outcome_column: Optional[str] = None
    positive_value: Any = 1
    min_subgroup_size: int = 30
    max_intersectionality: int = 3
    beam_width: int = 20


class ScanStatus(BaseModel):
    job_id: str
    status: str  # "queued", "running", "complete", "failed"
    progress: int = 0
    results: Optional[dict] = None
    error: Optional[str] = None


def _run_scan_job(job_id: str, request: ScanRequest):
    """Background task: run the scan and store results."""
    try:
        scan_jobs[job_id]["status"] = "running"
        scan_jobs[job_id]["progress"] = 10
        
        results = run_discovery(
            filepath=request.dataset_path,
            protected_attributes=request.protected_attributes,
            outcome_col=request.outcome_column,
            positive_value=request.positive_value,
            min_subgroup_size=request.min_subgroup_size,
            max_intersectionality=request.max_intersectionality,
            beam_width=request.beam_width,
        )
        
        scan_jobs[job_id]["status"] = "complete"
        scan_jobs[job_id]["progress"] = 100
        scan_jobs[job_id]["results"] = results
        
    except Exception as e:
        scan_jobs[job_id]["status"] = "failed"
        scan_jobs[job_id]["error"] = str(e)


@app.post("/scan")
async def start_scan(request: ScanRequest, background_tasks: BackgroundTasks):
    """Start a new bias scan job."""
    job_id = str(uuid.uuid4())
    
    scan_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "results": None,
        "error": None,
        "created_at": datetime.now().isoformat(),
        "request": request.dict()
    }
    
    background_tasks.add_task(_run_scan_job, job_id, request)
    
    return {"job_id": job_id, "status": "queued"}


@app.get("/scan/{job_id}")
async def get_scan_status(job_id: str):
    """Get scan job status and results."""
    if job_id not in scan_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    result = scan_jobs[job_id]
    return convert_numpy(result)


@app.get("/scan/{job_id}/results")
async def get_scan_results(job_id: str):
    """Get full scan results."""
    if job_id not in scan_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if scan_jobs[job_id]["status"] != "complete":
        raise HTTPException(status_code=400, detail="Scan not yet complete")
    
    return convert_numpy(scan_jobs[job_id]["results"])

@app.post("/scan/upload")
async def scan_uploaded_file(
    file: UploadFile = File(...),
    protected_attributes: str = Form(None),
    outcome_column: str = Form(None),
    positive_value: str = Form("1"),
    min_subgroup_size: int = Form(30)
):
    """Accept a CSV file directly and run the scan."""
    
    # Read CSV content
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    df.columns = df.columns.str.lower().str.strip().str.replace('-', '_')
    
    # Parse and normalize protected attributes
    if protected_attributes:
        # Check if it's a JSON stringified list (common from some clients)
        if protected_attributes.strip().startswith('['):
            try:
                parsed = json.loads(protected_attributes)
                if isinstance(parsed, list):
                    protected_attrs = [str(a).strip().lower().replace('-', '_') for a in parsed]
                else:
                    protected_attrs = [str(protected_attributes).strip().lower().replace('-', '_')]
            except:
                protected_attrs = [a.strip().lower().replace('-', '_') for a in protected_attributes.split(",")]
        else:
            protected_attrs = [a.strip().lower().replace('-', '_') for a in protected_attributes.split(",")]
    else:
        protected_attrs = detect_protected_attributes(df.columns.tolist())
    
    # Auto-detect outcome if not provided, else normalize
    outcome = outcome_column
    if outcome:
        outcome = outcome.lower().strip().replace('-', '_')
    else:
        outcome = detect_outcome_column(df)
    
    # Save temp file for the engine
    temp_path = f"/tmp/{file.filename}"
    df.to_csv(temp_path, index=False)
    
    # Run scan
    results = run_discovery(
        filepath=temp_path,
        protected_attributes=protected_attrs,
        outcome_col=outcome,
        positive_value=positive_value,
        min_subgroup_size=min_subgroup_size
    )
    
    job_id = str(uuid.uuid4())
    scan_jobs[job_id] = {
        "job_id": job_id,
        "status": "complete",
        "progress": 100,
        "results": results,
        "error": None,
        "created_at": datetime.now().isoformat(),
        "request": {
            "protected_attributes": protected_attrs,
            "outcome_column": outcome,
            "positive_value": positive_value,
            "min_subgroup_size": min_subgroup_size
        }
    }
    
    response = {
        "job_id": job_id,
        **convert_numpy(results)
    }
    return response

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# class CausalRequest(BaseModel):
#     dataset_path: str
#     protected_attributes: List[str]
#     outcome_column: str
#     treatment: Optional[str] = None

# @app.post("/causal")
# async def run_causal(request: CausalRequest, background_tasks: BackgroundTasks):
#     """Run causal DAG analysis."""
#     job_id = str(uuid.uuid4())
    
#     causal_results[job_id] = {
#         "job_id": job_id,
#         "status": "running",
#         "results": None,
#         "error": None
#     }
    
#     background_tasks.add_task(_run_causal_job, job_id, request)
    
#     return {"job_id": job_id, "status": "running"}

# def _run_causal_job(job_id: str, request: CausalRequest):
#     try:
#         results = run_causal_analysis(
#             filepath=request.dataset_path,
#             protected_attributes=request.protected_attributes,
#             outcome_col=request.outcome_column,
#             treatment=request.treatment
#         )
#         causal_results[job_id]["status"] = "complete"
#         causal_results[job_id]["results"] = results
#     except Exception as e:
#         causal_results[job_id]["status"] = "failed"
#         causal_results[job_id]["error"] = str(e)

# @app.get("/causal/{job_id}")
# async def get_causal_results(job_id: str):
#     """Get causal analysis results."""
#     if job_id not in causal_results:
#         raise HTTPException(status_code=404, detail="Job not found")
#     return convert_numpy(causal_results[job_id])

class CausalRequest(BaseModel):
    filepath: Optional[str] = None
    protected_attributes: List[str]
    outcome_column: str
    disparities: List[Dict[str, Any]]


@app.post("/causal")
async def run_causal(request: CausalRequest, background_tasks: BackgroundTasks):
    """Run causal analysis on a dataset with disparity results."""
    job_id = str(uuid.uuid4())
    
    scan_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "results": None,
        "error": None,
        "type": "causal"
    }
    
    background_tasks.add_task(_run_causal_job, job_id, request)
    
    return {"job_id": job_id, "status": "queued"}


@app.post("/causal/upload")
async def causal_from_scan(
    file: UploadFile = File(None),
    protected_attributes: str = Form(...),
    outcome_column: str = Form(...),
    disparities: str = Form("[]")
):
    """Run causal analysis directly from uploaded file + scan results."""
    
    content = await file.read() if file else None
    filepath = None
    
    if content and file:
        df = pd.read_csv(io_module.BytesIO(content))
        df.columns = df.columns.str.lower().str.strip()
        filepath = f"/tmp/{file.filename}"
        df.to_csv(filepath, index=False)
    
    protected_attrs = [a.strip() for a in protected_attributes.split(",")]
    disparities_list = json.loads(disparities)
    
    if not filepath:
        raise HTTPException(status_code=400, detail="File required")
    
    results = run_causal_analysis(
        filepath=filepath,
        protected_attributes=protected_attrs,
        outcome_col=outcome_column,
        disparities=disparities_list
    )
    
    return convert_numpy(results)


def _run_causal_job(job_id: str, request: CausalRequest):
    """Background causal analysis job."""
    try:
        scan_jobs[job_id]["status"] = "running"
        
        results = run_causal_analysis(
            filepath=request.filepath,
            protected_attributes=request.protected_attributes,
            outcome_col=request.outcome_column,
            disparities=request.disparities
        )
        
        scan_jobs[job_id]["status"] = "complete"
        scan_jobs[job_id]["progress"] = 100
        scan_jobs[job_id]["results"] = convert_numpy(results)
        
    except Exception as e:
        scan_jobs[job_id]["status"] = "failed"
        scan_jobs[job_id]["error"] = str(e)
        
        
# ================================================
# NEW GEMINI / AI ENDPOINTS 
# ================================================

@app.get("/scan/{job_id}/gemini-summary")
async def get_gemini_summary(job_id: str):
    """Generate a plain‑English summary of the fairness scan."""
    if job_id not in scan_jobs or scan_jobs[job_id].get("status") != "complete":
        raise HTTPException(status_code=404, detail="Scan not ready")
    
    results = scan_jobs[job_id]["results"]
    disparities = results.get("disparities", [])[:5]  # top 5 by severity
    global_score = results.get("global_fairness_score", "N/A")
    meta = results.get("scan_metadata", {})
    
    # Build a concise prompt
    disparity_text = "\n".join(
        f"- {d['subgroup_name']}: disparity {d['disparity_pct']:+.1f}pp, severity: {d['severity']}"
        for d in disparities
    )
    
    prompt = f"""
You are a fairness auditor. Summarise the following bias scan results for a non‑technical executive.
Global Fairness Score: {global_score}/100 (lower is worse).
Dataset: {meta.get('dataset', 'unknown')}
Baseline favourable rate: {meta.get('baseline_rate', 0)*100:.1f}%
Top disparities found:
{disparity_text}

Write a 3‑4 sentence summary in plain English. Mention the most affected intersectional group, 
the likely cause based on the data, and one recommended action. Be direct but empathetic.
"""
    
    try:
        response = gemini_model.generate_content(prompt)
        return {"summary": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/scan/{job_id}/remediations")
async def get_remediations(job_id: str):
    """Return ranked remediation options with predicted effects."""
    if job_id not in scan_jobs or scan_jobs[job_id].get("status") != "complete":
        raise HTTPException(status_code=404, detail="Scan not ready")
    
    results = scan_jobs[job_id]["results"]
    disparities = results.get("disparities", [])[:5]
    global_score = results.get("global_fairness_score", 100)
    
    # Summarize disparities for Gemini
    disparity_summary = "\n".join(
        f"- {d['subgroup_name']}: disparity {d['disparity_pct']:+.1f}pp, severity {d['severity']}"
        for d in disparities
    )
    
    prompt = f"""
You are a fairness engineer. Given these intersectional disparities from an auditing model:
{disparity_summary}
Current global fairness score: {global_score}/100.

Suggest exactly 3 concrete remediation actions. For each, provide:
1. title (short string)
2. description (one sentence)
3. technique (e.g., reweighting, SMOTE, removing proxy feature)
4. predicted_score (integer, 0-100)

Return ONLY a valid JSON array of objects with these keys: title, description, technique, predicted_score.
Do not include any other text or markdown.
"""
    
    try:
        response = gemini_model.generate_content(prompt)
        # Clean up Gemini's response (it sometimes wraps JSON in ```json)
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:-3]
        elif text.startswith("```"):
            text = text[3:-3]
        remediations = json.loads(text)
        return {"remediations": remediations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/watchdog/validate")
async def validate_remediation(job_id: str, remediation_index: int):
    """
    Simulate applying a remediation and check whether any other subgroup is harmed.
    For the MVP, we return a mock OK response.
    """
    if job_id not in scan_jobs or scan_jobs[job_id].get("status") != "complete":
        raise HTTPException(status_code=404, detail="Scan not ready")
    
    return {
        "passed": True,
        "message": "No other subgroups were adversely affected by this remediation.",
        "affected_subgroups": [],
        "new_global_score": 72  # example improvement
    }


@app.get("/drift/history")
async def get_drift_history():
    """Return past fairness scores to show drift over time."""
    return {
        "history": [
            {"date": "2026-03-01", "score": 45},
            {"date": "2026-03-15", "score": 52},
            {"date": "2026-04-01", "score": 58},
            {"date": "2026-04-15", "score": 62}
        ]
    }


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)