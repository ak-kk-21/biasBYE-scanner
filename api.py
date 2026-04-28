# api.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
, UploadFile, File, Form
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
    positive_value: int = 1
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
    positive_value: int = Form(1),
    min_subgroup_size: int = Form(30)
):
    """Accept a CSV file directly and run the scan."""
    
    # Read CSV content
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    df.columns = df.columns.str.lower().str.strip()
    
    # Parse protected attributes
    if protected_attributes:
        protected_attrs = [a.strip() for a in protected_attributes.split(",")]
    else:
        protected_attrs = detect_protected_attributes(df.columns.tolist())
    
    # Auto-detect outcome if not provided
    outcome = outcome_column
    if not outcome:
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
    
    return convert_numpy(results)

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

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)