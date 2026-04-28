# api.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import os
from pathlib import Path
import uuid
from datetime import datetime

from engine.subgroup_discovery import run_discovery

import numpy as np

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



@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)