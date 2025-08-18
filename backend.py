# from fastapi import FastAPI, UploadFile, File
# from functions import get_rag_response, process_video_with_videodb
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Or specify your frontend URL
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.post("/upload")
# async def upload_video(file: UploadFile = File(...)):
#     file_location = "uploaded.mp4"
#     with open(file_location, "wb") as f:
#         f.write(await file.read())
#      # Ensure video is processed/indexed now so queries return context
#     try:
#         process_video_with_videodb(file_location)
#     except Exception as e:
#         return {"status": "error", "error": str(e)}
#     return {"status": "processed"}

# @app.get("/query")
# def query_video(q: str):
#     try:
#         response = get_rag_response("uploaded.mp4", q)
#         return {"response": response}
#     except Exception as e:
#         return {"error": str(e)}
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from functions import transcribe_video, index_video, get_rag_response
import os
import logging
import glob
import uuid
from typing import Dict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store task status in memory
tasks: Dict[str, Dict] = {}

def _process_in_background(file_location: str, idx_dir: str, task_id: str):
    """Background worker: transcribe + index for an uploaded file."""
    try:
        transcript, segments = transcribe_video(file_location)
        index_video(transcript, persist_dir=idx_dir, segments=segments)
        # Write a marker file to indicate processing done
        os.makedirs(idx_dir, exist_ok=True)
        done_flag = f"{idx_dir}/.processed"
        with open(done_flag, "w") as f:
            f.write("done")
        tasks[task_id] = {"status": "completed", "result": {"transcript": transcript, "segments": segments}, "error": None}
    except Exception as e:
        logging.exception("Background processing failed for %s", file_location)
        tasks[task_id] = {"status": "failed", "result": None, "error": str(e)}

@app.post("/upload")
async def upload_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    filename = file.filename or "uploaded.mp4"
    safe_name = filename.replace(" ", "_")
    file_location = f"uploads/uploaded_{safe_name}"
    os.makedirs("uploads", exist_ok=True)
    
    # Save uploaded file
    with open(file_location, "wb") as f:
        f.write(await file.read())

    idx_dir = f"index_{os.path.splitext(safe_name)[0]}"
    # Initialize task status
    tasks[task_id] = {"status": "pending", "result": None, "error": None}
    
    # Schedule background processing
    try:
        background_tasks.add_task(_process_in_background, file_location, idx_dir, task_id)
    except Exception as e:
        logging.exception("Failed to schedule background task")
        tasks[task_id] = {"status": "failed", "result": None, "error": str(e)}
        return {"task_id": task_id, "status": "error", "error": str(e)}
    
    return {"task_id": task_id, "status": "processing", "filename": file_location}

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Check the status of a transcription task."""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.get("/status")
async def upload_status(filename: str = None):
    """Check if index for the latest (or given) uploaded file is processed."""
    try:
        if filename:
            idx_dir = f"index_{os.path.splitext(filename.replace(' ', '_'))[0]}"
        else:
            # Check latest uploaded_*
            files = [p for p in glob.glob("uploads/uploaded_*") if p.lower().endswith((".mp4", ".mov", ".mkv"))]
            if not files:
                return {"status": "no_uploads"}
            files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            latest = files[0]
            idx_dir = f"index_{os.path.splitext(os.path.basename(latest))[0].replace('uploaded_','')}"
        done_flag = os.path.join(idx_dir, ".processed")
        if os.path.exists(done_flag):
            return {"status": "processed", "index": idx_dir}
        return {"status": "processing"}
    except Exception as e:
        logging.exception("Status check failed")
        return {"status": "error", "error": str(e)}

@app.get("/query")
async def query_video(q: str):
    """Query the latest uploaded video."""
    try:
        files = [p for p in glob.glob("uploads/uploaded_*") if p.lower().endswith((".mp4", ".mov", ".mkv"))]
        if not files:
            return {"error": "No uploaded video found. Please POST /upload first."}
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        video_path = files[0]
        result = get_rag_response(video_path, q)
        answer = result.get("answer")
        frames = result.get("frames", [])
        return {"response": answer, "frames": frames}
    except Exception as e:
        logging.exception("query_video error")
        return {"error": str(e)}