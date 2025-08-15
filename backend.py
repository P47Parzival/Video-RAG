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
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from functions import transcribe_video, index_video, get_rag_response
import os
import logging
import glob

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    filename = file.filename or "uploaded.mp4"
    safe_name = filename.replace(" ", "_")
    file_location = f"uploaded_{safe_name}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    try:
        transcript = transcribe_video(file_location)
        idx_dir = f"index_{os.path.splitext(safe_name)[0]}"
        index_video(transcript, persist_dir=idx_dir)
    except Exception as e:
        return {"status": "error", "error": str(e)}
    return {"status": "processed", "filename": file_location}

# ...existing code...
@app.get("/query")
def query_video(q: str):
    try:
        # pick latest uploaded_* video in cwd
        files = [p for p in glob.glob("uploaded_*") if p.lower().endswith((".mp4", ".mov", ".mkv"))]
        if not files:
            return {"error": "No uploaded video found. Please POST /upload first."}
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        video_path = files[0]
        response = get_rag_response(video_path, q)
        # ensure response is always a string
        if response is None:
            response = "No answer returned from server. Check backend logs."
        return {"response": response}
    except Exception as e:
        logging.exception("query_video error")
        return {"error": str(e)}