from fastapi import FastAPI, UploadFile, File
from functions import get_rag_response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    file_location = "uploaded.mp4"
    with open(file_location, "wb") as f:
        f.write(await file.read())
    # Optionally process and index here
    return {"status": "processed"}

@app.get("/query")
def query_video(q: str):
    try:
        response = get_rag_response("uploaded.mp4", q)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}