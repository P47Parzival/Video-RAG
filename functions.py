import os
import logging
import subprocess
import shutil
import json
from typing import List, Tuple, Dict, Any

import whisper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

logging.basicConfig(level=logging.INFO)

# Load whisper model once
_whisper_model = None
def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model("small")  # change size if you want faster/slower
    return _whisper_model

def transcribe_video(video_path: str) -> str:
    """Transcribe video to text using local whisper. Returns transcript file path.
    Each segment dict includes 'start','end','text'
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found. Install ffmpeg and add to PATH.")
    
    model = _get_whisper()
    audio_path = f"{os.path.splitext(video_path)[0]}.wav"
    # convert to wav (16k mono) for stable transcription
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path, "-ar", "16000", "-ac", "1", "-f", "wav", audio_path], 
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    result = model.transcribe(audio_path)
    transcription = result.get("text", "").strip()
    segments = []
    # whisper returns segments list with start/end/text
    for s in result.get("segments", []) :
        segments.append({"start": float(s.get("start",0.0)), "end": float(s.get("end",0.0)), "text": s.get("text","").strip()})

    transcript_path = f"{os.path.splitext(video_path)[0]}_transcription.txt"
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcription)
    
    # save segments metadata for debugging / reuse
    seg_path = f"{os.path.splitext(video_path)[0]}_segments.json"
    with open(seg_path, "w", encoding="utf-8") as f:
        json.dump(segments, f)

    try:
        if os.path.exists(audio_path):
            os.remove(audio_path)
    except Exception:
        logging.warning("Could not remove temp audio %s", audio_path)
    
    logging.info("transcribe_video: wrote transcript %s (%d chars)", transcript_path, len(transcription), len(segments))
    return transcript_path, segments

def index_video(transcript_path: str, persist_dir: str = None, segments: List[Dict[str,Any]] = None):
    """Create a Chroma vectorstore. If segments provided, index per-segment (with timestamps)."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if segments and len(segments) > 0:
        texts = [s.get("text","") or "" for s in segments]
        metadatas = [{"start": s.get("start",0.0), "end": s.get("end",0.0)} for s in segments]
        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            vect = Chroma.from_texts(texts, embeddings, metadatas=metadatas, persist_directory=persist_dir)
            try:
                vect.persist()
            except Exception:
                pass
        else:
            vect = Chroma.from_texts(texts, embeddings, metadatas=metadatas)
        logging.info("index_video: indexed %d segments", len(texts))
        return vect

    # fallback: split transcript file into chunks
    with open(transcript_path, "r", encoding="utf-8") as f:
        text = f.read() or ""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks: List[str] = splitter.split_text(text) if text and text.strip() else []
    if persist_dir:
        os.makedirs(persist_dir, exist_ok=True)
        vect = Chroma.from_texts(chunks or [""], embeddings, persist_directory=persist_dir)
        try:
            vect.persist()
        except Exception:
            pass
    else:
        vect = Chroma.from_texts(chunks or [""], embeddings)
    logging.info("index_video: transcript chars=%d, chunks=%d", len(text), len(chunks))
    return vect

def extract_frames_at_timestamps(video_path: str, timestamps: List[float], out_dir: str, max_frames: int = 3) -> List[str]:
    """Extract single frames at given timestamps (seconds). Returns list of image paths."""
    os.makedirs(out_dir, exist_ok=True)
    out_paths = []
    for i, t in enumerate(timestamps[:max_frames]):
        out_path = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{i+1}.jpg")
        # use -ss before -i for fast seek (accurate enough for this purpose)
        cmd = ["ffmpeg", "-y", "-ss", str(t), "-i", video_path, "-frames:v", "1", "-q:v", "2", out_path]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            out_paths.append(out_path)
        except subprocess.CalledProcessError:
            logging.exception("Failed to extract frame at %s", t)
    return out_paths

def _call_llm(prompt_text: str) -> str:
    """Call Gemini via langchain wrapper and return plain string."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    # try common invocation styles
    res = None
    try:
        res = llm.invoke(prompt_text)
    except Exception as e_invoke:
        logging.debug("llm.invoke failed: %s", e_invoke)
        try:
            from langchain.schema import HumanMessage
            res = llm.generate([[HumanMessage(content=prompt_text)]])
        except Exception as e_gen:
            logging.exception("llm.generate fallback failed: %s", e_gen)
            res = None
    
    try:
        if hasattr(res, "content"):
            content = getattr(res, "content")
            if isinstance(content, str) and content.strip():
                return content
            # sometimes .content may be a dict with 'content' or 'text'
            if isinstance(content, dict):
                for key in ("content", "text"):
                    if key in content and isinstance(content[key], str):
                        return content[key]
    except Exception:
        logging.debug("extracting res.content failed", exc_info=True)

    if isinstance(res, str):
        return res
    if res is None:
        logging.warning("_call_llm: LLM returned None")
        return "No response from LLM"
    if hasattr(res, "generations"):
        try:
            return res.generations[0][0].text
        except Exception:
            pass
    if isinstance(res, dict):
        for key in ("output", "candidates", "choices", "content", "text"):
            if key in res:
                val = res[key]
                if isinstance(val, list) and val:
                    first = val[0]
                    if isinstance(first, dict):
                        for sub in ("content", "text", "message"):
                            if sub in first:
                                return first[sub]
                    elif isinstance(first, str):
                        return first
                elif isinstance(val, str):
                    return val
    try:
        return str(res)
    except Exception:
        return "Unrecognized LLM response"

def get_rag_response(video_path: str, query: str, k: int = 3) -> Dict[str, Any]:
    """Return dict {answer: str, frames: [paths]} for the query."""
    transcript_path = f"{os.path.splitext(video_path)[0]}_transcription.txt"
    segments = None
    if not os.path.exists(transcript_path):
        transcript_path, segments = transcribe_video(video_path)
    else:
        seg_path = f"{os.path.splitext(video_path)[0]}_segments.json"
        if os.path.exists(seg_path):
            try:
                with open(seg_path, "r", encoding="utf-8") as f:
                    segments = json.load(f)
            except Exception:
                segments = None

    vectorstore = index_video(transcript_path, segments=segments)
    docs = []
    try:
        if hasattr(vectorstore, "similarity_search"):
            docs = vectorstore.similarity_search(query, k=k)
    except Exception as e:
        logging.exception("similarity_search failed: %s", e)

    context = "\n".join([d.page_content for d in docs]) if docs else ""
    logging.info("get_rag_response: docs=%d context_len=%d", len(docs), len(context))

    prompt = PromptTemplate(
        input_variables=["query", "context"],
        template="You are given context extracted from a video:\n\n{context}\n\nQuestion: {query}\nAnswer concisely:"
    )
    prompt_text = prompt.format(query=query, context=context)

    # Call LLM with fallback; if LLM fails, we'll still return frames + top context
    answer = None
    try:
        answer = _call_llm(prompt_text)
    except Exception:
        logging.exception("LLM call failed")

    # find timestamps for top-matching docs
    timestamps = []
    for d in docs[:3]:
        meta = getattr(d, "metadata", {}) or {}
        start = meta.get("start")
        end = meta.get("end")
        if start is not None and end is not None:
            # choose center of segment
            timestamps.append((start + end) / 2.0)
    # if no timestamps available but segments exist, find first segment containing top text
    if not timestamps and segments:
        for seg in segments[:3]:
            if seg.get("text","").strip():
                timestamps.append((seg.get("start",0)+seg.get("end",0))/2.0)
    # final fallback: 3 equally spaced timestamps near start
    if not timestamps:
        timestamps = [1.0, 2.0, 3.0]

    # make small offsets to collect 3 frames (center +/- 1s)
    center = timestamps[0]
    frame_times = [max(0, center-1.0), center, center+1.0]

    frames_dir = "extracted_frames"
    frames = extract_frames_at_timestamps(video_path, frame_times, frames_dir, max_frames=3)

    if not answer or answer.strip().lower() in ("none", "no response from llm", ""):
        debug = f"transcript_exists={os.path.exists(transcript_path)}, docs={len(docs)}, context_len={len(context)}"
        logging.warning("get_rag_response: no useful answer -> %s", debug)
        # fallback: return top context and frames
        fallback = f"(LLM unavailable or no answer) Top context:\n\n{context}" if context else f"No answer found. {debug}"
        return {"answer": fallback, "frames": frames}
    return {"answer": answer, "frames": frames}
