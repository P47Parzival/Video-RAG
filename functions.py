import os
import logging
import subprocess
import shutil
import json
from typing import List, Tuple, Dict, Any
import time
import soundfile as sf
import librosa  # ✅ Added for audio duration check
import numpy as np

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
        _whisper_model = whisper.load_model("tiny")  # change size if you want faster/slower
    return _whisper_model

def transcribe_video(video_path: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Transcribe video to text using local whisper. Returns transcript file path and segments."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found. Install ffmpeg and add to PATH.")
    
    model = _get_whisper()
    audio_path = f"{os.path.splitext(video_path)[0]}.wav"

    # Capture FFmpeg output for better error handling
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path, "-ar", "16000", "-ac", "1", "-f", "wav", audio_path
            ],
            check=True,
            capture_output=True,
            text=True
        )
        logging.info("FFmpeg audio extraction successful for %s", video_path)
    except subprocess.CalledProcessError as e:
        logging.error("FFmpeg failed: %s", e.stderr)
        raise RuntimeError(f"FFmpeg failed to extract audio from {video_path}: {e.stderr}")

    # Check that audio file exists and has content
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        raise ValueError(f"Audio extraction failed, file is empty: {audio_path}")

    # Check audio duration and sampling rate
    try:
        duration = librosa.get_duration(path=audio_path)
        data, sr = sf.read(audio_path)
        if len(data) == 0:
            raise ValueError(f"Extracted audio is empty: {audio_path}")
        if duration < 0.1:
            raise ValueError(f"Audio duration too short ({duration:.2f}s) for {audio_path}")
        if sr != 16000:
            logging.warning("Audio sampling rate is %d Hz, expected 16000 Hz", sr)
        # ✅ Check for silent audio
        if np.max(np.abs(data)) < 1e-6:  # Threshold for near-silent audio
            logging.warning("Audio appears to be silent or nearly silent: %s", audio_path)
            raise ValueError(f"Audio {audio_path} is silent or has very low amplitude")
    except Exception as e:
        raise RuntimeError(f"Failed to read audio file {audio_path}: {e}")

    # Log audio properties for debugging
    logging.info("Audio file %s: duration=%.2fs, sample_rate=%d Hz, size=%d bytes, max_amplitude=%.6f",
                 audio_path, duration, sr, os.path.getsize(audio_path), np.max(np.abs(data)))

    # ✅ Try transcription with kv_cache disabled to avoid KeyError
    try:
        result = model.transcribe(audio_path, verbose=False, no_speech_threshold=0.6, suppress_tokens=[])  # ✅ Adjusted params
        if not result.get("text") and not result.get("segments"):
            raise ValueError("Whisper returned empty transcription")
    except Exception as e:
        logging.error("Whisper transcription failed: %s", str(e))
        # ✅ Fallback: Try transcription without key-value cache
        try:
            logging.info("Retrying transcription without kv_cache for %s", audio_path)
            result = model.transcribe(audio_path, verbose=False, no_speech_threshold=0.6, suppress_tokens=[], use_kv_caching=False)
            if not result.get("text") and not result.get("segments"):
                raise ValueError("Whisper returned empty transcription even without kv_cache")
        except Exception as e:
            raise RuntimeError(f"Whisper failed to transcribe {audio_path}: {e}")

    transcription = result.get("text", "").strip()
    segments = [
        {"start": float(s.get("start", 0.0)), 
         "end": float(s.get("end", 0.0)), 
         "text": s.get("text", "").strip()}
        for s in result.get("segments", [])
    ]

    # Save transcription only if non-empty
    if not transcription:
        logging.warning("No transcription produced for %s", audio_path)

    transcript_path = f"{os.path.splitext(video_path)[0]}_transcription.txt"
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcription if transcription else "No transcription available")

    seg_path = f"{os.path.splitext(video_path)[0]}_segments.json"
    with open(seg_path, "w", encoding="utf-8") as f:
        json.dump(segments, f)

    # Clean up audio file safely
    try:
        if os.path.exists(audio_path):
            os.remove(audio_path)
            logging.info("Removed temporary audio file %s", audio_path)
    except Exception as e:
        logging.warning("Could not remove temp audio %s: %s", audio_path, e)

    logging.info("transcribe_video: wrote transcript %s (%d chars, %d segments)", 
                 transcript_path, len(transcription), len(segments))

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

def _call_llm(prompt_text: str, max_retries: int = 3, backoff: float = 2.0) -> str:
    """Call Gemini via langchain wrapper and return plain string. Deterministic and retry on transient errors."""
    # set temperature=0 for deterministic outputs
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
    attempt = 0
    res = None
    while attempt < max_retries:
        try:
            # prefer invoke
            res = llm.invoke(prompt_text)
            break
        except Exception as e:
            attempt += 1
            logging.warning("LLM call attempt %d failed: %s", attempt, e)
            # if it's clearly transient/internal server error, retry
            if attempt < max_retries:
                time.sleep(backoff ** attempt)
            else:
                logging.exception("LLM permanently failed after retries")
                res = None
                break

    # prefer .content if present
    try:
        if hasattr(res, "content"):
            content = getattr(res, "content")
            if isinstance(content, str) and content.strip():
                return content
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