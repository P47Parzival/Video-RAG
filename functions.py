import os
import logging
import subprocess
from typing import List

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
    """Transcribe video to text using local whisper. Returns transcript file path."""
    model = _get_whisper()
    audio_path = f"{os.path.splitext(video_path)[0]}.wav"
    # convert to wav (16k mono) for stable transcription
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path, "-ar", "16000", "-ac", "1", "-f", "wav", audio_path
    ], check=True)
    result = model.transcribe(audio_path)
    transcription = result.get("text", "").strip()
    transcript_path = f"{os.path.splitext(video_path)[0]}_transcription.txt"
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcription)
    logging.info("transcribe_video: wrote transcript %s (%d chars)", transcript_path, len(transcription))
    return transcript_path

def index_video(transcript_path: str, persist_dir: str = None):
    """Create a Chroma vectorstore from transcript text using HuggingFace embeddings."""
    with open(transcript_path, "r", encoding="utf-8") as f:
        text = f.read() or ""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks: List[str] = splitter.split_text(text) if text and text.strip() else []
    logging.info("index_video: transcript chars=%d, chunks=%d", len(text), len(chunks))

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    if persist_dir:
        os.makedirs(persist_dir, exist_ok=True)
        vect = Chroma.from_texts(chunks or [""], embeddings, persist_directory=persist_dir)
        try:
            vect.persist()
        except Exception:
            pass
    else:
        vect = Chroma.from_texts(chunks or [""], embeddings)
    return vect

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

def get_rag_response(video_path: str, query: str, k: int = 3) -> str:
    """Return a string answer for query about the uploaded video."""
    transcript_path = f"{os.path.splitext(video_path)[0]}_transcription.txt"
    if not os.path.exists(transcript_path):
        transcript_path = transcribe_video(video_path)

    vectorstore = index_video(transcript_path)
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
    answer = _call_llm(prompt_text)

    if not answer or answer.strip().lower() in ("none", "no response from llm", ""):
        debug = f"transcript_exists={os.path.exists(transcript_path)}, docs={len(docs)}, context_len={len(context)}"
        logging.warning("get_rag_response: no useful answer -> %s", debug)
        return f"No answer found. {debug}"
    return answer