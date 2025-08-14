import subprocess
from videodb import connect, SceneExtractionType
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from google.cloud import speech
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

def transcribe_video(video_path):
    client = speech.SpeechClient()
    # Convert video to audio (wav)
    audio_path = f"{os.path.splitext(video_path)[0]}.wav"
    subprocess.run([
        "ffmpeg", "-i", video_path, "-ar", "16000", "-ac", "1", "-f", "wav", audio_path
    ], check=True)

    with open(audio_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US"
    )

    response = client.recognize(config=config, audio=audio)
    transcription = " ".join([result.alternatives[0].transcript for result in response.results])
    transcript_path = f"{os.path.splitext(video_path)[0]}_transcription.txt"
    with open(transcript_path, "w") as f:
        f.write(transcription)
    return transcript_path

def extract_frames(video_path, output_dir="frames"):
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-i", video_path, "-vf", "fps=1", f"{output_dir}/frame_%04d.png"
    ], check=True)

def process_video_with_videodb(video_path):
    api_key = os.getenv("VIDEO_DB_API_KEY")
    if not api_key:
        raise ValueError("VIDEO_DB_API_KEY not set in environment.")
    conn = connect(api_key=api_key)
    video = conn.upload(url=video_path)
    video.index_spoken_words()
    video.index_scenes(
        extraction_type=SceneExtractionType.shot_based,
        prompt="Describe scenes in detail"
    )
    return True

def index_video(transcript_path):
    with open(transcript_path, "r") as f:
        text = f.read()
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=20)
    chunks = splitter.split_text(text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_texts(chunks, embeddings)
    return vectorstore

def get_rag_response(video_path, query):
    transcript_path = transcribe_video(video_path)
    vectorstore = index_video(transcript_path)
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = PromptTemplate(
        input_variables=["query", "context"],
        template="Answer based on video context: {context}\nQuestion: {query}\nAnswer:"
    )
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    response = llm.invoke(prompt.format(query=query, context=context))
    return response