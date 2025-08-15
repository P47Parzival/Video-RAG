import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"  # Change if backend runs elsewhere

st.title("Video Q&A RAG System")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov"])
if uploaded_file:
    try:
        # Send file to backend
        files = {"file": uploaded_file.getvalue()}
        res = requests.post(f"{BACKEND_URL}/upload", files={"file": uploaded_file})
        
        if res.status_code == 200:
            st.success("Video uploaded and processed!")
        else:
            st.error(f"Backend error: {res.text}")
    except Exception as e:
        st.error(f"File upload error: {e}")

    query = st.text_input("Ask a question about the video:")
    if st.button("Submit") and query:
        try:
            res = requests.get(f"{BACKEND_URL}/query", params={"q": query})
            if res.status_code == 200:
                st.write(res.json().get("response"))
            else:
                st.error(f"Backend error: {res.text}")
        except Exception as e:
            st.error(f"Query error: {e}")
