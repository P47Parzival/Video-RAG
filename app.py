import streamlit as st
import requests, os

BACKEND_URL = "http://localhost:8000"  # Change if backend runs elsewhere

st.title("Video Q&A RAG System")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov"])
if uploaded_file:
    try:
        # send to backend as multipart file correctly
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        res = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=120)
        if res.status_code == 200:
            st.success("Video uploaded and processed!")
        else:
            st.error(f"Backend error: {res.text}")
    except Exception as e:
        st.error(f"File upload error: {e}")

    query = st.text_input("Ask a question about the video:")
    if st.button("Submit") and query:
        try:
            res = requests.get(f"{BACKEND_URL}/query", params={"q": query}, timeout=60)
            if res.status_code == 200:
                data = res.json()
                st.write(data.get("response"))
                frames = data.get("frames", [])
                # show frames inline if returned
                if frames:
                    # frames are local paths on server; Streamlit is local so it can open them
                    for p in frames:
                        if os.path.exists(p):
                            st.image(p, use_column_width=True)
                        else:
                            st.write(f"Frame not found: {p}")
            else:
                st.error(f"Backend error: {res.text}")
        except Exception as e:
            st.error(f"Query error: {e}")