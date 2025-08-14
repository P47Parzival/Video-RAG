# import streamlit as st

# st.title("Video Q&A RAG System")

# # Upload video
# uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov"])
# if uploaded_file:
#     with open("uploaded_video.mp4", "wb") as f:
#         f.write(uploaded_file.getbuffer())
#     st.success("Video uploaded! Processing...")

# # Query input
# query = st.text_input("Ask a question about the video:")
# if st.button("Submit"):
#     # Call backend API to process and get response (implement in next steps)
#     response = get_rag_response("uploaded_video.mp4", query)
#     st.write(response)

import streamlit as st
from functions import get_rag_response

st.title("Video Q&A RAG System")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov"])
if uploaded_file:
    video_path = "uploaded_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("Video uploaded! Processing...")

    query = st.text_input("Ask a question about the video:")
    if st.button("Submit") and query:
        try:
            response = get_rag_response(video_path, query)
            st.write(response)
        except Exception as e:
            st.error(f"Error: {e}")