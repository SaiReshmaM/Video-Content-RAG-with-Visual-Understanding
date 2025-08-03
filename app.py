import os
import streamlit as st

st.write("FFMPEG:", os.system("which ffmpeg"))
st.write("LIBGL:", os.system("ldconfig -p | grep libGL"))

# 🧠 Your existing imports
from extract_frames import extract_key_frames
from transcribe_audio import transcribe_audio
from transcribe_audio import transcribe_audio
from extract_frames import extract_key_frames
from clip_search import embed_and_search, search_query



st.title(" Video RAG with Audio + Visual Understanding")

uploaded_video = st.file_uploader("Upload a video", type=["mp4"])

if uploaded_video:
    video_path = os.path.abspath("sample_video.mp4")
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    st.video(video_path)

    if st.button(" Process Video"):
        st.info("Extracting key frames...")
        frame_paths = extract_key_frames(video_path)
        st.success(f"{len(frame_paths)} frames extracted.")

        st.info("Transcribing audio...")
        transcript = transcribe_audio(video_path)
        st.success("Transcription complete and saved.")

        st.info("Indexing with CLIP...")
        embed_and_search(frame_paths, transcript)
        st.success("Search system ready!")

query = st.text_input("Ask something (e.g., 'whiteboard', 'talking about finance')")

if query:
    st.info("Searching...")
    images, texts = search_query(query)
    for img, txt in zip(images, texts):
        st.image(img, width=300)
        st.caption(txt)
        st.write("Starting transcription...")





