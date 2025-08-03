# Video-Content-RAG-with-Visual-Understanding

# Problem Statement
Create a multimodal RAG system that processes video content, extracts key frames,
transcribes audio, and allows users to query both visual and audio elements from video
libraries.
# Key Requirements
• Video processing and key frame extraction
• Audio transcription and synchronization
• Visual element recognition and tagging
• Multi-modal search capabilities (visual + audio)
• Temporal indexing with timestamp referencing

 Project Title:
Video Content RAG with Visual Understanding

# Description:
A multimodal retrieval-augmented generation (RAG) system that extracts key frames, transcribes video audio, and allows users to query across both modalities using CLIP and Whisper.

# Features:
Key frame extraction from videos using OpenCV.

Audio transcription with Whisper.

CLIP-based visual semantic search.
# Usage:
Upload a video, get transcriptions and frame-based visual search via prompt.

# Example Prompt:
"person speaking on whiteboard" or "talking about finance"
# Install Python Requirements
Install required Python libraries:
pip install -r requirements.txt
If requirements.txt is not available, manually:
pip install streamlit torch torchvision transformers faiss-cpu openai-whisper opencv-python Pillow
Run the Streamlit App Locally:
streamlit run app.py


