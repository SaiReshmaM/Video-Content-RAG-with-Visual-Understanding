import whisper
import os

def transcribe_audio(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    model = whisper.load_model("base")
    result = model.transcribe(video_path)
    transcript_text = result["text"]

    os.makedirs("transcripts", exist_ok=True)
    filename = os.path.splitext(os.path.basename(video_path))[0]
    transcript_file = f"transcripts/transcript_{filename}.txt"

    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(transcript_text)

    return transcript_text