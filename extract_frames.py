import cv2
import os

def extract_key_frames(video_path):
    os.makedirs("frames", exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    count, saved = 0, 0
    frame_paths = []

    while True:
        success, frame = vidcap.read()
        if not success:
            break
        if count % 30 == 0:
            path = f"frames/frame{saved}.jpg"
            cv2.imwrite(path, frame)
            frame_paths.append(path)
            saved += 1
        count += 1

    return frame_paths
