import os
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss

# Constants
IMAGE_DIR = "keyframes"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Globals
image_map = {}
index = None

def build_index():
    global image_map, index
    image_embeddings = []
    image_map.clear()

    if not os.path.exists(IMAGE_DIR):
        print(f"Directory '{IMAGE_DIR}' not found.")
        return None

    image_files = [
        os.path.join(IMAGE_DIR, f)
        for f in os.listdir(IMAGE_DIR)
        if f and f.lower().endswith((".jpg", ".png")) and os.path.isfile(os.path.join(IMAGE_DIR, f))
    ]

    if not image_files:
        print("No images found in keyframes.")
        return None

    for idx, path in enumerate(image_files):
        try:
            image = Image.open(path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                embedding = model.get_image_features(**inputs).cpu().numpy().flatten()
            image_embeddings.append(embedding)
            image_map[idx] = path
        except Exception as e:
            print(f"Error processing {path}: {e}")

    embedding_dim = len(image_embeddings[0])
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(image_embeddings).astype(np.float32))
    return index

def search_query(text_query, k=3):
    if index is None:
        build_index()

    if index is None or index.ntotal == 0:
        return [], []

    inputs = processor(text=[text_query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs).cpu().numpy().astype(np.float32)

    D, I = index.search(text_embedding, k)

    results = []
    captions = []
    for i in I[0]:
        if i in image_map:
            results.append(image_map[i])
            captions.append(f"Matched: {os.path.basename(image_map[i])}")
    return results, captions
