import os
import torch
import numpy as np
import faiss
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Load CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Image directory
IMAGE_DIR = "keyframes"

# Embedding storage
image_embeddings = []
image_map = {}  # index -> file path

# Build image embeddings and FAISS index
def build_index():
    global image_embeddings, image_map, index
    image_embeddings = []
    image_map = {}

    if not os.path.exists(IMAGE_DIR):
        print(f"[Error] Image directory not found: {IMAGE_DIR}")
        return None

    image_files = sorted([
        os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".png"))
    ])

    if not image_files:
        print("[Warning] No keyframes found.")
        return None

    for idx, path in enumerate(image_files):
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            embedding = model.get_image_features(**inputs).cpu().numpy().flatten()
        image_embeddings.append(embedding)
        image_map[idx] = path

    embedding_dim = len(image_embeddings[0])
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.array(image_embeddings).astype(np.float32))

    return index

# Search using text query
def search_query(query, k=5):
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs).cpu().numpy()

    D, I = index.search(text_embedding.astype(np.float32), k)
    results = [image_map[i] for i in I[0] if i in image_map]

    return results, [f"Score: {d:.2f}" for d in D[0]]

# Initialize the index
index = build_index()
