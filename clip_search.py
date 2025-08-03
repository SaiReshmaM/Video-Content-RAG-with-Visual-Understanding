import os
import torch
import faiss
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Path to store image embeddings
INDEX_PATH = "clip.index"
IMAGE_DIR = "frames"

image_map = []  # Maps index to image file path

def embed_and_search(frame_paths, transcript_text):
    images = []
    global image_map
    image_map = []

    for path in frame_paths:
        try:
            image = Image.open(path).convert("RGB")
            images.append(image)
            image_map.append(path)
        except Exception as e:
            print(f"Error reading image {path}: {e}")

    if not images:
        print("No valid images found.")
        return

    # Process and embed images
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        image_embeddings = model.get_image_features(**inputs)

    image_embeddings = image_embeddings.cpu().numpy()
    
    # Save to FAISS index
    index = faiss.IndexFlatL2(image_embeddings.shape[1])
    index.add(image_embeddings)
    faiss.write_index(index, INDEX_PATH)

def search_query(text):
    global image_map
    if not os.path.exists(INDEX_PATH):
        return [], []

    # Load FAISS index
    index = faiss.read_index(INDEX_PATH)

    # Get text embedding
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs)

    text_embedding = text_embedding.cpu().numpy()
    D, I = index.search(text_embedding, k=5)

    results = [image_map[i] for i in I[0] if i < len(image_map)]
    return results, [text] * len(results)
