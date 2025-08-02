from transformers import CLIPProcessor, CLIPModel
import torch
import faiss
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
index = faiss.IndexFlatL2(512)

image_map = []
text_list = []

def embed_and_search(frame_paths, transcript):
    global image_map, text_list
    image_map.clear()
    text_list.clear()

    for path in frame_paths:
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            emb = model.get_image_features(**inputs).cpu().numpy()
        index.add(emb)
        image_map.append(path)
        text_list.append(transcript)

def search_query(query):
    inputs = processor(text=query, return_tensors="pt", padding=True)
    with torch.no_grad():
        query_emb = model.get_text_features(**inputs).cpu().numpy()
    D, I = index.search(query_emb, k=3)
    results = [image_map[i] for i in I[0]]
    texts = [text_list[i] for i in I[0]]
    return results, texts
