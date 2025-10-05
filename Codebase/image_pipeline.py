# image_pipeline.py
"""
ImagePipeline: wrapper around CLIP model to produce image embeddings.
Saves cache files in the same directory as img_{basename}_{mtime}.pt
(no sub-directories).
"""

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os

class ImagePipeline:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        """
        Initialize CLIP image encoder.
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ImagePipeline] loading {model_name} on {self.device}")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def encode(self, image_path, use_cache=True):
        """
        Return 1-D image feature tensor. Cache name: img_{basename}_{mtime}.pt
        """
        mtime = int(os.path.getmtime(image_path))
        basename = os.path.basename(image_path)
        key = f"img_{basename}_{mtime}.pt"
        if use_cache and os.path.exists(key):
            try:
                emb = torch.load(key)
                return emb.to(self.device)
            except Exception:
                pass
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.model.get_image_features(**inputs).squeeze(0)
        if use_cache:
            try:
                torch.save(emb.cpu(), key)
            except Exception:
                pass
        return emb.to(self.device)
