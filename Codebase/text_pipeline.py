# text_pipeline.py
"""
TextPipeline: small wrapper around sentence-transformers for quick text embeddings.
Saves cache files in the same directory (no subfolders) as text_{hash}.pt.
"""

import os
import torch
from sentence_transformers import SentenceTransformer

class TextPipeline:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device=None):
        """
        Initialize the text encoder.
        model_name: Hugging Face / sentence-transformers model id.
        device: 'cuda' or 'cpu' (auto-selected if None).
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[TextPipeline] loading {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, text, use_cache=True):
        """
        Return embedding tensor for `text`. Optionally cache to ./text_{hash}.pt
        """
        key = f"text_{abs(hash(text))}.pt"
        if use_cache and os.path.exists(key):
            try:
                emb = torch.load(key)
                return emb.to(self.device)
            except Exception:
                pass
        emb = self.model.encode(text, convert_to_tensor=True)
        if use_cache:
            try:
                torch.save(emb.cpu(), key)
            except Exception:
                pass
        return emb.to(self.device)
