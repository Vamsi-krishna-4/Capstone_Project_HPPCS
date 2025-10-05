# fusion_pipeline.py
"""
FusionPipeline:
- Uses TextPipeline and ImagePipeline
- Projects both embeddings to 256-d via small MLP projection heads
- Can train projection heads contrastively on paired data (fast; train only small heads)
- All files (weights, caches) are stored in the same directory (no subfolders)
"""

import os
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from text_pipeline import TextPipeline
from image_pipeline import ImagePipeline

class SimpleProjection(nn.Module):
    """Two-layer MLP projection head."""
    def __init__(self, in_dim, out_dim=256, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class PairDataset(Dataset):
    """Simple paired dataset wrapper of (image_path, text)."""
    def __init__(self, pairs: List[Tuple[str, str]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

class FusionPipeline:
    def __init__(self,
                 text_model_name="sentence-transformers/all-MiniLM-L6-v2",
                 clip_model_name="openai/clip-vit-base-patch32",
                 device=None,
                 proj_dim=256):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.text_pipe = TextPipeline(model_name=text_model_name, device=self.device)
        self.img_pipe = ImagePipeline(model_name=clip_model_name, device=self.device)

        # infer dims via a small sample
        sample_text = "sample"
        t_emb = self.text_pipe.encode(sample_text, use_cache=False)
        text_dim = t_emb.shape[-1]
        # For image, we'll read a small placeholder if exists, else set 512
        img_dim = 512
        # build projections
        self.text_proj = SimpleProjection(in_dim=text_dim, out_dim=proj_dim).to(self.device)
        self.img_proj = SimpleProjection(in_dim=img_dim, out_dim=proj_dim).to(self.device)
        self._weights_file = "proj_weights.pt"

    def fuse(self, clinical_note: str, image_path: str, use_cache=True) -> float:
        """
        Return cosine similarity between projected text and image embeddings.
        """
        with torch.no_grad():
            t_emb = self.text_pipe.encode(clinical_note, use_cache=use_cache).float()
            i_emb = self.img_pipe.encode(image_path, use_cache=use_cache).float()

            # ensure dims: if i_emb dim unknown, pad/truncate to match img_proj.in_features
            # but we assume CLIP gives a consistent dim (we set img_proj based on an assumed 512)
            t_proj = self.text_proj(t_emb.unsqueeze(0).to(self.device)).squeeze(0)
            i_proj = self.img_proj(i_emb.unsqueeze(0).to(self.device)).squeeze(0)

            t_norm = F.normalize(t_proj, dim=-1)
            i_norm = F.normalize(i_proj, dim=-1)
            score = float(torch.matmul(t_norm, i_norm).cpu().item())
        return score

    def train_projection(self, pairs: List[Tuple[str, str]],
                         epochs: int = 6, batch_size: int = 4,
                         lr: float = 1e-3, temperature: float = 0.07,
                         save: bool = True):
        """
        Train projection heads on provided pairs (only small heads are trained).
        Pairs: list of (image_path, text)
        """
        ds = PairDataset(pairs)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
        opt = torch.optim.Adam(list(self.text_proj.parameters()) + list(self.img_proj.parameters()), lr=lr)
        self.text_proj.train(); self.img_proj.train()

        for epoch in range(epochs):
            total_loss = 0.0
            for batch in loader:
                img_paths, texts = zip(*batch)
                # encode images and texts (no caching for training to ensure freshness)
                img_embs = []
                for p in img_paths:
                    emb = self.img_pipe.encode(p, use_cache=False)
                    img_embs.append(emb)
                img_embs = torch.stack(img_embs).to(self.device)

                txt_embs = self.text_pipe.model.encode(list(texts), convert_to_tensor=True).to(self.device)

                i_proj = self.img_proj(img_embs.float())
                t_proj = self.text_proj(txt_embs.float())

                i_norm = F.normalize(i_proj, dim=-1)
                t_norm = F.normalize(t_proj, dim=-1)

                logits = torch.matmul(t_norm, i_norm.t()) / temperature
                labels = torch.arange(logits.size(0)).long().to(self.device)
                loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss += float(loss.item())

            avg = total_loss / (len(loader) or 1)
            print(f"[Fusion train] Epoch {epoch+1}/{epochs} loss={avg:.4f}")

        if save:
            torch.save({
                "text_proj": self.text_proj.state_dict(),
                "img_proj": self.img_proj.state_dict()
            }, self._weights_file)
            print("[Fusion] Saved projection weights to", self._weights_file)

        self.text_proj.eval(); self.img_proj.eval()

    def load_projection(self, path: Optional[str] = None):
        p = path or self._weights_file
        if os.path.exists(p):
            st = torch.load(p, map_location=self.device)
            self.text_proj.load_state_dict(st["text_proj"])
            self.img_proj.load_state_dict(st["img_proj"])
            print("[Fusion] Loaded projection weights from", p)
        else:
            print("[Fusion] projection weights not found; run train_projection() if desired.")
