# fusion_pipeline.py
"""
Robust FusionPipeline replacement.
- Handles many DataLoader batch shapes robustly.
- Skips bad samples (with warnings) instead of crashing.
- Saves projection weights only if training processed >=1 batch.
All files (weights/caches) are saved in the working directory to meet submission rules.
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
    """Dataset returns (image_path, text) tuples."""
    def __init__(self, pairs: List[Tuple[str, str]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

def unpack_batch(batch):
    """
    Robustly unpack a DataLoader batch into (img_paths_list, texts_list).
    Accepts:
      - list of tuples: [(img1, txt1), (img2, txt2), ...]
      - tuple of lists: ([img1,img2,...], [txt1,txt2,...])
      - nested singletons, etc.
    Returns: (list_of_img_paths, list_of_texts) or ([],[]) on failure.
    """
    # empty batch
    if batch is None:
        return [], []
    # If it's a list
    if isinstance(batch, list):
        if len(batch) == 0:
            return [], []
        # list of tuples/lists
        if isinstance(batch[0], (tuple, list)) and len(batch[0]) >= 2:
            try:
                img_paths, texts = zip(*batch)
                return list(img_paths), list(texts)
            except Exception:
                # try flattening
                flat = []
                for item in batch:
                    if isinstance(item, (tuple, list)) and len(item) >= 2:
                        flat.append((item[0], item[1]))
                if flat:
                    img_paths, texts = zip(*flat)
                    return list(img_paths), list(texts)
                return [], []
        # else it's a list but maybe nested differently: try zip
        try:
            img_paths, texts = zip(*batch)
            return list(img_paths), list(texts)
        except Exception:
            return [], []

    # If it's a tuple (likely tuple-of-lists)
    if isinstance(batch, tuple):
        if len(batch) == 2 and isinstance(batch[0], (list, tuple)) and isinstance(batch[1], (list, tuple)):
            return list(batch[0]), list(batch[1])
        # otherwise try to zip
        try:
            img_paths, texts = zip(*batch)
            return list(img_paths), list(texts)
        except Exception:
            # try to extract (a,b) pairs from elements
            flat = []
            for item in batch:
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    flat.append((item[0], item[1]))
            if flat:
                img_paths, texts = zip(*flat)
                return list(img_paths), list(texts)
            return [], []

    # fallback
    try:
        img_paths, texts = zip(*batch)
        return list(img_paths), list(texts)
    except Exception:
        return [], []

class FusionPipeline:
    def __init__(self,
                 text_model_name="sentence-transformers/all-MiniLM-L6-v2",
                 clip_model_name="openai/clip-vit-base-patch32",
                 device: Optional[str] = None,
                 proj_dim: int = 256):
        """
        Initialize the pipeline and projection heads.
        device: 'cuda' or 'cpu' (auto-detected if None).
        proj_dim: target projection dimensionality for both modalities.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[FusionPipeline] device={self.device}")

        # Load encoders
        self.text_pipe = TextPipeline(model_name=text_model_name, device=self.device)
        self.img_pipe = ImagePipeline(model_name=clip_model_name, device=self.device)

        # infer text dim
        try:
            sample_text = "sample text to infer dimension"
            t_emb = self.text_pipe.encode(sample_text, use_cache=False)
            text_dim = t_emb.shape[-1]
            print(f"[FusionPipeline] detected text embedding dim = {text_dim}")
        except Exception as e:
            print("[FusionPipeline] Warning: couldn't infer text dim, defaulting to 384:", e)
            text_dim = 384

        # infer img dim (try to encode a local png if present)
        img_dim = None
        try:
            demo_candidates = [p for p in os.listdir(".") if p.lower().endswith(".png")]
            if demo_candidates:
                demo_img = demo_candidates[0]
                i_emb = self.img_pipe.encode(demo_img, use_cache=False)
                img_dim = i_emb.shape[-1]
                print(f"[FusionPipeline] detected image embedding dim = {img_dim} (from {demo_img})")
        except Exception as e:
            print("[FusionPipeline] Warning: couldn't infer img dim:", e)

        if img_dim is None:
            img_dim = 512
            print(f"[FusionPipeline] falling back to default img dim = {img_dim}")

        # projection heads
        self.text_proj = SimpleProjection(in_dim=text_dim, out_dim=proj_dim).to(self.device)
        self.img_proj = SimpleProjection(in_dim=img_dim, out_dim=proj_dim).to(self.device)

        self._weights_file = "proj_weights.pt"

    def fuse(self, clinical_note: str, image_path: str, use_cache=True) -> float:
        """
        Compute cosine similarity between projected text and image embeddings.
        Returns float in [-1,1]. If encoding fails returns 0.0.
        """
        with torch.no_grad():
            try:
                t_emb = self.text_pipe.encode(clinical_note, use_cache=use_cache).float().to(self.device)
            except Exception as e:
                print(f"[Fusion] Error encoding text: {e}; returning 0.0")
                return 0.0
            try:
                i_emb = self.img_pipe.encode(image_path, use_cache=use_cache).float().to(self.device)
            except Exception as e:
                print(f"[Fusion] Error encoding image {image_path}: {e}; returning 0.0")
                return 0.0

            t_proj = self.text_proj(t_emb.unsqueeze(0)).squeeze(0)
            i_proj = self.img_proj(i_emb.unsqueeze(0)).squeeze(0)

            t_norm = F.normalize(t_proj, dim=-1)
            i_norm = F.normalize(i_proj, dim=-1)
            score = float(torch.matmul(t_norm, i_norm).cpu().item())
        return score

    def train_projection(self, pairs: List[Tuple[str, str]],
                         epochs: int = 6, batch_size: int = 4,
                         lr: float = 1e-3, temperature: float = 0.07,
                         save: bool = True):
        """
        Train only projection heads using contrastive loss.
        - Silently skips bad samples
        - Saves weights only if >=1 batch processed
        """
        if not pairs:
            print("[Fusion.train_projection] No pairs provided; returning.")
            return

        ds = PairDataset(pairs)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
        opt = torch.optim.Adam(list(self.text_proj.parameters()) + list(self.img_proj.parameters()), lr=lr)

        self.text_proj.train(); self.img_proj.train()

        total_processed_batches = 0

        for epoch in range(epochs):
            total_loss = 0.0
            processed_batches = 0
            for batch_idx, batch in enumerate(loader):
                img_paths, texts = unpack_batch(batch)
                if not img_paths or not texts:
                    print(f"[Fusion.train_projection] Could not unpack batch #{batch_idx}; skipping")
                    continue

                # Encode images, skip failures
                valid_pairs = []
                for p, t in zip(img_paths, texts):
                    try:
                        emb_i = self.img_pipe.encode(p, use_cache=False).float()
                        valid_pairs.append((emb_i, t))
                    except Exception as e:
                        print(f"[Fusion.train_projection] Warning: failed image encode {p}: {e}; skipping sample")
                        continue

                if not valid_pairs:
                    print(f"[Fusion.train_projection] Warning: no valid pairs in batch #{batch_idx}; skipping")
                    continue

                # Build tensors
                img_embs = torch.stack([v[0] for v in valid_pairs]).to(self.device)
                txts = [v[1] for v in valid_pairs]

                # Encode texts (try batch encode, fallback per-sample)
                try:
                    txt_embs = self.text_pipe.model.encode(txts, convert_to_tensor=True).to(self.device).float()
                except Exception as e:
                    print(f"[Fusion.train_projection] Warning: batch text encode failed: {e}; trying per-sample")
                    tmp = []
                    for tt in txts:
                        try:
                            tmp_emb = self.text_pipe.encode(tt, use_cache=False).float()
                            tmp.append(tmp_emb)
                        except Exception as e2:
                            print(f"[Fusion.train_projection] Warning: single text encode failed for '{tt}': {e2}")
                    if not tmp:
                        print("[Fusion.train_projection] Warning: no valid text embeddings for this batch; skipping")
                        continue
                    txt_embs = torch.stack(tmp).to(self.device).float()

                # Ensure same batch size
                if img_embs.size(0) != txt_embs.size(0):
                    m = min(img_embs.size(0), txt_embs.size(0))
                    img_embs = img_embs[:m]
                    txt_embs = txt_embs[:m]

                # Project
                i_proj = self.img_proj(img_embs)
                t_proj = self.text_proj(txt_embs)

                # Normalize
                i_norm = F.normalize(i_proj, dim=-1)
                t_norm = F.normalize(t_proj, dim=-1)

                logits = torch.matmul(t_norm, i_norm.t()) / temperature
                labels = torch.arange(logits.size(0)).long().to(self.device)

                loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss += float(loss.item())
                processed_batches += 1

            avg_loss = (total_loss / processed_batches) if processed_batches else 0.0
            print(f"[Fusion.train_projection] Epoch {epoch+1}/{epochs} avg_loss={avg_loss:.4f} (processed_batches={processed_batches})")
            total_processed_batches += processed_batches

        # Save only if some batches processed
        if save and total_processed_batches > 0:
            try:
                torch.save({
                    "text_proj": self.text_proj.state_dict(),
                    "img_proj": self.img_proj.state_dict()
                }, self._weights_file)
                print(f"[Fusion.train_projection] Saved projection weights to {self._weights_file}")
            except Exception as e:
                print(f"[Fusion.train_projection] Warning: failed to save weights: {e}")
        else:
            print("[Fusion.train_projection] No batches processed -> not saving projection weights.")

        self.text_proj.eval(); self.img_proj.eval()

    def load_projection(self, path: Optional[str] = None):
        p = path or self._weights_file
        if os.path.exists(p):
            try:
                st = torch.load(p, map_location=self.device)
                self.text_proj.load_state_dict(st["text_proj"])
                self.img_proj.load_state_dict(st["img_proj"])
                print(f"[FusionPipeline] Loaded projection weights from {p}")
            except Exception as e:
                print(f"[FusionPipeline] Error loading projection weights from {p}: {e}")
        else:
            print(f"[FusionPipeline] No projection weights found at {p}; continue without loading.")
