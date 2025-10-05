# fusion_pipeline.py
"""
Robust FusionPipeline (updated)
- Replaces BatchNorm1d with LayerNorm to avoid batch-size=1 issues.
- Robust unpack_batch() that detects image-path vs text using heuristics (os.path.exists, extension check).
- Skips batches with < 2 valid pairs (contrastive loss needs >=2).
- Saves projection weights only if >=1 processed batch.
- Ensures projection heads are eval() during inference (fuse()).
All outputs and caches remain in the working directory (no subdirs) to meet submission rules.
"""

import os
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from text_pipeline import TextPipeline
from image_pipeline import ImagePipeline

# Allowed image extensions for quick heuristic
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

class SimpleProjection(nn.Module):
    """Two-layer MLP projection head; uses LayerNorm instead of BatchNorm."""
    def __init__(self, in_dim, out_dim=256, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),   # safe for batch_size=1
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

def _looks_like_path(s: str) -> bool:
    """Heuristic: string looks like a path if it exists or ends with common image extension."""
    if not isinstance(s, str):
        return False
    # existing file
    if os.path.exists(s):
        return True
    # extension heuristic
    _, ext = os.path.splitext(s.lower())
    if ext in _IMAGE_EXTS:
        return True
    # quick path-like tokens (./ or /)
    if s.startswith("./") or s.startswith("/") or "\\" in s:
        return True
    return False

def unpack_batch(batch):
    """
    Robustly unpack a DataLoader batch into (list_of_img_paths, list_of_texts).
    Handles many shapes:
      - list of tuples: [(img1, txt1), (img2, txt2), ...]
      - tuple of lists: ([img1,img2,...], [txt1,txt2,...])
      - nested / singleton variants
    Uses per-item heuristics to detect which element is image path vs text.
    Returns two lists (img_paths, texts). Could return empty lists on failure.
    """
    if batch is None:
        return [], []

    # Case: tuple-of-lists (common)
    if isinstance(batch, tuple) and len(batch) == 2 and isinstance(batch[0], (list, tuple)) and isinstance(batch[1], (list, tuple)):
        # try to determine which list corresponds to paths by checking first element
        a0 = batch[0][0] if len(batch[0]) > 0 else None
        b0 = batch[1][0] if len(batch[1]) > 0 else None
        # if a0 looks like path, good. else if b0 looks like path, swap.
        if _looks_like_path(a0):
            return list(batch[0]), list(batch[1])
        if _looks_like_path(b0):
            return list(batch[1]), list(batch[0])
        # fallback: assume first is images
        return list(batch[0]), list(batch[1])

    # Case: list of tuples/lists
    if isinstance(batch, list):
        img_paths = []
        texts = []
        for item in batch:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                # try to handle nested singletons or strange collates: skip
                continue
            a, b = item[0], item[1]
            # prefer the element that looks like a path
            if _looks_like_path(a) and not _looks_like_path(b):
                img_paths.append(a); texts.append(b); continue
            if _looks_like_path(b) and not _looks_like_path(a):
                img_paths.append(b); texts.append(a); continue
            # if both or neither look like path, use type and content heuristics:
            # if a contains newline or long? assume it's text
            if isinstance(a, str) and isinstance(b, str):
                # prefer the one with image extension
                _, ext_a = os.path.splitext(a.lower())
                _, ext_b = os.path.splitext(b.lower())
                if ext_a in _IMAGE_EXTS and ext_b not in _IMAGE_EXTS:
                    img_paths.append(a); texts.append(b); continue
                if ext_b in _IMAGE_EXTS and ext_a not in _IMAGE_EXTS:
                    img_paths.append(b); texts.append(a); continue
                # fallback: assume first is image path (legacy behavior)
                img_paths.append(a); texts.append(b); continue
            # if not strings (unlikely), append as-is (best-effort)
            img_paths.append(a); texts.append(b)
        return img_paths, texts

    # Fallback: try unzip
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
        # ensure projections are in eval mode (LayerNorm is safe, but this avoids any training-time behavior)
        self.text_proj.eval()
        self.img_proj.eval()

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
        - pairs: list of (image_path, clinical_note)
        - Robust: skips invalid samples; skips batches with < 2 valid pairs.
        - Saves weights only if some batches were processed successfully.
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

                # Build valid pair list
                valid_img_embs = []
                valid_texts = []
                for p, t in zip(img_paths, texts):
                    # If either feels like path but doesn't exist, skip gracefully
                    if not _looks_like_path(p):
                        # maybe swapped; check t
                        if _looks_like_path(t):
                            p, t = t, p  # swap if second looks like path
                    # if p still not path and t is path, swap
                    if not _looks_like_path(p) and _looks_like_path(t):
                        p, t = t, p
                    # now check file existence
                    if not os.path.exists(p):
                        print(f"[Fusion.train_projection] Warning: image file not found '{p}'; skipping sample")
                        continue
                    try:
                        emb_i = self.img_pipe.encode(p, use_cache=False).float()
                    except Exception as e:
                        print(f"[Fusion.train_projection] Warning: failed image encode {p}: {e}; skipping sample")
                        continue
                    valid_img_embs.append(emb_i)
                    valid_texts.append(t)

                if len(valid_img_embs) < 2:
                    print(f"[Fusion.train_projection] Warning: only {len(valid_img_embs)} valid pairs in batch #{batch_idx}; need >=2 for contrastive; skipping")
                    continue

                # Encode texts (batch)
                try:
                    txt_embs = self.text_pipe.model.encode(valid_texts, convert_to_tensor=True).to(self.device).float()
                except Exception:
                    # fallback per-item
                    tmp = []
                    for tt in valid_texts:
                        try:
                            tmp_emb = self.text_pipe.encode(tt, use_cache=False).float()
                            tmp.append(tmp_emb)
                        except Exception as e:
                            print(f"[Fusion.train_projection] Warning: failed single text encode for '{tt}': {e}; skipping")
                    if len(tmp) < 2:
                        print("[Fusion.train_projection] Warning: insufficient valid text embeddings after fallback; skipping")
                        continue
                    txt_embs = torch.stack(tmp).to(self.device).float()

                img_embs_tensor = torch.stack(valid_img_embs).to(self.device)

                # ensure same batch size
                if img_embs_tensor.size(0) != txt_embs.size(0):
                    m = min(img_embs_tensor.size(0), txt_embs.size(0))
                    img_embs_tensor = img_embs_tensor[:m]
                    txt_embs = txt_embs[:m]

                # Project
                i_proj = self.img_proj(img_embs_tensor)
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

        # Save only if processed at least one batch
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
            print("[Fusion.train_projection] No processed batches -> not saving projection weights.")

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
