# fusion_pipeline.py
"""
Robust FusionPipeline (REPLACEMENT)
- Uses TextPipeline and ImagePipeline from text_pipeline.py and image_pipeline.py
- Projects text and image embeddings to a common latent space (small MLPs)
- Contrastive training available via train_projection()
- This version is resilient to different DataLoader batch shapes and to encoding errors
  (bad images/texts won't crash the training loop; they will be skipped with a warning).
Note: All caches / saved weights are written to the current working directory (no subdirectories),
so they are accessible to the grader (matches submission rules).
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
    """
    Simple paired dataset: returns (image_path, text) tuples.
    DataLoader default collate may return either:
      - list of tuples: [(img1, txt1), (img2, txt2), ...]
      - tuple of lists: ( [img1,img2,...], [txt1,txt2,...] )
    This trainer handles both forms.
    """
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
                 device: Optional[str] = None,
                 proj_dim: int = 256):
        """
        Initialize the pipeline and projection heads.
        device: 'cuda' or 'cpu' (auto-detected if None).
        proj_dim: target projection dimensionality for both modalities.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[FusionPipeline] device={self.device}")

        # Load lightweight encoders
        self.text_pipe = TextPipeline(model_name=text_model_name, device=self.device)
        self.img_pipe = ImagePipeline(model_name=clip_model_name, device=self.device)

        # Determine text embedding dim
        try:
            sample_text = "sample text to infer dimension"
            t_emb = self.text_pipe.encode(sample_text, use_cache=False)
            text_dim = t_emb.shape[-1]
            print(f"[FusionPipeline] detected text embedding dim = {text_dim}")
        except Exception as e:
            print("[FusionPipeline] Warning: couldn't infer text dim, defaulting to 384:", e)
            text_dim = 384

        # Determine image embedding dim (attempt to query CLIP by encoding a demo image if present)
        img_dim = None
        demo_candidates = [p for p in os.listdir(".") if p.lower().endswith(".png")]
        if demo_candidates:
            demo_img = demo_candidates[0]
            try:
                i_emb = self.img_pipe.encode(demo_img, use_cache=False)
                img_dim = i_emb.shape[-1]
                print(f"[FusionPipeline] detected image embedding dim = {img_dim} (from {demo_img})")
            except Exception as e:
                print("[FusionPipeline] Warning: failed to encode demo image to infer dim:", e)
                img_dim = None

        if img_dim is None:
            # fall back to commonly used CLIP dim
            img_dim = 512
            print(f"[FusionPipeline] falling back to default image embedding dim = {img_dim}")

        # Projection heads (map text/image embeddings -> shared proj_dim)
        self.text_proj = SimpleProjection(in_dim=text_dim, out_dim=proj_dim).to(self.device)
        self.img_proj = SimpleProjection(in_dim=img_dim, out_dim=proj_dim).to(self.device)

        # weights file (in working directory)
        self._weights_file = "proj_weights.pt"

    def fuse(self, clinical_note: str, image_path: str, use_cache=True) -> float:
        """
        Compute cosine similarity between projected text and image embeddings.
        Returns a float scalar in [-1, 1].
        """
        with torch.no_grad():
            try:
                t_emb = self.text_pipe.encode(clinical_note, use_cache=use_cache).float().to(self.device)
            except Exception as e:
                print(f"[Fusion] Error encoding text: {e}; returning score 0.0")
                return 0.0
            try:
                i_emb = self.img_pipe.encode(image_path, use_cache=use_cache).float().to(self.device)
            except Exception as e:
                print(f"[Fusion] Error encoding image {image_path}: {e}; returning score 0.0")
                return 0.0

            # Project and normalize
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
        Train only the projection heads with a contrastive (InfoNCE) style loss.
        - pairs: list of (image_path, clinical_note)
        - This function is robust to mixed/failed encodings; bad samples are skipped.
        """
        if not pairs:
            print("[Fusion.train_projection] No pairs provided; aborting training.")
            return

        ds = PairDataset(pairs)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
        optimizer = torch.optim.Adam(list(self.text_proj.parameters()) + list(self.img_proj.parameters()), lr=lr)

        self.text_proj.train(); self.img_proj.train()

        for epoch in range(epochs):
            total_loss = 0.0
            batches = 0
            for batch_idx, batch in enumerate(loader):
                # Handle different batch shapes robustly:
                # - batch could be a list of tuples [(img1,txt1), ...]
                # - or a tuple of lists ([img1,img2], [txt1,txt2])
                try:
                    if isinstance(batch, (list, tuple)) and len(batch) == 2 and isinstance(batch[0], list):
                        # tuple of lists: ( [img_paths], [texts] )
                        img_paths, texts = batch[0], batch[1]
                    else:
                        # likely list of tuples
                        img_paths, texts = zip(*batch)
                except Exception:
                    # fallback: attempt zip
                    try:
                        img_paths, texts = zip(*batch)
                    except Exception as e:
                        print(f"[Fusion.train_projection] Could not unpack batch #{batch_idx}: {e}; skipping batch")
                        continue

                # Encode images (per-sample) and texts in batch; skip those that fail.
                valid_img_embs = []
                valid_texts = []
                for p, t in zip(img_paths, texts):
                    try:
                        emb_i = self.img_pipe.encode(p, use_cache=False).float()
                    except Exception as e:
                        print(f"[Fusion.train_projection] Warning: failed image encode {p}: {e}; skipping sample")
                        continue
                    # record
                    valid_img_embs.append(emb_i)
                    valid_texts.append(t)

                if len(valid_img_embs) == 0:
                    print(f"[Fusion.train_projection] Warning: no valid images in batch #{batch_idx}; skipping")
                    continue

                # Encode texts in batch (try efficient batch encode, fallback to per-sample)
                try:
                    # sentence-transformers provides batch encode
                    text_embs = self.text_pipe.model.encode(list(valid_texts), convert_to_tensor=True)
                    text_embs = text_embs.to(self.device).float()
                except Exception as e:
                    print(f"[Fusion.train_projection] Warning: batch text encode failed: {e}; trying per-sample")
                    tmp = []
                    for tt in valid_texts:
                        try:
                            tmp_emb = self.text_pipe.encode(tt, use_cache=False).float()
                            tmp.append(tmp_emb)
                        except Exception as e2:
                            print(f"[Fusion.train_projection] Warning: failed single text encode for '{tt}': {e2}; skipping")
                    if len(tmp) == 0:
                        print("[Fusion.train_projection] Warning: no valid text embeddings for this batch; skipping")
                        continue
                    text_embs = torch.stack(tmp).to(self.device).float()

                # stack image emb list -> tensor
                try:
                    img_embs_tensor = torch.stack(valid_img_embs).to(self.device)
                except Exception as e:
                    print(f"[Fusion.train_projection] Warning: failed stacking image embeddings: {e}; skipping batch")
                    continue

                # ensure same batch size for images & texts
                if img_embs_tensor.size(0) != text_embs.size(0):
                    min_b = min(img_embs_tensor.size(0), text_embs.size(0))
                    img_embs_tensor = img_embs_tensor[:min_b]
                    text_embs = text_embs[:min_b]

                # project
                i_proj = self.img_proj(img_embs_tensor)
                t_proj = self.text_proj(text_embs)

                # normalize
                i_norm = F.normalize(i_proj, dim=-1)
                t_norm = F.normalize(t_proj, dim=-1)

                # logits and contrastive loss (bidirectional)
                logits = torch.matmul(t_norm, i_norm.t()) / temperature
                labels = torch.arange(logits.size(0)).long().to(self.device)
                loss_t2i = F.cross_entropy(logits, labels)
                loss_i2t = F.cross_entropy(logits.t(), labels)
                loss = 0.5 * (loss_t2i + loss_i2t)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                batches += 1

            avg_loss = (total_loss / batches) if batches > 0 else 0.0
            print(f"[Fusion.train_projection] Epoch {epoch+1}/{epochs} avg_loss={avg_loss:.4f} (batches={batches})")

        # Save projection head weights
        if save:
            try:
                torch.save({
                    "text_proj": self.text_proj.state_dict(),
                    "img_proj": self.img_proj.state_dict()
                }, self._weights_file)
                print(f"[Fusion.train_projection] Saved projection weights to {self._weights_file}")
            except Exception as e:
                print(f"[Fusion.train_projection] Warning: failed to save weights: {e}")

        # switch to eval
        self.text_proj.eval()
        self.img_proj.eval()

    def load_projection(self, path: Optional[str] = None):
        """
        Load saved projection weights (if present).
        """
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
            print(f"[FusionPipeline] No projection weights found at {p}; continuing without loaded weights.")
