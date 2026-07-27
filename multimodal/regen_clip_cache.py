#!/usr/bin/env python3
"""
Regenerate the CLIP image-embedding cache after the 9:10 image regeneration.

Replicates multimodal/clip_embeddings.ipynb exactly:
  - openai/clip-vit-base-patch32
  - vision_model -> visual_projection -> L2-normalise  (512-dim float32)
  - keyed by filename ("img_XXX.png")

Encodes from web/data/images (the dir the backend actually serves) and writes to
BOTH cache locations so research + production stay in sync:
  - web/data/cache/clip_image_embeddings.npz   (consumed by asr_service / baked into Modal)
  - multimodal/cache/clip_image_embeddings.npz  (research copy)
plus a clip_metadata.json alongside each.

Run from repo root:
  source venv/bin/activate && python multimodal/regen_clip_cache.py
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

CLIP_MODEL = "openai/clip-vit-base-patch32"
REPO_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = REPO_ROOT / "web" / "data" / "images"
CACHE_TARGETS = [
    REPO_ROOT / "web" / "data" / "cache",   # production (asr_service + Modal build)
    REPO_ROOT / "multimodal" / "cache",     # research copy
]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_image_features(model, pixel_values):
    """L2-normalised image embeddings — matches asr_service text-side normalisation."""
    vision_out = model.vision_model(pixel_values=pixel_values)
    features = model.visual_projection(vision_out.pooler_output)
    return features / features.norm(dim=-1, keepdim=True)


def main():
    device = pick_device()
    print(f"Device: {device}  |  CLIP model: {CLIP_MODEL}")
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
    model.eval()

    image_files = sorted(IMAGE_DIR.glob("*.png"))
    if not image_files:
        raise FileNotFoundError(f"No .png files in {IMAGE_DIR}")

    embeddings: dict[str, np.ndarray] = {}
    for img_path in image_files:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            features = get_image_features(model, pixel_values)
        embeddings[img_path.name] = features.cpu().numpy().squeeze().astype(np.float32)
        print(f"  Encoded {img_path.name}  (norm={np.linalg.norm(embeddings[img_path.name]):.4f})")

    metadata = {
        "model": CLIP_MODEL,
        "num_images": len(embeddings),
        "image_files": sorted(embeddings.keys()),
        "embedding_dim": int(next(iter(embeddings.values())).shape[0]),
    }

    for cache_dir in CACHE_TARGETS:
        cache_dir.mkdir(parents=True, exist_ok=True)
        npz_path = cache_dir / "clip_image_embeddings.npz"
        np.savez(str(npz_path), **embeddings)
        with open(cache_dir / "clip_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Wrote {len(embeddings)} embeddings -> {npz_path}")


if __name__ == "__main__":
    main()
