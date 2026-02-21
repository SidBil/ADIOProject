"""Generate and cache CLIP embeddings for all images in the image bank.

Produces a .npz cache file mapping each image filename to its L2-normalized
CLIP vision embedding, plus a metadata JSON with model info and file list.

Usage (from project root):
    python -m multimodal.clip_embeddings [--image-dir imagegen/images] [--cache-dir multimodal/cache]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_IMAGE_DIR = Path("imagegen/images")
DEFAULT_CACHE_DIR = Path("multimodal/cache")


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_clip(
    model_name: str = DEFAULT_CLIP_MODEL, device: str | None = None
) -> tuple[CLIPModel, CLIPProcessor, str]:
    """Load CLIP model and processor onto the chosen device."""
    if device is None:
        device = _pick_device()
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, processor, device


def encode_images(
    model: CLIPModel,
    processor: CLIPProcessor,
    image_dir: Path | str,
    device: str,
) -> dict[str, np.ndarray]:
    """Encode every PNG in *image_dir* and return {filename: embedding}."""
    image_dir = Path(image_dir)
    image_files = sorted(image_dir.glob("*.png"))
    if not image_files:
        raise FileNotFoundError(f"No .png files found in {image_dir}")

    embeddings: dict[str, np.ndarray] = {}
    for img_path in image_files:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            features = _get_image_features(model, pixel_values)
        embeddings[img_path.name] = features.cpu().numpy().squeeze()
        print(f"  Encoded {img_path.name}")

    return embeddings


def encode_texts(
    model: CLIPModel,
    processor: CLIPProcessor,
    texts: list[str],
    device: str,
) -> np.ndarray:
    """Encode a batch of texts and return L2-normalised embeddings (N, D)."""
    inputs = processor(
        text=texts, return_tensors="pt", padding=True, truncation=True
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        features = _get_text_features(model, input_ids, attention_mask)
    return features.cpu().numpy()


def _get_image_features(model: CLIPModel, pixel_values: torch.Tensor) -> torch.Tensor:
    """Extract L2-normalised image embeddings (works across transformers versions)."""
    vision_out = model.vision_model(pixel_values=pixel_values)
    features = model.visual_projection(vision_out.pooler_output)
    return features / features.norm(dim=-1, keepdim=True)


def _get_text_features(
    model: CLIPModel, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Extract L2-normalised text embeddings (works across transformers versions)."""
    text_out = model.text_model(input_ids=input_ids, attention_mask=attention_mask)
    features = model.text_projection(text_out.pooler_output)
    return features / features.norm(dim=-1, keepdim=True)


def cache_embeddings(embeddings: dict[str, np.ndarray], cache_path: Path | str):
    """Save image embeddings to a .npz file."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(cache_path), **embeddings)
    print(f"Cached {len(embeddings)} embeddings → {cache_path}")


def load_cached_embeddings(cache_path: Path | str) -> dict[str, np.ndarray]:
    """Load embeddings previously saved with *cache_embeddings*."""
    data = np.load(str(cache_path))
    return dict(data)


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Pre-compute CLIP image embeddings")
    parser.add_argument(
        "--image-dir", type=Path, default=DEFAULT_IMAGE_DIR, help="Directory of .png images"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Where to write cache files"
    )
    parser.add_argument("--model", default=DEFAULT_CLIP_MODEL, help="HuggingFace CLIP model id")
    args = parser.parse_args()

    model, processor, device = load_clip(args.model)
    print(f"Device: {device}  |  CLIP model: {args.model}")

    embeddings = encode_images(model, processor, args.image_dir, device)

    cache_path = args.cache_dir / "clip_image_embeddings.npz"
    cache_embeddings(embeddings, cache_path)

    metadata = {
        "model": args.model,
        "num_images": len(embeddings),
        "image_files": sorted(embeddings.keys()),
        "embedding_dim": int(next(iter(embeddings.values())).shape[0]),
    }
    meta_path = args.cache_dir / "clip_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata → {meta_path}")


if __name__ == "__main__":
    main()
