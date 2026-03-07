"""
Visualize CLIP's shared embedding space: one image vs. multiple text candidates.

Projects the 512-d CLIP embeddings down to 2D with PCA, then draws a scatter
plot where the spatial distance between points reflects cosine similarity.
The most similar text naturally clusters closest to the image embedding.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import torch
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from sklearn.decomposition import PCA
from transformers import CLIPModel, CLIPProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = PROJECT_ROOT / "imagegen" / "images" / "img_001.png"
OUTPUT_PATH = PROJECT_ROOT / "multimodal" / "clip_text_image_similarity.png"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

TEXT_CANDIDATES = [
    "a cat sleeping on a windowsill",
    "a cat resting near a window",
    "a small animal sleeping indoors",
    "a cat watching birds outside",
    "a dog catching a ball in a park",
    "two children building a sandcastle",
    "a boat floating on a calm lake",
]


# ── helpers ──────────────────────────────────────────────────────────────────


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def encode_image(model, processor, image_path, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        vis_out = model.vision_model(pixel_values=pixel_values)
        features = model.visual_projection(vis_out.pooler_output)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().squeeze()


def encode_texts(model, processor, texts, device):
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        txt_out = model.text_model(input_ids=input_ids, attention_mask=attention_mask)
        features = model.text_projection(txt_out.pooler_output)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    device = _pick_device()
    print(f"Device: {device}")

    print("Loading CLIP model...")
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    model.eval()

    print("Encoding image...")
    img_emb = encode_image(model, processor, IMAGE_PATH, device)

    print("Encoding text candidates...")
    txt_embs = encode_texts(model, processor, TEXT_CANDIDATES, device)

    similarities = txt_embs @ img_emb  # dot product on unit vectors = cosine sim

    # Stack image embedding (row 0) + text embeddings for joint PCA
    all_embs = np.vstack([img_emb.reshape(1, -1), txt_embs])  # (N+1, 512)
    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(all_embs)

    img_xy = coords_2d[0]
    txt_xy = coords_2d[1:]

    # ── plot ──────────────────────────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(12, 8), facecolor="#fafafa")
    ax.set_facecolor("#fafafa")

    sim_min, sim_max = similarities.min(), similarities.max()
    norm_sims = (similarities - sim_min) / (sim_max - sim_min + 1e-8)

    cmap = plt.cm.RdYlGn
    best_idx = int(np.argmax(similarities))

    # Draw connection lines (opacity proportional to similarity)
    for i, (xy, ns) in enumerate(zip(txt_xy, norm_sims)):
        alpha = 0.25 + 0.65 * ns
        lw = 1.0 + 2.5 * ns
        ax.plot(
            [img_xy[0], xy[0]],
            [img_xy[1], xy[1]],
            color=cmap(ns),
            alpha=alpha,
            lw=lw,
            zorder=1,
            linestyle="--" if i != best_idx else "-",
        )

    # Text-embedding scatter points
    scatter = ax.scatter(
        txt_xy[:, 0],
        txt_xy[:, 1],
        c=norm_sims,
        cmap=cmap,
        s=200,
        edgecolors="white",
        linewidths=1.5,
        zorder=3,
    )

    # Highlight the best match with a ring
    ax.scatter(
        [txt_xy[best_idx, 0]],
        [txt_xy[best_idx, 1]],
        s=450,
        facecolors="none",
        edgecolors=cmap(1.0),
        linewidths=2.5,
        zorder=4,
    )

    # Text labels
    for i, (xy, txt, sim) in enumerate(
        zip(txt_xy, TEXT_CANDIDATES, similarities)
    ):
        label = f'"{txt}"\nsim = {sim:.3f}'
        weight = "bold" if i == best_idx else "normal"
        fontsize = 9.5 if i == best_idx else 8.5
        offset = (14, 8) if i % 2 == 0 else (14, -12)
        ax.annotate(
            label,
            xy=(xy[0], xy[1]),
            xytext=offset,
            textcoords="offset points",
            fontsize=fontsize,
            fontweight=weight,
            color="#222",
            path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            zorder=5,
        )

    # Image embedding — show thumbnail
    ax.scatter(
        [img_xy[0]], [img_xy[1]], s=400, marker="*", color="#e63946",
        edgecolors="white", linewidths=1.5, zorder=6,
    )
    img_thumb = Image.open(IMAGE_PATH).convert("RGB").resize((80, 80))
    imagebox = OffsetImage(np.array(img_thumb), zoom=1.0)
    ab = AnnotationBbox(
        imagebox, (img_xy[0], img_xy[1]),
        xybox=(-60, 50), boxcoords="offset points",
        frameon=True,
        bboxprops=dict(boxstyle="round,pad=0.3", fc="white", ec="#e63946", lw=2),
        arrowprops=dict(arrowstyle="->", color="#e63946", lw=1.5),
    )
    ax.add_artist(ab)
    ax.annotate(
        "Image Embedding\n(img_001.png)",
        xy=(img_xy[0], img_xy[1]),
        xytext=(-60, -35),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
        color="#e63946",
        ha="center",
        path_effects=[pe.withStroke(linewidth=3, foreground="white")],
        zorder=5,
    )

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Cosine Similarity (normalised)", fontsize=10)

    # Title and axis styling
    ax.set_title(
        "CLIP Shared Embedding Space: Image vs. Text Candidates\n"
        "(PCA projection — closer = higher cosine similarity)",
        fontsize=13,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel(f"PC 1  ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=10)
    ax.set_ylabel(f"PC 2  ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=10)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    margin = 0.08
    x_range = txt_xy[:, 0].max() - txt_xy[:, 0].min()
    y_range = txt_xy[:, 1].max() - txt_xy[:, 1].min()
    ax.set_xlim(
        min(img_xy[0], txt_xy[:, 0].min()) - margin * x_range - 0.5,
        max(img_xy[0], txt_xy[:, 0].max()) + margin * x_range + 0.5,
    )
    ax.set_ylim(
        min(img_xy[1], txt_xy[:, 1].min()) - margin * y_range - 0.5,
        max(img_xy[1], txt_xy[:, 1].max()) + margin * y_range + 0.5,
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight", facecolor="#fafafa")
    print(f"\nSaved to {OUTPUT_PATH}")

    print("\n── Cosine similarities ──")
    order = np.argsort(-similarities)
    for rank, i in enumerate(order, 1):
        marker = " <-- best match" if i == best_idx else ""
        print(f"  {rank}. {similarities[i]:.4f}  \"{TEXT_CANDIDATES[i]}\"{marker}")


if __name__ == "__main__":
    main()
