"""Grid search for the optimal fusion coefficient (alpha).

Alpha controls the balance between Whisper ASR confidence and CLIP visual
context similarity:

    final = (1 - α) · ASR_prob + α · CLIP_prob

α = 0  →  pure ASR (no visual rescoring)
α = 1  →  pure CLIP (ignoring ASR confidence)

The search pre-computes n-best hypotheses once, then sweeps alpha cheaply.

Usage (from project root):
    python -m multimodal.fusion_tuning  # prints expected sample format
"""

import argparse
import json
from pathlib import Path

import numpy as np
from jiwer import wer as compute_wer
from PIL import Image
import torch

from .multimodal_asr import MultimodalASR

DEFAULT_ALPHA_RANGE = np.round(np.arange(0.0, 1.05, 0.05), 2)


def tune_alpha(
    pipeline: MultimodalASR,
    test_samples: list[dict],
    alpha_range: np.ndarray = DEFAULT_ALPHA_RANGE,
) -> dict:
    """Grid-search alpha on paired image-audio test samples.

    Each element of *test_samples* must contain:
        audio_array  : np.ndarray — raw waveform
        sr           : int        — sampling rate (default 16 000)
        reference    : str        — ground-truth transcription
        image_id     : str | None — cached embedding key  (at least one of
        image_path   : str | None — path to image file      these two)
    """
    # ── Step 1: pre-compute n-best hypotheses + image embeddings ─────
    print(f"Pre-computing n-best hypotheses for {len(test_samples)} samples …")
    precomputed: list[dict] = []

    for i, sample in enumerate(test_samples):
        hyps = pipeline.generate_nbest(
            sample["audio_array"], sample.get("sr", 16000)
        )
        img_emb = pipeline._resolve_image_embedding(
            sample.get("image_id"), sample.get("image_path")
        )
        precomputed.append({
            "hypotheses": hyps,
            "image_embedding": img_emb,
            "reference": sample["reference"].strip().lower(),
        })
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(test_samples)} done")

    # ── Step 2: sweep alpha ──────────────────────────────────────────
    print(f"\nSweeping {len(alpha_range)} alpha values …")
    grid_results: list[dict] = []

    for alpha in alpha_range:
        alpha = float(alpha)
        refs, hyps_texts = [], []

        for item in precomputed:
            ref = item["reference"]
            if item["image_embedding"] is not None:
                rescored = pipeline.rescore(
                    item["hypotheses"], item["image_embedding"], alpha
                )
                best = rescored[0]["text"] if rescored else ""
            else:
                best = (
                    item["hypotheses"][0]["text"] if item["hypotheses"] else ""
                )
            refs.append(ref)
            hyps_texts.append(best)

        alpha_wer = compute_wer(refs, hyps_texts)
        grid_results.append({"alpha": alpha, "wer": alpha_wer})
        print(f"  α = {alpha:.2f}   WER = {alpha_wer * 100:.1f}%")

    best = min(grid_results, key=lambda r: r["wer"])
    baseline_wer = next(r["wer"] for r in grid_results if r["alpha"] == 0.0)

    summary = {
        "grid_results": grid_results,
        "best_alpha": best["alpha"],
        "best_wer": best["wer"],
        "baseline_wer": baseline_wer,
        "wer_reduction": baseline_wer - best["wer"],
        "num_samples": len(test_samples),
        "alpha_range": [float(a) for a in alpha_range],
    }

    print(f"\n{'─' * 50}")
    print(f"Baseline WER (α=0):  {baseline_wer * 100:.1f}%")
    print(f"Best WER:            {best['wer'] * 100:.1f}%  (α = {best['alpha']:.2f})")
    print(f"Absolute reduction:  {(baseline_wer - best['wer']) * 100:.1f}%")

    return summary


def save_tuning_results(results: dict, output_path: Path | str):
    """Persist grid-search results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Tuning results → {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    print("Fusion tuning requires paired image-audio test data.")
    print("Call tune_alpha() programmatically with your test samples.\n")
    print("Expected sample format:")
    print("  {")
    print('      "audio_array": np.ndarray,')
    print('      "sr": 16000,')
    print('      "reference": "a cat sleeping on a windowsill",')
    print('      "image_id": "img_001.png"')
    print("  }")
    print()
    print("Example:")
    print("  from multimodal.multimodal_asr import MultimodalASR")
    print("  from multimodal.fusion_tuning import tune_alpha")
    print()
    print("  pipeline = MultimodalASR(alpha=0.0)")
    print("  results = tune_alpha(pipeline, test_samples)")
    print("  print(f\"Best alpha: {results['best_alpha']}\")")


if __name__ == "__main__":
    main()
