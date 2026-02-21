"""Complete ASR + CLIP multimodal rescoring pipeline.

Workflow
--------
1. Whisper beam-search → n-best hypotheses with log-prob scores.
2. CLIP encodes the visual context (image) and each text hypothesis.
3. Cosine similarity between image embedding and each text embedding.
4. Fusion:  score = (1-α)·ASR_prob + α·CLIP_prob
5. Re-rank and return the best hypothesis.

Usage (from project root):
    python -m multimodal.multimodal_asr --audio path/to/audio.wav --image imagegen/images/img_001.png
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from .clip_embeddings import (
    DEFAULT_CLIP_MODEL,
    load_clip,
    encode_texts,
    load_cached_embeddings,
    _get_image_features,
)
from .transcript_normalization import normalize_transcript, to_caption_style

DEFAULT_WHISPER_MODEL = "openai/whisper-small"
DEFAULT_CACHE_PATH = Path("multimodal/cache/clip_image_embeddings.npz")


class MultimodalASR:
    """Whisper ASR with optional CLIP visual-context rescoring."""

    def __init__(
        self,
        whisper_model_id: str = DEFAULT_WHISPER_MODEL,
        clip_model_id: str = DEFAULT_CLIP_MODEL,
        cache_path: Path | str = DEFAULT_CACHE_PATH,
        alpha: float = 0.3,
        num_beams: int = 5,
        device: str | None = None,
    ):
        if device is None:
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = device
        self.alpha = alpha
        self.num_beams = num_beams

        # Whisper
        self.whisper_processor = WhisperProcessor.from_pretrained(whisper_model_id)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
            whisper_model_id
        ).to(device)
        self.whisper_model.eval()

        # CLIP
        self.clip_model, self.clip_processor, _ = load_clip(clip_model_id, device)

        # Pre-computed image embeddings
        self.image_embeddings: dict[str, np.ndarray] = {}
        cache_path = Path(cache_path)
        if cache_path.exists():
            self.image_embeddings = load_cached_embeddings(cache_path)
            print(f"Loaded {len(self.image_embeddings)} cached image embeddings")

    # ── n-best generation ────────────────────────────────────────────

    def generate_nbest(
        self, audio_array: np.ndarray, sr: int = 16000
    ) -> list[dict]:
        """Return n-best Whisper hypotheses sorted by score (descending).

        Each entry: {"text": str, "score": float (log-prob)}.
        """
        input_features = self.whisper_processor(
            audio_array, sampling_rate=sr, return_tensors="pt"
        ).input_features.to(self.device)

        with torch.no_grad():
            outputs = self.whisper_model.generate(
                input_features,
                num_beams=self.num_beams,
                num_return_sequences=self.num_beams,
                return_dict_in_generate=True,
                output_scores=True,
            )

        sequences = outputs.sequences
        seq_scores = outputs.sequences_scores.cpu().numpy()

        hypotheses: list[dict] = []
        seen: set[str] = set()
        for i, seq in enumerate(sequences):
            text = self.whisper_processor.decode(
                seq, skip_special_tokens=True
            ).strip().lower()
            if text in seen:
                continue
            seen.add(text)
            hypotheses.append({"text": text, "score": float(seq_scores[i])})

        hypotheses.sort(key=lambda h: h["score"], reverse=True)
        return hypotheses

    # ── CLIP similarity ──────────────────────────────────────────────

    def clip_similarity(
        self, image_embedding: np.ndarray, texts: list[str]
    ) -> np.ndarray:
        """Cosine similarities between one image embedding and N texts."""
        if not texts:
            return np.array([])
        text_embeddings = encode_texts(
            self.clip_model, self.clip_processor, texts, self.device
        )
        sims = (image_embedding.reshape(1, -1) @ text_embeddings.T).squeeze()
        return np.atleast_1d(sims)

    # ── rescoring ────────────────────────────────────────────────────

    def rescore(
        self,
        hypotheses: list[dict],
        image_embedding: np.ndarray,
        alpha: float | None = None,
        caption_style: bool = True,
    ) -> list[dict]:
        """Fuse ASR log-probs with CLIP cosine similarity.

        final = (1 - α) · softmax(asr_scores) + α · softmax(clip_scores · τ)
        where τ is CLIP's logit-scale temperature (~100).
        """
        if alpha is None:
            alpha = self.alpha
        if not hypotheses:
            return hypotheses

        texts = [h["text"] for h in hypotheses]
        clip_texts = [
            to_caption_style(t) if caption_style else normalize_transcript(t)
            for t in texts
        ]

        clip_scores = self.clip_similarity(image_embedding, clip_texts)

        asr_logits = np.array([h["score"] for h in hypotheses])
        asr_probs = _softmax(asr_logits)

        # CLIP logit-scale temperature (learned ~100 in ViT-B/32)
        clip_probs = _softmax(clip_scores * 100.0)

        fused = (1 - alpha) * asr_probs + alpha * clip_probs

        rescored = [
            {
                "text": h["text"],
                "asr_score": float(asr_probs[i]),
                "clip_score": float(clip_scores[i]),
                "fused_score": float(fused[i]),
            }
            for i, h in enumerate(hypotheses)
        ]
        rescored.sort(key=lambda h: h["fused_score"], reverse=True)
        return rescored

    # ── main entry point ─────────────────────────────────────────────

    def transcribe(
        self,
        audio_array: np.ndarray,
        image_id: str | None = None,
        image_path: str | Path | None = None,
        sr: int = 16000,
        alpha: float | None = None,
    ) -> dict:
        """Full transcription pipeline.

        Supply *image_id* (cached) or *image_path* (computed on the fly) for
        multimodal rescoring; omit both for ASR-only output.
        """
        hypotheses = self.generate_nbest(audio_array, sr)

        if image_id is None and image_path is None:
            return {
                "transcription": hypotheses[0]["text"] if hypotheses else "",
                "hypotheses": hypotheses,
                "mode": "asr_only",
            }

        img_emb = self._resolve_image_embedding(image_id, image_path)
        if img_emb is None:
            return {
                "transcription": hypotheses[0]["text"] if hypotheses else "",
                "hypotheses": hypotheses,
                "mode": "asr_only",
                "warning": f"Image not found (id={image_id}, path={image_path})",
            }

        rescored = self.rescore(hypotheses, img_emb, alpha)
        return {
            "transcription": rescored[0]["text"] if rescored else "",
            "hypotheses": rescored,
            "mode": "multimodal",
            "alpha": alpha if alpha is not None else self.alpha,
        }

    # ── helpers ───────────────────────────────────────────────────────

    def _resolve_image_embedding(
        self, image_id: str | None, image_path: str | Path | None
    ) -> np.ndarray | None:
        if image_id and image_id in self.image_embeddings:
            return self.image_embeddings[image_id]
        if image_path:
            image_path = Path(image_path)
            if not image_path.exists():
                return None
            image = Image.open(image_path).convert("RGB")
            inputs = self.clip_processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            with torch.no_grad():
                emb = _get_image_features(self.clip_model, pixel_values)
            return emb.cpu().numpy().squeeze()
        return None


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Multimodal ASR: Whisper + CLIP")
    parser.add_argument("--audio", type=Path, required=True, help="Path to .wav file")
    parser.add_argument("--image", type=Path, default=None, help="Path to image (PNG)")
    parser.add_argument("--image-id", default=None, help="Cached image id (e.g. img_001.png)")
    parser.add_argument("--alpha", type=float, default=0.3, help="Fusion coefficient")
    parser.add_argument("--beams", type=int, default=5, help="Beam search width")
    parser.add_argument("--whisper", default=DEFAULT_WHISPER_MODEL, help="Whisper model id")
    args = parser.parse_args()

    import soundfile as sf

    audio_array, sr = sf.read(str(args.audio))
    print(f"Audio: {args.audio}  ({len(audio_array)/sr:.1f}s @ {sr} Hz)")

    pipeline = MultimodalASR(
        whisper_model_id=args.whisper,
        alpha=args.alpha,
        num_beams=args.beams,
    )

    result = pipeline.transcribe(
        audio_array,
        image_id=args.image_id,
        image_path=str(args.image) if args.image else None,
        sr=sr,
    )

    print(f"\nMode: {result['mode']}")
    print(f"Transcription: {result['transcription']}")
    print("\nAll hypotheses:")
    for h in result["hypotheses"]:
        line = f"  {h['text']}"
        if "fused_score" in h:
            line += f"  (fused={h['fused_score']:.4f}  asr={h['asr_score']:.4f}  clip={h['clip_score']:.4f})"
        else:
            line += f"  (score={h['score']:.4f})"
        print(line)


if __name__ == "__main__":
    main()
