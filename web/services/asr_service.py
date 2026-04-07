"""
ASR Service — Calls the Modal-hosted Whisper+LoRA+CLIP endpoint.

In production (MODAL_ASR_URL set), sends audio to the remote GPU endpoint.
In development (no URL set), falls back to loading models locally.
"""

import os
import io
import re
import numpy as np
import requests
import soundfile as sf
from pathlib import Path

WEB_ROOT = Path(__file__).resolve().parent.parent

MODAL_ASR_URL = os.environ.get("MODAL_ASR_URL")


class ASRService:
    """Whisper + LoRA ASR with optional CLIP visual-context rescoring."""

    def __init__(self):
        self._local_model = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return

        if MODAL_ASR_URL:
            print(f"[ASR] Using remote Modal endpoint: {MODAL_ASR_URL}")
            self._loaded = True
            return

        print("[ASR] No MODAL_ASR_URL set — loading models locally...")
        self._load_local()
        self._loaded = True

    def _load_local(self):
        import torch
        from transformers import (
            WhisperProcessor, WhisperForConditionalGeneration,
            CLIPModel, CLIPProcessor,
        )
        from peft import PeftModel

        PROJECT_ROOT = WEB_ROOT.parent
        DEFAULT_WHISPER_MODEL = "openai/whisper-small"
        DEFAULT_LORA_PATH = PROJECT_ROOT / "asr" / "finetuned-whsiper-lora"
        DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
        DEFAULT_CACHE_PATH = WEB_ROOT / "data" / "cache" / "clip_image_embeddings.npz"

        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self._device = device
        print(f"[ASR] Loading models on {device}...")

        self._whisper_processor = WhisperProcessor.from_pretrained(DEFAULT_WHISPER_MODEL)
        base = WhisperForConditionalGeneration.from_pretrained(DEFAULT_WHISPER_MODEL)

        if DEFAULT_LORA_PATH.exists():
            print(f"[ASR] Applying LoRA from {DEFAULT_LORA_PATH}")
            base = PeftModel.from_pretrained(base, str(DEFAULT_LORA_PATH))
            base = base.merge_and_unload()

        self._whisper_model = base.to(device)
        self._whisper_model.eval()
        self._whisper_model.generation_config.forced_decoder_ids = None
        self._whisper_model.generation_config.suppress_tokens = []

        self._clip_processor = CLIPProcessor.from_pretrained(DEFAULT_CLIP_MODEL)
        self._clip_model = CLIPModel.from_pretrained(DEFAULT_CLIP_MODEL).to(device)
        self._clip_model.eval()

        self._image_embeddings: dict[str, np.ndarray] = {}
        if DEFAULT_CACHE_PATH.exists():
            data = np.load(str(DEFAULT_CACHE_PATH))
            self._image_embeddings = dict(data)
            print(f"[ASR] Loaded {len(self._image_embeddings)} cached image embeddings")

        self._local_model = True
        print("[ASR] Models ready.")

    def transcribe(self, audio_path: str | Path, image_id: str | None = None,
                   alpha: float = 0.3, num_beams: int = 5) -> dict:
        if MODAL_ASR_URL:
            return self._transcribe_remote(audio_path, image_id, alpha)
        return self._transcribe_local(audio_path, image_id, alpha, num_beams)

    def _transcribe_remote(self, audio_path: str | Path, image_id: str | None,
                           alpha: float) -> dict:
        with open(str(audio_path), "rb") as f:
            audio_bytes = f.read()

        files = {"audio": ("recording.wav", audio_bytes)}
        data = {"alpha": str(alpha)}
        if image_id:
            data["image_id"] = image_id

        resp = requests.post(MODAL_ASR_URL, files=files, data=data, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def _transcribe_local(self, audio_path: str | Path, image_id: str | None,
                          alpha: float, num_beams: int) -> dict:
        import torch

        audio_array, sr = sf.read(str(audio_path))

        input_features = self._whisper_processor(
            audio_array, sampling_rate=sr, return_tensors="pt"
        ).input_features.to(self._device)

        with torch.no_grad():
            outputs = self._whisper_model.generate(
                input_features,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=100,
            )

        hypotheses = []
        seen = set()
        for i, seq in enumerate(outputs.sequences):
            text = self._whisper_processor.decode(seq, skip_special_tokens=True).strip().lower()
            if text not in seen:
                seen.add(text)
                hypotheses.append({
                    "text": text,
                    "score": float(outputs.sequences_scores[i].cpu()),
                })
        hypotheses.sort(key=lambda h: h["score"], reverse=True)

        if not hypotheses:
            return {"transcription": "", "hypotheses": [], "mode": "asr_only"}

        if image_id is None or image_id not in self._image_embeddings:
            return {
                "transcription": hypotheses[0]["text"],
                "hypotheses": hypotheses,
                "mode": "asr_only",
            }

        img_emb = self._image_embeddings[image_id]
        rescored = self._rescore(hypotheses, img_emb, alpha)
        return {
            "transcription": rescored[0]["text"],
            "hypotheses": rescored,
            "mode": "multimodal",
            "alpha": alpha,
        }

    def _rescore(self, hypotheses, img_emb, alpha):
        import torch

        def normalize(text):
            text = text.strip().lower()
            if not text:
                return text
            text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text)
            fillers = {"um", "uh", "uh-huh", "hmm", "hm", "ah", "er", "oh",
                       "like", "okay", "ok", "so", "well"}
            words = [w for w in text.split() if w not in fillers]
            text = " ".join(words).strip()
            if len(text.split()) <= 2:
                return f"an image showing {text}"
            return text

        def softmax(x):
            e = np.exp(x - x.max())
            return e / e.sum()

        texts = [normalize(h["text"]) for h in hypotheses]
        inputs = self._clip_processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            text_out = self._clip_model.text_model(
                input_ids=inputs["input_ids"].to(self._device),
                attention_mask=inputs["attention_mask"].to(self._device),
            )
            text_features = self._clip_model.text_projection(text_out.pooler_output)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        text_np = text_features.cpu().numpy()
        clip_sims = np.atleast_1d((img_emb.reshape(1, -1) @ text_np.T).squeeze())

        asr_logits = np.array([h["score"] for h in hypotheses])
        asr_probs = softmax(asr_logits)
        clip_probs = softmax(clip_sims * 100.0)
        fused = (1 - alpha) * asr_probs + alpha * clip_probs

        rescored = []
        for i, h in enumerate(hypotheses):
            rescored.append({
                "text": h["text"],
                "asr_score": float(asr_probs[i]),
                "clip_score": float(clip_sims[i]),
                "fused_score": float(fused[i]),
            })
        rescored.sort(key=lambda h: h["fused_score"], reverse=True)
        return rescored
