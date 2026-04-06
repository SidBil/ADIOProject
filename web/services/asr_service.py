"""
ASR Service — Whisper + LoRA + CLIP multimodal transcription pipeline.

Loads the fine-tuned Whisper model once, then provides fast inference
for audio transcription with optional CLIP-based visual rescoring.
"""

import re
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration, CLIPModel, CLIPProcessor
from peft import PeftModel

WEB_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = WEB_ROOT.parent

DEFAULT_WHISPER_MODEL = "openai/whisper-small"
DEFAULT_LORA_PATH = PROJECT_ROOT / "asr" / "finetuned-whsiper-lora"
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_CACHE_PATH = WEB_ROOT / "data" / "cache" / "clip_image_embeddings.npz"


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _normalize_for_clip(text: str) -> str:
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


class ASRService:
    """Whisper + LoRA ASR with optional CLIP visual-context rescoring."""

    def __init__(self):
        self.device = _pick_device()
        self.whisper_model = None
        self.whisper_processor = None
        self.clip_model = None
        self.clip_processor = None
        self.image_embeddings: dict[str, np.ndarray] = {}
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        print(f"[ASR] Loading models on {self.device}...")

        self.whisper_processor = WhisperProcessor.from_pretrained(DEFAULT_WHISPER_MODEL)
        base = WhisperForConditionalGeneration.from_pretrained(DEFAULT_WHISPER_MODEL)

        if DEFAULT_LORA_PATH.exists():
            print(f"[ASR] Applying LoRA from {DEFAULT_LORA_PATH}")
            base = PeftModel.from_pretrained(base, str(DEFAULT_LORA_PATH))
            base = base.merge_and_unload()

        self.whisper_model = base.to(self.device)
        self.whisper_model.eval()
        self.whisper_model.generation_config.forced_decoder_ids = None
        self.whisper_model.generation_config.suppress_tokens = []

        self.clip_processor = CLIPProcessor.from_pretrained(DEFAULT_CLIP_MODEL)
        self.clip_model = CLIPModel.from_pretrained(DEFAULT_CLIP_MODEL).to(self.device)
        self.clip_model.eval()

        if DEFAULT_CACHE_PATH.exists():
            data = np.load(str(DEFAULT_CACHE_PATH))
            self.image_embeddings = dict(data)
            print(f"[ASR] Loaded {len(self.image_embeddings)} cached image embeddings")

        self._loaded = True
        print("[ASR] Models ready.")

    def transcribe(self, audio_path: str | Path, image_id: str | None = None,
                   alpha: float = 0.3, num_beams: int = 5) -> dict:
        audio_array, sr = sf.read(str(audio_path))

        input_features = self.whisper_processor(
            audio_array, sampling_rate=sr, return_tensors="pt"
        ).input_features.to(self.device)

        with torch.no_grad():
            outputs = self.whisper_model.generate(
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
            text = self.whisper_processor.decode(seq, skip_special_tokens=True).strip().lower()
            if text not in seen:
                seen.add(text)
                hypotheses.append({
                    "text": text,
                    "score": float(outputs.sequences_scores[i].cpu()),
                })
        hypotheses.sort(key=lambda h: h["score"], reverse=True)

        if not hypotheses:
            return {"transcription": "", "hypotheses": [], "mode": "asr_only"}

        if image_id is None or image_id not in self.image_embeddings:
            return {
                "transcription": hypotheses[0]["text"],
                "hypotheses": hypotheses,
                "mode": "asr_only",
            }

        img_emb = self.image_embeddings[image_id]
        rescored = self._rescore(hypotheses, img_emb, alpha)
        return {
            "transcription": rescored[0]["text"],
            "hypotheses": rescored,
            "mode": "multimodal",
            "alpha": alpha,
        }

    def _rescore(self, hypotheses: list[dict], img_emb: np.ndarray,
                 alpha: float) -> list[dict]:
        texts = [_normalize_for_clip(h["text"]) for h in hypotheses]

        inputs = self.clip_processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            text_out = self.clip_model.text_model(
                input_ids=inputs["input_ids"].to(self.device),
                attention_mask=inputs["attention_mask"].to(self.device),
            )
            text_features = self.clip_model.text_projection(text_out.pooler_output)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        text_np = text_features.cpu().numpy()
        clip_sims = np.atleast_1d((img_emb.reshape(1, -1) @ text_np.T).squeeze())

        asr_logits = np.array([h["score"] for h in hypotheses])
        asr_probs = _softmax(asr_logits)
        clip_probs = _softmax(clip_sims * 100.0)
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
