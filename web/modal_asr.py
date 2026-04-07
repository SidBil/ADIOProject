"""
Modal serverless GPU app — Whisper+LoRA+CLIP transcription endpoint.

Deploy:  modal deploy web/modal_asr.py
Test:    modal serve web/modal_asr.py   (hot-reload dev mode)
"""

import modal
import fastapi

WHISPER_MODEL = "openai/whisper-small"
LORA_REPO = "zorbbbb/whisper-small-lora-torgo"
CLIP_MODEL = "openai/clip-vit-base-patch32"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch", "torchaudio",
        "transformers==4.51.3", "peft", "accelerate",
        "soundfile", "numpy", "pillow",
        "fastapi[standard]",
    )
    .add_local_file(
        "web/data/cache/clip_image_embeddings.npz",
        "/cache/clip_image_embeddings.npz",
        copy=True,
    )
    .run_commands(
        "python -c \"from transformers import WhisperProcessor, WhisperForConditionalGeneration; "
        f"WhisperProcessor.from_pretrained('{WHISPER_MODEL}'); "
        f"WhisperForConditionalGeneration.from_pretrained('{WHISPER_MODEL}')\"",

        "python -c \"from transformers import CLIPModel, CLIPProcessor; "
        f"CLIPProcessor.from_pretrained('{CLIP_MODEL}'); "
        f"CLIPModel.from_pretrained('{CLIP_MODEL}')\"",

        "python -c \"from peft import PeftModel; "
        "from transformers import WhisperForConditionalGeneration; "
        f"base = WhisperForConditionalGeneration.from_pretrained('{WHISPER_MODEL}'); "
        f"PeftModel.from_pretrained(base, '{LORA_REPO}')\"",
    )
)

app = modal.App("adio-asr", image=image)


@app.cls(
    gpu="T4",
    scaledown_window=300,
)
class ASRModel:
    @modal.enter()
    def load_models(self):
        import torch
        import numpy as np
        from transformers import (
            WhisperProcessor, WhisperForConditionalGeneration,
            CLIPModel, CLIPProcessor,
        )
        from peft import PeftModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.whisper_processor = WhisperProcessor.from_pretrained(WHISPER_MODEL)
        base = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL)
        base = PeftModel.from_pretrained(base, LORA_REPO)
        base = base.merge_and_unload()
        self.whisper_model = base.to(self.device)
        self.whisper_model.eval()
        self.whisper_model.generation_config.forced_decoder_ids = None
        self.whisper_model.generation_config.suppress_tokens = []

        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL).to(self.device)
        self.clip_model.eval()

        cache_path = "/cache/clip_image_embeddings.npz"
        data = np.load(cache_path)
        self.image_embeddings = dict(data)
        print(f"[Modal ASR] Ready — {len(self.image_embeddings)} image embeddings loaded")

    @modal.method()
    def transcribe(self, audio_bytes: bytes, image_id: str | None = None,
                   alpha: float = 0.3, num_beams: int = 5) -> dict:
        import re
        import io
        import subprocess
        import tempfile
        import torch
        import numpy as np
        import soundfile as sf

        try:
            audio_array, sr = sf.read(io.BytesIO(audio_bytes))
        except Exception:
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_in:
                tmp_in.write(audio_bytes)
                tmp_in_path = tmp_in.name
            tmp_out_path = tmp_in_path.replace(".webm", ".wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_in_path, "-ar", "16000", "-ac", "1", tmp_out_path],
                capture_output=True, check=True,
            )
            audio_array, sr = sf.read(tmp_out_path)
            import os
            os.unlink(tmp_in_path)
            os.unlink(tmp_out_path)

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

    def _rescore(self, hypotheses, img_emb, alpha):
        import re
        import torch
        import numpy as np

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


@app.function(gpu="T4", scaledown_window=300, image=image)
@modal.fastapi_endpoint(method="POST")
async def transcribe_endpoint(
    audio: fastapi.UploadFile = fastapi.File(...),
    image_id: str = fastapi.Form(default=None),
    alpha: float = fastapi.Form(default=0.3),
):
    audio_bytes = await audio.read()
    model = ASRModel()
    result = model.transcribe.remote(audio_bytes, image_id=image_id, alpha=alpha)
    return result
