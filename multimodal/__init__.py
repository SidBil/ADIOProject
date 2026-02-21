"""Phase 3: Multimodal ASR — CLIP visual context rescoring for Whisper hypotheses.

Modules
-------
clip_embeddings           Pre-compute and cache CLIP image embeddings.
transcript_normalization  Clean ASR output for CLIP text encoding.
multimodal_asr            Full Whisper + CLIP rescoring pipeline.
fusion_tuning             Grid search for optimal fusion coefficient α.
analysis                  Compare ASR-only vs multimodal performance.
"""


def __getattr__(name: str):
    """Lazy imports so lightweight modules don't pull in torch eagerly."""
    if name in ("load_clip", "encode_images", "encode_texts"):
        from . import clip_embeddings
        return getattr(clip_embeddings, name)
    if name in ("normalize_transcript", "to_caption_style"):
        from . import transcript_normalization
        return getattr(transcript_normalization, name)
    if name == "MultimodalASR":
        from .multimodal_asr import MultimodalASR
        return MultimodalASR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "load_clip",
    "encode_images",
    "encode_texts",
    "normalize_transcript",
    "to_caption_style",
    "MultimodalASR",
]
