"""Analyze where multimodal CLIP rescoring helps vs hurts ASR accuracy.

Compares ASR-only (α=0) against multimodal (α>0) at the sample level,
then slices results by speech status, WER bucket, and more.

Usage (from project root):
    python -m multimodal.analysis  # prints expected usage info
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from jiwer import wer as compute_wer, cer as compute_cer

from .multimodal_asr import MultimodalASR


# ── per-sample comparison ────────────────────────────────────────────


def compare_asr_vs_multimodal(
    pipeline: MultimodalASR,
    test_samples: list[dict],
    alpha: float = 0.3,
) -> dict:
    """Run ASR-only and multimodal on every sample, return full analysis.

    Each element of *test_samples* must contain:
        audio_array   : np.ndarray
        sr            : int        (default 16 000)
        reference     : str        (ground-truth transcription)
        image_id      : str | None
        image_path    : str | None
        speech_status : str | None (optional, e.g. "dysarthria" / "healthy")
    """
    per_sample: list[dict] = []

    for i, sample in enumerate(test_samples):
        ref = sample["reference"].strip().lower()
        sr = sample.get("sr", 16000)

        asr_result = pipeline.transcribe(audio_array=sample["audio_array"], sr=sr)
        asr_text = asr_result["transcription"]

        mm_result = pipeline.transcribe(
            audio_array=sample["audio_array"],
            image_id=sample.get("image_id"),
            image_path=sample.get("image_path"),
            sr=sr,
            alpha=alpha,
        )
        mm_text = mm_result["transcription"]

        asr_wer = compute_wer(ref, asr_text) if ref else 0.0
        mm_wer = compute_wer(ref, mm_text) if ref else 0.0

        per_sample.append({
            "index": i,
            "reference": ref,
            "asr_hypothesis": asr_text,
            "multimodal_hypothesis": mm_text,
            "asr_wer": asr_wer,
            "multimodal_wer": mm_wer,
            "wer_delta": mm_wer - asr_wer,
            "improved": mm_wer < asr_wer,
            "degraded": mm_wer > asr_wer,
            "unchanged": abs(mm_wer - asr_wer) < 1e-9,
            "speech_status": sample.get("speech_status", "unknown"),
        })

        if (i + 1) % 10 == 0:
            print(f"  Compared {i + 1}/{len(test_samples)} samples …")

    return _aggregate(per_sample)


# ── aggregation ──────────────────────────────────────────────────────


def _aggregate(per_sample: list[dict]) -> dict:
    n = len(per_sample)
    if n == 0:
        return {"error": "No results to analyse"}

    improved = sum(r["improved"] for r in per_sample)
    degraded = sum(r["degraded"] for r in per_sample)
    unchanged = sum(r["unchanged"] for r in per_sample)

    all_refs = [r["reference"] for r in per_sample]
    all_asr = [r["asr_hypothesis"] for r in per_sample]
    all_mm = [r["multimodal_hypothesis"] for r in per_sample]

    overall_asr_wer = compute_wer(all_refs, all_asr)
    overall_mm_wer = compute_wer(all_refs, all_mm)
    overall_asr_cer = compute_cer(all_refs, all_asr)
    overall_mm_cer = compute_cer(all_refs, all_mm)

    # ── group by speech status ───────────────────────────────────────
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in per_sample:
        groups[r["speech_status"]].append(r)

    group_stats: dict[str, dict] = {}
    for status, items in groups.items():
        g_refs = [r["reference"] for r in items]
        g_asr = [r["asr_hypothesis"] for r in items]
        g_mm = [r["multimodal_hypothesis"] for r in items]
        group_stats[status] = {
            "count": len(items),
            "asr_wer": compute_wer(g_refs, g_asr),
            "multimodal_wer": compute_wer(g_refs, g_mm),
            "asr_cer": compute_cer(g_refs, g_asr),
            "multimodal_cer": compute_cer(g_refs, g_mm),
            "improved": sum(r["improved"] for r in items),
            "degraded": sum(r["degraded"] for r in items),
        }

    # ── top movers ───────────────────────────────────────────────────
    sorted_by_delta = sorted(per_sample, key=lambda r: r["wer_delta"])
    top_improvements = sorted_by_delta[:5]
    top_degradations = [r for r in reversed(sorted_by_delta) if r["degraded"]][:5]

    return {
        "summary": {
            "total_samples": n,
            "improved": improved,
            "degraded": degraded,
            "unchanged": unchanged,
            "improvement_rate": improved / n,
            "overall_asr_wer": overall_asr_wer,
            "overall_multimodal_wer": overall_mm_wer,
            "overall_asr_cer": overall_asr_cer,
            "overall_multimodal_cer": overall_mm_cer,
            "wer_reduction": overall_asr_wer - overall_mm_wer,
            "relative_improvement": (
                (overall_asr_wer - overall_mm_wer) / overall_asr_wer
                if overall_asr_wer > 0 else 0.0
            ),
            "success_criterion_met": improved / n >= 0.5,
        },
        "group_stats": group_stats,
        "per_sample": per_sample,
        "top_improvements": top_improvements,
        "top_degradations": top_degradations,
    }


# ── pretty-print ─────────────────────────────────────────────────────


def print_analysis(analysis: dict):
    """Human-readable report."""
    s = analysis["summary"]

    print("=" * 70)
    print("MULTIMODAL RESCORING ANALYSIS")
    print("=" * 70)

    print(f"\n  Total samples:           {s['total_samples']}")
    print(f"  Improved by multimodal:  {s['improved']} ({s['improvement_rate'] * 100:.1f}%)")
    print(f"  Degraded:                {s['degraded']}")
    print(f"  Unchanged:               {s['unchanged']}")

    print(f"\n  Overall ASR WER:         {s['overall_asr_wer'] * 100:.1f}%")
    print(f"  Overall Multimodal WER:  {s['overall_multimodal_wer'] * 100:.1f}%")
    print(f"  WER reduction:           {s['wer_reduction'] * 100:.1f}% abs "
          f"({s['relative_improvement'] * 100:.1f}% rel)")

    criterion = "PASSED" if s["success_criterion_met"] else "FAILED"
    print(f"\n  Success criterion (≥50% helped): {criterion}")

    # ── group breakdown ──────────────────────────────────────────────
    if analysis["group_stats"]:
        print(f"\n{'  Per-Group Breakdown  ':=^70}")
        header = f"  {'Group':<14} {'N':>5} {'ASR WER':>9} {'MM WER':>9} {'Improved':>9} {'Degraded':>9}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for group, d in sorted(analysis["group_stats"].items()):
            print(
                f"  {group:<14} {d['count']:>5} "
                f"{d['asr_wer'] * 100:>8.1f}% "
                f"{d['multimodal_wer'] * 100:>8.1f}% "
                f"{d['improved']:>9} {d['degraded']:>9}"
            )

    # ── top improvements ─────────────────────────────────────────────
    print(f"\n{'  Top Improvements  ':=^70}")
    for r in analysis.get("top_improvements", [])[:5]:
        print(f"  [{r['index']:>4}] WER {r['asr_wer'] * 100:.0f}% → {r['multimodal_wer'] * 100:.0f}%"
              f"  ({r['speech_status']})")
        print(f"        ref: {r['reference']}")
        print(f"        asr: {r['asr_hypothesis']}")
        print(f"        mm:  {r['multimodal_hypothesis']}")

    # ── top degradations ─────────────────────────────────────────────
    if analysis.get("top_degradations"):
        print(f"\n{'  Top Degradations  ':=^70}")
        for r in analysis["top_degradations"][:5]:
            print(f"  [{r['index']:>4}] WER {r['asr_wer'] * 100:.0f}% → {r['multimodal_wer'] * 100:.0f}%"
                  f"  ({r['speech_status']})")
            print(f"        ref: {r['reference']}")
            print(f"        asr: {r['asr_hypothesis']}")
            print(f"        mm:  {r['multimodal_hypothesis']}")


# ── persistence ──────────────────────────────────────────────────────


def save_analysis(analysis: dict, output_path: Path | str):
    """Write analysis dict to JSON (handles numpy types)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    class _Enc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)

    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2, cls=_Enc)
    print(f"Analysis saved → {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    print("Multimodal analysis requires paired image-audio test data.")
    print("Call compare_asr_vs_multimodal() programmatically.\n")
    print("Example:")
    print("  from multimodal.multimodal_asr import MultimodalASR")
    print("  from multimodal.analysis import compare_asr_vs_multimodal, print_analysis")
    print()
    print("  pipeline = MultimodalASR(alpha=0.3)")
    print("  analysis = compare_asr_vs_multimodal(pipeline, test_samples, alpha=0.3)")
    print("  print_analysis(analysis)")


if __name__ == "__main__":
    main()
