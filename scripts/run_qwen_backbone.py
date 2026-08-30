#!/usr/bin/env python3


from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.measure import measure_render  # noqa: E402
from model.qwen_backbone import (  # noqa: E402
    ANALYSIS_SAMPLE_RATE,
    BACKBONE_MODEL_ID,
    BACKBONE_SAMPLE_RATE,
    DEFAULT_DISCLOSURE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_QUESTION,
    answer,
    build_prompt,
    describe_cues,
    load_backbone,
    prepare_audio,
    probe_channel_handling,
)


def read_stereo(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float64", always_2d=True)
    if audio.shape[1] != 2:
        raise ValueError(f"{path} has {audio.shape[1]} channels; expected stereo")
    return audio, sample_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "audio", type=Path, nargs="?", help="stereo WAV file to ask about"
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="only report whether the processor keeps the second channel",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="measure cues and print the prompt without loading the model",
    )
    parser.add_argument(
        "-q", "--question", default=DEFAULT_QUESTION, help="question to ask"
    )
    parser.add_argument("--instrument", default=None, help="instrument name for the prompt")
    parser.add_argument(
        "--instrument-code",
        default=None,
        metavar="CODE",
        help=(
            "URMP instrument code (vn, va, vc, db, fl, ...; see "
            "cues/pitch.py's INSTRUMENT_RANGE_HZ) to track pitch against. "
            "Without this, f0 is not measured and does not reach the prompt."
        ),
    )
    parser.add_argument(
        "--f0-hz",
        type=float,
        default=None,
        help="override the measured median f0 (default: use what --instrument-code measures)",
    )
    parser.add_argument(
        "--disclosure",
        choices=("evidence", "conclusions", "no_numbers"),
        default=DEFAULT_DISCLOSURE,
        help=(
            "how much the prompt gives away: evidence (default, the "
            "measurements and the panning law, model derives the regime "
            "and the angle), conclusions (the original Part 6 prompt, "
            "which states both), no_numbers (ablation, directions in "
            "words with every scalar removed)"
        ),
    )
    parser.add_argument("--model-id", default=BACKBONE_MODEL_ID)
    parser.add_argument(
        "--device", default="cpu", help="torch device for the model (default: cpu)"
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--quantization",
        default=None,
        choices=["4bit"],
        help=(
            "load via bitsandbytes 4-bit (NF4) instead of full precision -- "
            "needed to fit the ~16.8 GB model on a 16 GB GPU (e.g. a free "
            "Colab T4); omit on a GPU with enough headroom for full precision"
        ),
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS
    )
    args = parser.parse_args()
    if args.audio is None and not args.probe:
        parser.error("an audio file is required unless --probe is given")
    return args


def main() -> None:
    args = parse_args()

    if args.probe:
        from transformers import Qwen2AudioProcessor

        processor = Qwen2AudioProcessor.from_pretrained(args.model_id)
        handling = probe_channel_handling(processor)
        print(handling.summary())
        print(f"  mono input  -> input_features {handling.mono_feature_shape}")
        print(f"  stereo input -> input_features {handling.stereo_feature_shape}")
        print(f"  keeps_stereo = {handling.keeps_stereo}")
        print(
            "  channels_became_batch_entries = "
            f"{handling.channels_became_batch_entries}"
        )
        if args.audio is None:
            return

    stereo, sample_rate = read_stereo(args.audio)
    if sample_rate != ANALYSIS_SAMPLE_RATE:
        print(
            f"note: {args.audio} is {sample_rate} Hz; cues are measured at this "
            f"rate, and {ANALYSIS_SAMPLE_RATE} Hz is what the delay tolerance "
            "assumes"
        )

    evidence = measure_render(stereo, sample_rate, args.instrument_code)


    f0_hz = args.f0_hz
    if f0_hz is None and evidence.f0_hz == evidence.f0_hz:
        f0_hz = evidence.f0_hz
    cue_text = describe_cues(
        evidence,
        instrument=args.instrument,
        f0_hz=f0_hz,
        disclosure=args.disclosure,
    )
    print(f"\nmeasured cues for {args.audio}:\n  {cue_text}\n")

    mono = prepare_audio(stereo, sample_rate)
    print(
        f"audio for the backbone: {mono.size} samples at {BACKBONE_SAMPLE_RATE} Hz "
        f"({mono.size / BACKBONE_SAMPLE_RATE:.2f} s), downmixed from "
        f"{stereo.shape[1]} channels at {sample_rate} Hz"
    )

    if args.dry_run:
        from transformers import Qwen2AudioProcessor

        processor = Qwen2AudioProcessor.from_pretrained(args.model_id)
        print("\nprompt:\n" + build_prompt(processor, cue_text, args.question))
        return

    print(
        f"\nloading {args.model_id} on {args.device} ({args.dtype}"
        f"{', 4bit' if args.quantization else ''})..."
    )
    started = time.time()
    backbone = load_backbone(
        args.model_id, device=args.device, dtype=args.dtype, quantization=args.quantization
    )
    print(f"loaded in {time.time() - started:.1f} s")

    started = time.time()
    reply = answer(
        backbone,
        stereo,
        sample_rate,
        cue_text,
        question=args.question,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"generated in {time.time() - started:.1f} s\n")
    print(f"question: {args.question}")
    print(f"answer:   {reply}")


if __name__ == "__main__":
    main()
