#!/usr/bin/env python3


from __future__ import annotations

import json
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_ROOT = "stereomusicqa_v0.2"

app = modal.App("binauralcot-qwen2audio")
volume = modal.Volume.from_name("binauralcot-data", create_if_missing=True)

HF_CACHE_DIR = "/data/hf_cache"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1")
    .pip_install(
        "numpy>=1.24",
        "scipy>=1.11",
        "librosa>=0.10",
        "soundfile>=0.12",
        "torch>=2.2",
        "transformers>=4.45",
        "accelerate>=0.30",
        "peft>=0.11",
        "bitsandbytes>=0.43",
    )
    .add_local_dir(
        REPO_ROOT,
        remote_path="/app",
        ignore=[
            "**/__pycache__",
            "**/.pytest_cache",
            ".venv",
            "data",
            "artifacts",
            ".git",
        ],
    )
)

INSTRUMENT_CODE_FOR_NAME = {
    "violin": "vn",
    "viola": "va",
    "cello": "vc",
    "double bass": "db",
    "flute": "fl",
    "oboe": "ob",
    "clarinet": "cl",
    "saxophone": "sax",
    "bassoon": "bn",
    "trumpet": "tpt",
    "horn": "hn",
    "trombone": "tbn",
    "tuba": "tba",
}


CHAIN_OF_THOUGHT_SUFFIX = (
    " Work through the conversion step by step before answering: (1) r = "
    "10^(level difference in dB / 20); (2) d = (r - 1) / (r + 1); (3) "
    "azimuth = arctan(d * tan 30 degrees), in degrees. Show each "
    "intermediate value, then state the final azimuth as a single number."
)


@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": volume},
    timeout=13 * 60 * 60,
)
def generate(
    items: list[dict],
    answers_path: str,
    chain_of_thought: bool = False,
    audio_only: bool = False,
    adapter: str | None = None,
    dither: bool = False,
    dither_seed: int | None = None,
) -> int:
    import os
    import sys
    import time

    os.environ["HF_HOME"] = HF_CACHE_DIR
    sys.path.insert(0, "/app")

    import numpy as np
    import soundfile as sf

    from benchmark.measure import measure_render
    from model.qwen_backbone import (
        DEFAULT_MAX_NEW_TOKENS,
        answer,
        answer_from_stereo,
        describe_cues,
        load_backbone,
    )

    dither_rng = np.random.default_rng(dither_seed) if dither else None

    full_answers_path = f"/data/{answers_path}"
    os.makedirs(os.path.dirname(full_answers_path), exist_ok=True)
    done: set[str] = set()
    if os.path.exists(full_answers_path):
        with open(full_answers_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    done.add(json.loads(line)["item_id"])
    print(f"{len(done)} items already answered; {len(items) - len(done)} remaining")

    started = time.time()
    if adapter is not None:
        print(f"loading Qwen2-Audio 4-bit + adapter {adapter}...")
        backbone = load_backbone(
            device="cuda",
            dtype="bfloat16",
            quantization="4bit",
            adapter_dir=f"/data/{adapter}",
        )
    else:
        print("loading Qwen2-Audio-7B-Instruct (bf16, cuda)...")
        backbone = load_backbone(device="cuda", dtype="bfloat16")
    print(f"loaded in {time.time() - started:.1f}s")

    new_answers = 0
    with open(full_answers_path, "a", encoding="utf-8") as out:
        for index, item in enumerate(items):
            if item["item_id"] in done:
                continue
            audio_path = f"/data/{item['audio_path']}"
            stereo, sample_rate = sf.read(
                audio_path, dtype="float64", always_2d=True
            )

            item_started = time.time()
            if audio_only:
                reply = answer_from_stereo(
                    backbone,
                    stereo,
                    sample_rate,
                    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                    dither=dither,
                    dither_rng=dither_rng,
                )
            else:
                instrument = item["target_instrument"]
                instrument_code = INSTRUMENT_CODE_FOR_NAME.get(instrument)
                evidence = measure_render(stereo, sample_rate, instrument_code)
                f0_hz = evidence.f0_hz if evidence.f0_hz == evidence.f0_hz else None
                cue_text = describe_cues(evidence, instrument=instrument, f0_hz=f0_hz)

                question = item["question"]
                if chain_of_thought:
                    question += CHAIN_OF_THOUGHT_SUFFIX
                reply = answer(
                    backbone,
                    stereo,
                    sample_rate,
                    cue_text,
                    question=question,
                    max_new_tokens=400 if chain_of_thought else DEFAULT_MAX_NEW_TOKENS,
                )
            elapsed = time.time() - item_started
            print(
                f"[{index + 1}/{len(items)}] {item['item_id']} "
                f"({elapsed:.1f}s): {reply[:80]!r}"
            )

            record = {"item_id": item["item_id"], "answer": reply}


            if item.get("build_id") is not None:
                record["build_id"] = item["build_id"]
            out.write(json.dumps(record) + "\n")
            out.flush()
            new_answers += 1
            if new_answers % 10 == 0:
                volume.commit()

    volume.commit()
    return new_answers


def _read_test_items(test_public: Path, limit: int | None) -> list[dict]:
    items = []
    with test_public.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                items.append(json.loads(line))
    if limit is not None:
        import random

        shuffled = items[:]
        random.Random(0).shuffle(shuffled)
        items = shuffled[:limit]
    return items


@app.local_entrypoint()
def main(
    limit: int | None = None,
    upload_only: bool = False,
    chain_of_thought: bool = False,
    audio_only: bool = False,
    adapter: str | None = None,
    answers_path: str | None = None,
    dither: bool = False,
    dither_seed: int | None = None,
    artifacts_root: str = DEFAULT_ARTIFACTS_ROOT,
):
    is_default_root = artifacts_root == DEFAULT_ARTIFACTS_ROOT
    root_path = REPO_ROOT / "artifacts" / artifacts_root
    test_public = root_path / "test_public.jsonl"


    remote_audio_dir = "audio/test" if is_default_root else f"audio/{artifacts_root}_test"

    if answers_path is None:
        suffix = "_finetuned" if adapter else ""
        suffix += "_dithered" if dither else ""
        suffix += f"_seed{dither_seed}" if dither and dither_seed is not None else ""
        suffix += "" if is_default_root else f"_{artifacts_root}"
        if audio_only:
            answers_path = f"answers/test_answers_audio_only{suffix}.jsonl"
        elif chain_of_thought:
            answers_path = f"answers/test_answers_cot{suffix}.jsonl"
        else:
            answers_path = f"answers/test_answers{suffix}.jsonl"
    if not test_public.is_file():
        raise SystemExit(
            f"no {test_public} -- build the benchmark first "
            "(scripts/build_benchmark.py)"
        )

    print(f"uploading {artifacts_root}'s held-out test audio to {remote_audio_dir}/ "
          "(skips files already there)...")
    audio_dir = root_path / "audio" / "test"
    try:
        existing = {
            entry.path.lstrip("/") for entry in volume.listdir(remote_audio_dir)
        }
    except (FileNotFoundError, modal.exception.NotFoundError):
        existing = set()
    to_upload = [
        wav for wav in sorted(audio_dir.glob("*.wav"))
        if f"{remote_audio_dir}/{wav.name}" not in existing
    ]
    if to_upload:
        with volume.batch_upload(force=False) as batch:
            for wav in to_upload:
                batch.put_file(wav, f"{remote_audio_dir}/{wav.name}")
    print(f"upload done: {len(to_upload)} new, {len(existing)} already present.")

    if upload_only:
        return

    items = _read_test_items(test_public, limit)
    if not is_default_root:
        for item in items:
            item["audio_path"] = item["audio_path"].replace(
                "audio/test/", f"{remote_audio_dir}/", 1
            )
    print(
        f"generating for {len(items)} item(s) (artifacts_root={artifacts_root}, "
        f"limit={limit}, chain_of_thought={chain_of_thought}, audio_only={audio_only}, "
        f"adapter={adapter}, dither={dither}, dither_seed={dither_seed})..."
    )
    new_answers = generate.remote(
        items, answers_path, chain_of_thought, audio_only, adapter, dither, dither_seed
    )
    print(f"wrote {new_answers} new answers to volume path {answers_path}")
    print(
        "pull them down with:\n"
        f"  modal volume get binauralcot-data {answers_path} ."
    )
