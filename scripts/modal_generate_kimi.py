from __future__ import annotations

import json
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = REPO_ROOT / "artifacts" / "stereomusicqa_v0.2"
TEST_PUBLIC = ARTIFACTS_ROOT / "test_public.jsonl"

app = modal.App("binauralcot-kimiaudio")
volume = modal.Volume.from_name("binauralcot-data", create_if_missing=True)

HF_CACHE_DIR = "/data/hf_cache"

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("libsndfile1", "git")
    .pip_install(
        "numpy>=1.24",
        "scipy>=1.11",
        "librosa>=0.10",
        "soundfile>=0.12",
        "torch==2.4.0",
        "torchaudio==2.4.0",
    )
    .pip_install("packaging", "ninja", "wheel", "setuptools")
    .pip_install("transformers==4.46.0", "accelerate>=0.30")
    .pip_install(
        "git+https://github.com/MoonshotAI/Kimi-Audio.git",
        extra_options="--no-build-isolation",
        gpu="L4",
    )
    .pip_install("transformers==4.46.0")
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


@app.function(
    image=image,
    gpu="L4",
    volumes={"/data": volume},
    timeout=13 * 60 * 60,
)
def generate(items: list[dict], answers_path: str, audio_only: bool = False) -> int:
    import os
    import sys
    import time

    os.environ["HF_HOME"] = HF_CACHE_DIR
    sys.path.insert(0, "/app")

    import soundfile as sf

    from benchmark.measure import measure_render
    from model.kimi_backbone import answer, answer_from_stereo, load_backbone
    from model.qwen_backbone import describe_cues

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
    print("loading Kimi-Audio-7B-Instruct...")
    backbone = load_backbone()
    print(f"loaded in {time.time() - started:.1f}s")

    new_answers = 0
    with open(full_answers_path, "a", encoding="utf-8") as out:
        for index, item in enumerate(items):
            if item["item_id"] in done:
                continue
            audio_path = f"/data/{item['audio_path']}"
            stereo, sample_rate = sf.read(audio_path, dtype="float64", always_2d=True)

            item_started = time.time()
            if audio_only:
                reply = answer_from_stereo(backbone, stereo, sample_rate)
            else:
                instrument = item["target_instrument"]
                instrument_code = INSTRUMENT_CODE_FOR_NAME.get(instrument)
                evidence = measure_render(stereo, sample_rate, instrument_code)
                f0_hz = evidence.f0_hz if evidence.f0_hz == evidence.f0_hz else None
                cue_text = describe_cues(evidence, instrument=instrument, f0_hz=f0_hz)
                reply = answer(backbone, stereo, sample_rate, cue_text, question=item["question"])
            elapsed = time.time() - item_started
            print(
                f"[{index + 1}/{len(items)}] {item['item_id']} "
                f"({elapsed:.1f}s): {reply[:80]!r}"
            )

            out.write(json.dumps({"item_id": item["item_id"], "answer": reply}) + "\n")
            out.flush()
            new_answers += 1
            if new_answers % 10 == 0:
                volume.commit()

    volume.commit()
    return new_answers


def _read_test_items(limit: int | None) -> list[dict]:
    items = []
    with TEST_PUBLIC.open("r", encoding="utf-8") as file:
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
    audio_only: bool = False,
    answers_path: str | None = None,
):
    if answers_path is None:
        answers_path = (
            "answers/test_answers_kimi_audio_only.jsonl"
            if audio_only
            else "answers/test_answers_kimi_cues.jsonl"
        )
    if not TEST_PUBLIC.is_file():
        raise SystemExit(
            f"no {TEST_PUBLIC} -- build the benchmark first (scripts/build_benchmark.py)"
        )

    print("uploading held-out test audio to the volume (skips files already there)...")
    audio_dir = ARTIFACTS_ROOT / "audio" / "test"
    try:
        existing = {entry.path.lstrip("/") for entry in volume.listdir("audio/test")}
    except FileNotFoundError:
        existing = set()
    to_upload = [
        wav for wav in sorted(audio_dir.glob("*.wav")) if f"audio/test/{wav.name}" not in existing
    ]
    if to_upload:
        with volume.batch_upload(force=False) as batch:
            for wav in to_upload:
                batch.put_file(wav, f"audio/test/{wav.name}")
    print(f"upload done: {len(to_upload)} new, {len(existing)} already present.")

    if upload_only:
        return

    items = _read_test_items(limit)
    print(f"running {len(items)} items (audio_only={audio_only}) -> {answers_path}")
    new_answers = generate.remote(items, answers_path, audio_only)
    print(f"wrote {new_answers} new answers")
    print("pull results with:")
    print(f"  modal volume get binauralcot-data {answers_path} .")
