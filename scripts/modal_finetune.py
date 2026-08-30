#!/usr/bin/env python3


from __future__ import annotations

import json
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = REPO_ROOT / "artifacts" / "stereomusicqa_v0.2"

app = modal.App("binauralcot-finetune")
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
        ignore=["**/__pycache__", "**/.pytest_cache", ".venv", "data", "artifacts", ".git"],
    )
)


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/data": volume},
    timeout=12 * 60 * 60,
)
def train(
    examples: list[dict],
    condition: str,
    epochs: float = 1.0,
    learning_rate: float = 2e-4,
    grad_accum: int = 8,
    encoder: str = "qkv",
    dither: bool = False,
    dither_seed: int | None = None,
    dense_augment: int = 0,
) -> dict:
    import os
    import sys
    import time

    os.environ["HF_HOME"] = HF_CACHE_DIR
    sys.path.insert(0, "/app")

    import numpy as np
    import soundfile as sf
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        BitsAndBytesConfig,
        Qwen2AudioForConditionalGeneration,
        Qwen2AudioProcessor,
    )

    from model.finetune_config import (
        ENCODER_MODES,
        adapter_dir,
        directly_trained,
        format_trainable,
        trainable_by_component,
    )
    from model.qwen_backbone import (
        BACKBONE_MODEL_ID,
        BACKBONE_SAMPLE_RATE,
        dither_stereo_channels,
        downmix_to_mono,
        resample_for_backbone,
    )

    if dither_seed is not None:
        torch.manual_seed(dither_seed)
    dither_rng = np.random.default_rng(dither_seed) if dither else None

    processor = Qwen2AudioProcessor.from_pretrained(BACKBONE_MODEL_ID)

    if encoder not in ENCODER_MODES:
        raise ValueError(f"unknown encoder mode {encoder!r}; pick one of {ENCODER_MODES}")

    print(f"loading Qwen2-Audio 4-bit (encoder={encoder}, dither={dither})...")
    started = time.time()
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,


        llm_int8_skip_modules=list(directly_trained(encoder)) or None,
    )
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        BACKBONE_MODEL_ID, quantization_config=quant, device_map="cuda", low_cpu_mem_usage=True
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=ENCODER_MODES[encoder],
    )
    model = get_peft_model(model, lora)

    direct = directly_trained(encoder)
    if direct:
        for name, parameter in model.named_parameters():
            if any(component in name for component in direct):
                parameter.requires_grad_(True)

    print(format_trainable(trainable_by_component(model)))
    print(f"model ready in {time.time() - started:.1f}s")

    def encode(example: dict):
        stereo, sr = sf.read(f"/data/{example['audio_path']}", dtype="float64", always_2d=True)
        if condition == "audio_only":
            left = resample_for_backbone(stereo[:, 0], sr)
            right = resample_for_backbone(stereo[:, 1], sr)
            if dither:
                left, right = dither_stereo_channels(left, right, rng=dither_rng)
            audio = [left, right]
            audio_content = [
                {"type": "text", "text": "Audio 1 is the left channel. Audio 2 is the right channel."},
                {"type": "audio", "audio_url": "l.wav"},
                {"type": "audio", "audio_url": "r.wav"},
            ]
        else:
            mono = resample_for_backbone(downmix_to_mono(stereo), sr)
            audio = [mono]
            audio_content = [{"type": "audio", "audio_url": "a.wav"}]

        user_turn = {"role": "user", "content": audio_content + [{"type": "text", "text": example["prompt"]}]}
        full_text = processor.apply_chat_template(
            [user_turn, {"role": "assistant", "content": [{"type": "text", "text": example["target"]}]}],
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text = processor.apply_chat_template(
            [user_turn], tokenize=False, add_generation_prompt=True
        )
        full = processor(text=full_text, audio=audio, sampling_rate=BACKBONE_SAMPLE_RATE, return_tensors="pt")
        prompt_len = processor(
            text=prompt_text, audio=audio, sampling_rate=BACKBONE_SAMPLE_RATE, return_tensors="pt"
        )["input_ids"].shape[1]
        labels = full["input_ids"].clone()
        labels[:, :prompt_len] = -100
        full["labels"] = labels
        return {k: v.to("cuda") for k, v in full.items()}

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=learning_rate
    )
    model.train()

    total_steps = int(len(examples) * epochs)
    suffix = (
        ("_dithered" if dither else "")
        + (f"_seed{dither_seed}" if dither_seed is not None else "")
        + (f"_denseaug{dense_augment}" if dense_augment > 0 else "")
    )
    out_dir = f"/data/adapters/{adapter_dir(condition, encoder)}{suffix}"
    print(f"training {total_steps} steps over {len(examples)} examples (grad_accum={grad_accum})")
    running = 0.0
    optimizer.zero_grad()
    step = 0
    for epoch in range(int(np.ceil(epochs))):
        for example in examples:
            if step >= total_steps:
                break
            try:
                batch = encode(example)
            except Exception as error:
                print(f"skip {example['item_id']}: {error}")
                step += 1
                continue
            out = model(**batch)
            loss = out.loss / grad_accum
            loss.backward()
            running += float(out.loss)
            if (step + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
            if (step + 1) % 20 == 0:
                print(f"[{step + 1}/{total_steps}] loss {running / 20:.4f}")
                running = 0.0
            if (step + 1) % 500 == 0:
                model.save_pretrained(out_dir)
                volume.commit()
                print(f"  checkpointed adapter at step {step + 1}")
            step += 1

    model.save_pretrained(out_dir)
    if direct:
        import torch as _torch

        _torch.save(
            {
                name: parameter.detach().to("cpu")
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
                and any(component in name for component in direct)
            },
            f"{out_dir}/direct_weights.pt",
        )
    volume.commit()
    print(f"saved adapter to {out_dir}")
    return {
        "steps": step,
        "adapter_dir": out_dir,
        "trainable_condition": condition,
        "encoder": encoder,
        "dither": dither,
        "dither_seed": dither_seed,
        "dense_augment": dense_augment,
    }


def _upload_needed_audio(examples: list[dict]) -> None:
    try:
        existing = {e.path.lstrip("/") for e in volume.listdir("audio/train")}
    except Exception:
        existing = set()
    to_upload = [
        ex["audio_path"] for ex in examples if ex["audio_path"] not in existing
    ]
    to_upload = sorted(set(to_upload))
    if to_upload:
        with volume.batch_upload(force=False) as batch:
            for rel in to_upload:
                batch.put_file(ARTIFACTS_ROOT / rel, rel)
    print(f"train audio: {len(to_upload)} uploaded, {len(existing)} already present")


@app.local_entrypoint()
def main(
    condition: str = "cues",
    limit: int | None = None,
    epochs: float = 1.0,
    encoder: str = "qkv",
    dither: bool = False,
    dither_seed: int | None = None,
    dense_augment: int = 0,
    dense_augment_seed: int = 2026,
):
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from model.finetune_data import build_examples

    manifest = ARTIFACTS_ROOT / "train.jsonl"
    examples = build_examples(manifest, condition)
    if limit is not None:
        import random

        random.Random(0).shuffle(examples)
        examples = examples[:limit]
    payload = [
        {"item_id": e.item_id, "audio_path": e.audio_path, "prompt": e.prompt, "target": e.target}
        for e in examples
    ]

    if dense_augment > 0:
        if condition != "cues":
            raise SystemExit("--dense-augment only applies to --condition cues")
        from model.dense_cue_augmentation import (
            DenseAugmentationConfig,
            build_dense_synthetic_examples,
        )

        real_audio_paths = sorted({e["audio_path"] for e in payload})
        synthetic = build_dense_synthetic_examples(
            real_audio_paths,
            DenseAugmentationConfig(count=dense_augment, seed=dense_augment_seed),
        )
        payload += [
            {"item_id": e.item_id, "audio_path": e.audio_path, "prompt": e.prompt, "target": e.target}
            for e in synthetic
        ]
        print(f"added {len(synthetic)} synthetic dense-cue examples "
              f"(seed={dense_augment_seed}); real audio reused, no new upload")

    print(
        f"condition={condition} encoder={encoder} "
        f"examples={len(payload)} epochs={epochs} "
        f"dither={dither} dither_seed={dither_seed} "
        f"dense_augment={dense_augment}"
    )
    _upload_needed_audio(payload)
    result = train.remote(
        payload, condition, epochs,
        encoder=encoder, dither=dither, dither_seed=dither_seed, dense_augment=dense_augment,
    )
    print(json.dumps(result, indent=2))
