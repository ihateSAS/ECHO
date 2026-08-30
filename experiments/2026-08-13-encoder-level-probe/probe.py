#!/usr/bin/env python3


from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmark.inventory import build_inventory  # noqa: E402
from benchmark.splits import assign_piece_splits  # noqa: E402
from model.qwen_backbone import (  # noqa: E402
    BACKBONE_MODEL_ID,
    BACKBONE_SAMPLE_RATE,
    resample_for_backbone,
)
from spatialize.amplitude import gains_for_azimuth  # noqa: E402

ANGLES = (-25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0)
EXCERPT_SECONDS = 6.0
N_EXCERPTS = 10
SEED = 2026
OUT = Path(__file__).resolve().parent / "result.json"


def load_encoder():
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoConfig
    from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioEncoder

    config = AutoConfig.from_pretrained(BACKBONE_MODEL_ID)
    encoder = Qwen2AudioEncoder(config.audio_config)
    shard = hf_hub_download(BACKBONE_MODEL_ID, "model-00001-of-00005.safetensors")
    state = load_file(shard)
    prefix = "audio_tower."
    missing, unexpected = encoder.load_state_dict(
        {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)},
        strict=False,
    )
    if missing or unexpected:
        raise RuntimeError(f"audio tower did not load cleanly: {missing} {unexpected}")
    return encoder.eval()


def test_excerpts(rng: np.random.Generator) -> list[tuple[str, np.ndarray]]:
    inventory = build_inventory(REPO_ROOT / "data" / "URMP")
    assignments = assign_piece_splits(inventory, SEED)
    stems = [r for r in inventory if assignments[r.piece_id] == "test"]
    stems.sort(key=lambda r: r.stem_id)
    chosen = stems[:: max(1, len(stems) // N_EXCERPTS)][:N_EXCERPTS]

    out: list[tuple[str, np.ndarray]] = []
    for record in chosen:
        audio, sample_rate = sf.read(record.path, dtype="float64", always_2d=True)
        mono = audio[:, 0]
        want = int(EXCERPT_SECONDS * sample_rate)
        if mono.size <= want:
            continue

        best, best_rms = 0, -1.0
        for _ in range(20):
            start = int(rng.integers(0, mono.size - want))
            rms = float(np.sqrt(np.mean(mono[start : start + want] ** 2)))
            if rms > best_rms:
                best, best_rms = start, rms
        excerpt = mono[best : best + want]
        peak = np.abs(excerpt).max()
        if peak > 0:
            excerpt = excerpt / peak * 0.7
        out.append((record.stem_id, resample_for_backbone(excerpt, sample_rate)))
    return out


def shuffle_within_groups(
    targets: np.ndarray, groups: np.ndarray, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shuffled = np.array(targets, dtype=float, copy=True)
    for group in np.unique(groups):
        index = np.flatnonzero(groups == group)
        permuted = index.copy()
        rng.shuffle(permuted)
        shuffled[index] = np.asarray(targets, dtype=float)[permuted]
    return shuffled


def leave_one_out_probe(
    features: np.ndarray, targets: np.ndarray, groups: np.ndarray, ridge: float = 1.0
) -> dict[str, float]:
    predictions = np.zeros_like(targets, dtype=float)
    for group in np.unique(groups):
        train = groups != group
        held = groups == group
        x = features[train]
        y = targets[train]
        centre = x.mean(axis=0)
        x = x - centre
        y_mean = y.mean()
        gram = x.T @ x + ridge * np.eye(x.shape[1])
        weights = np.linalg.solve(gram, x.T @ (y - y_mean))
        predictions[held] = (features[held] - centre) @ weights + y_mean
    residual = float(np.sum((targets - predictions) ** 2))
    total = float(np.sum((targets - targets.mean()) ** 2))
    return {
        "mae_deg": float(np.mean(np.abs(targets - predictions))),
        "r2": 1.0 - residual / total,
        "within_5deg": float(np.mean(np.abs(targets - predictions) <= 5.0)),
    }


def main() -> None:
    from transformers import AutoProcessor

    rng = np.random.default_rng(SEED)
    extractor = AutoProcessor.from_pretrained(BACKBONE_MODEL_ID).feature_extractor
    encoder = load_encoder()
    excerpts = test_excerpts(rng)
    print(f"{len(excerpts)} held-out excerpts x {len(ANGLES)} angles")


    valid_frames = int(round(750 * EXCERPT_SECONDS / 30.0))

    def embed(channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mel = extractor(
            [channel], sampling_rate=BACKBONE_SAMPLE_RATE, return_tensors="pt"
        )["input_features"]
        mel_pooled = mel[0, :, : int(3000 * EXCERPT_SECONDS / 30.0)].mean(axis=1)
        with torch.no_grad():
            hidden = encoder(mel).last_hidden_state
        enc_pooled = hidden[0, :valid_frames].mean(axis=0)
        return mel_pooled.numpy(), enc_pooled.numpy()

    mel_rows, enc_rows, angles, groups = [], [], [], []
    for index, (stem_id, mono) in enumerate(excerpts):
        for angle in ANGLES:
            gain_l, gain_r = gains_for_azimuth(angle)
            mel_l, enc_l = embed((mono * gain_l).astype(np.float32))
            mel_r, enc_r = embed((mono * gain_r).astype(np.float32))
            mel_rows.append(mel_l - mel_r)
            enc_rows.append(enc_l - enc_r)
            angles.append(angle)
            groups.append(index)
        print(f"  {stem_id}: done ({index + 1}/{len(excerpts)})", flush=True)

    targets = np.array(angles, dtype=float)
    group_ids = np.array(groups)
    mel_features = np.array(mel_rows, dtype=float)
    enc_features = np.array(enc_rows, dtype=float)


    np.savez_compressed(
        Path(__file__).resolve().parent / "embeddings.npz",
        mel=mel_features,
        encoder=enc_features,
        targets=targets,
        groups=group_ids,
    )


    shuffled = shuffle_within_groups(targets, group_ids, SEED + 1)

    results = {
        "n_excerpts": len(excerpts),
        "n_items": len(targets),
        "angles_deg": list(ANGLES),
        "log_mel_control": leave_one_out_probe(mel_features, targets, group_ids),
        "encoder_output": leave_one_out_probe(enc_features, targets, group_ids),
        "encoder_shuffled_labels": leave_one_out_probe(
            enc_features, shuffled, group_ids
        ),
        "log_mel_shuffled_labels": leave_one_out_probe(
            mel_features, shuffled, group_ids
        ),
    }
    OUT.write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
