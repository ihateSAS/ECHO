from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import librosa
import numpy as np

from benchmark.schemas import CueMeasurements

BACKBONE_MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
BACKBONE_SAMPLE_RATE = 16_000
ANALYSIS_SAMPLE_RATE = 48_000
DEFAULT_QUESTION = "Where is this instrument positioned, and why?"
DEFAULT_MAX_NEW_TOKENS = 200

CueDisclosure = Literal["evidence", "conclusions", "no_numbers"]
DEFAULT_DISCLOSURE: CueDisclosure = "evidence"

COMPACT_COHERENCE = 0.90
DIFFUSE_COHERENCE = 0.40

PANNING_LAW_TEXT = (
    "An amplitude pan changes level only, never timing: both channels carry "
    "the same waveform at gains gL and gR. Write r = 10^(level difference in "
    "dB / 20), so the panning coordinate is d = (r - 1) / (r + 1), and the "
    "angle is theta = arctan(d * tan 30 degrees). A genuine time-of-arrival "
    "difference instead follows the Woodworth relation, delay = 2.551e-4 * "
    "(theta + sin theta) seconds with theta in radians. Positive azimuth "
    "means left."
)


@dataclass(frozen=True)
class ChannelHandling:
    model_id: str
    feature_extractor: str
    expects_sample_rate: int
    mono_feature_shape: tuple[int, ...]
    stereo_feature_shape: tuple[int, ...]
    keeps_stereo: bool
    channels_became_batch_entries: bool

    def summary(self) -> str:
        verdict = (
            "keeps both channels"
            if self.keeps_stereo
            else "drops the second channel: the feature tensor has no channel axis"
        )
        extra = (
            " (a (2, N) array is read as two unrelated clips, not as one "
            "stereo clip)"
            if self.channels_became_batch_entries
            else ""
        )
        return (
            f"{self.model_id}: {self.feature_extractor} at "
            f"{self.expects_sample_rate} Hz, mono features "
            f"{self.mono_feature_shape}, stereo input features "
            f"{self.stereo_feature_shape} -- {verdict}{extra}"
        )


@dataclass(frozen=True)
class Backbone:
    model: Any
    processor: Any
    model_id: str
    device: str


def probe_channel_handling(processor: Any) -> ChannelHandling:
    samples = 200
    times = np.arange(samples) / BACKBONE_SAMPLE_RATE
    left = (0.5 * np.sin(2.0 * np.pi * 220.0 * times)).astype(np.float32)
    right = (0.5 * np.sin(2.0 * np.pi * 880.0 * times)).astype(np.float32)
    text = "<|audio_bos|><|AUDIO|><|audio_eos|>\n"

    def features(audio: np.ndarray) -> Any:
        return processor(
            text=text,
            audio=audio,
            sampling_rate=BACKBONE_SAMPLE_RATE,
            return_tensors="pt",
        )["input_features"]

    mono = features(((left + right) / 2.0).astype(np.float32))
    stereo = features(np.stack((left, right)))
    left_alone = features(left)
    right_alone = features(right)

    channels_became_batch_entries = bool(
        stereo.shape[0] == 2
        and np.allclose(stereo[0].numpy(), left_alone[0].numpy(), atol=1e-4)
        and np.allclose(stereo[1].numpy(), right_alone[0].numpy(), atol=1e-4)
    )
    keeps_stereo = (
        len(stereo.shape) > len(mono.shape) or stereo.shape[1:] != mono.shape[1:]
    )

    return ChannelHandling(
        model_id=getattr(processor, "name_or_path", BACKBONE_MODEL_ID),
        feature_extractor=type(processor.feature_extractor).__name__,
        expects_sample_rate=int(processor.feature_extractor.sampling_rate),
        mono_feature_shape=tuple(mono.shape),
        stereo_feature_shape=tuple(stereo.shape),
        keeps_stereo=keeps_stereo,
        channels_became_batch_entries=channels_became_batch_entries,
    )


def downmix_to_mono(audio: np.ndarray) -> np.ndarray:
    signal = np.asarray(audio, dtype=np.float64)
    if signal.ndim == 1:
        return signal
    if signal.ndim != 2:
        raise ValueError("audio must be (samples,) or (samples, channels)")
    return signal.mean(axis=1)


def resample_for_backbone(mono: np.ndarray, sample_rate: int) -> np.ndarray:
    signal = np.asarray(mono, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("mono must be one-dimensional")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if sample_rate == BACKBONE_SAMPLE_RATE:
        return signal.astype(np.float32)
    return librosa.resample(
        signal, orig_sr=sample_rate, target_sr=BACKBONE_SAMPLE_RATE
    ).astype(np.float32)


def prepare_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    return resample_for_backbone(downmix_to_mono(audio), sample_rate)


def coherence_in_words(coherence: float) -> str:
    if coherence >= COMPACT_COHERENCE:
        return "a single compact source"
    if coherence >= DIFFUSE_COHERENCE:
        return "a partly spread image"
    return "a widely spread or reverberant image"


def describe_cues(
    evidence: CueMeasurements,
    instrument: str | None = None,
    f0_hz: float | None = None,
    disclosure: CueDisclosure = DEFAULT_DISCLOSURE,
) -> str:
    louder = "left" if evidence.ild_db > 0.0 else "right"
    later = "right" if evidence.itd_us > 0.0 else "left"
    side = "left" if evidence.estimated_azimuth_deg > 0.0 else "right"
    delay_us = evidence.itd_us if abs(evidence.itd_us) >= 0.5 else 0.0
    spread = coherence_in_words(evidence.coherence)

    if disclosure == "no_numbers":
        parts = [
            f"The {louder} channel is louder.",
            (
                f"The {later} channel arrives later."
                if abs(delay_us) >= 1.0
                else "Neither channel arrives earlier than the other."
            ),
            f"The stereo image is {spread}.",
        ]
        if instrument:
            parts.insert(0, f"Instrument: {instrument}.")
        return " ".join(parts)

    parts = [
        f"Inter-channel level difference: {evidence.ild_db:.1f} dB, "
        f"{louder} louder.",
        f"Above 2 kHz: {evidence.high_frequency_ild_db:.1f} dB "
        f"(flatness across octave bands {evidence.ild_flatness_db:.1f} dB).",
        f"Inter-channel delay: {delay_us:.0f} microseconds"
        + (f", {later} channel later." if abs(delay_us) >= 1.0 else "."),
        f"Coherence: {evidence.coherence:.2f} ({spread}).",
    ]
    if disclosure == "conclusions":
        parts.extend(
            [
                f"Measured regime: {evidence.estimated_regime}.",
                f"Azimuth implied by those numbers: "
                f"{abs(evidence.estimated_azimuth_deg):.1f} degrees {side} "
                f"(positive is left).",
            ]
        )
    if instrument:
        parts.insert(0, f"Instrument: {instrument}.")
    if f0_hz is not None and f0_hz > 0.0:
        parts.append(f"Median tracked fundamental frequency: {f0_hz:.1f} Hz.")
    if disclosure == "evidence":
        parts.append(PANNING_LAW_TEXT)
    return " ".join(parts)


def build_messages(cue_text: str, question: str = DEFAULT_QUESTION) -> list[dict]:
    if not cue_text.strip():
        raise ValueError("cue_text must not be empty")
    if not question.strip():
        raise ValueError("question must not be empty")
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "audio.wav"},
                {"type": "text", "text": f"{cue_text}\n{question}"},
            ],
        }
    ]


def build_prompt(
    processor: Any, cue_text: str, question: str = DEFAULT_QUESTION
) -> str:
    return processor.apply_chat_template(
        build_messages(cue_text, question), add_generation_prompt=True, tokenize=False
    )


def load_backbone(
    model_id: str = BACKBONE_MODEL_ID,
    device: str = "cpu",
    dtype: str = "bfloat16",
    quantization: str | None = None,
    adapter_dir: str | None = None,
) -> Backbone:
    import torch
    from transformers import Qwen2AudioForConditionalGeneration, Qwen2AudioProcessor

    processor = Qwen2AudioProcessor.from_pretrained(model_id)

    if quantization is None:
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_id, dtype=getattr(torch, dtype), low_cpu_mem_usage=True
        )
        model.to(device)
    elif quantization == "4bit":
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, dtype),
        )
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map=device,
            low_cpu_mem_usage=True,
        )
    else:
        raise ValueError(f"unknown quantization {quantization!r}; expected None or '4bit'")

    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)

    model.eval()
    return Backbone(model=model, processor=processor, model_id=model_id, device=device)


def answer(
    backbone: Backbone,
    audio: np.ndarray,
    sample_rate: int,
    cue_text: str,
    question: str = DEFAULT_QUESTION,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> str:
    import torch

    prompt = build_prompt(backbone.processor, cue_text, question)
    inputs = backbone.processor(
        text=prompt,
        audio=prepare_audio(audio, sample_rate),
        sampling_rate=BACKBONE_SAMPLE_RATE,
        return_tensors="pt",
    ).to(backbone.device)

    with torch.no_grad():
        generated = backbone.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )

    new_tokens = generated[:, inputs["input_ids"].shape[1] :]
    return backbone.processor.batch_decode(new_tokens, skip_special_tokens=True)[
        0
    ].strip()


STEREO_PREAMBLE = (
    "Audio 1 is the left channel of a stereo recording. Audio 2 is the "
    "right channel of the same recording."
)
DEFAULT_STEREO_QUESTION = (
    "From the two channels alone, where is the instrument positioned in the "
    "stereo image? Return its side (left, right, or centre) and, if you can, "
    "an azimuth in degrees."
)


DUAL_BEATS_DITHER_AMPLITUDE = 0.05


def dither_stereo_channels(
    left: np.ndarray,
    right: np.ndarray,
    amplitude: float = DUAL_BEATS_DITHER_AMPLITUDE,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    noise_left = rng.standard_normal(left.shape)
    noise_right = rng.standard_normal(right.shape)
    return left + noise_left * amplitude, right + noise_right * amplitude


def build_stereo_messages(
    preamble: str = STEREO_PREAMBLE, question: str = DEFAULT_STEREO_QUESTION
) -> list[dict]:
    if not question.strip():
        raise ValueError("question must not be empty")
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": preamble},
                {"type": "audio", "audio_url": "left.wav"},
                {"type": "audio", "audio_url": "right.wav"},
                {"type": "text", "text": question},
            ],
        }
    ]


def build_stereo_prompt(
    processor: Any,
    preamble: str = STEREO_PREAMBLE,
    question: str = DEFAULT_STEREO_QUESTION,
) -> str:
    return processor.apply_chat_template(
        build_stereo_messages(preamble, question),
        add_generation_prompt=True,
        tokenize=False,
    )


def answer_from_stereo(
    backbone: Backbone,
    stereo: np.ndarray,
    sample_rate: int,
    question: str = DEFAULT_STEREO_QUESTION,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    dither: bool = False,
    dither_amplitude: float = DUAL_BEATS_DITHER_AMPLITUDE,
    dither_rng: np.random.Generator | None = None,
) -> str:
    import torch

    audio = np.asarray(stereo, dtype=np.float64)
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError("stereo must be a (samples, 2) array")
    left = resample_for_backbone(audio[:, 0], sample_rate)
    right = resample_for_backbone(audio[:, 1], sample_rate)
    if dither:
        left, right = dither_stereo_channels(
            left, right, amplitude=dither_amplitude, rng=dither_rng
        )

    prompt = build_stereo_prompt(backbone.processor, question=question)
    inputs = backbone.processor(
        text=prompt,
        audio=[left, right],
        sampling_rate=BACKBONE_SAMPLE_RATE,
        return_tensors="pt",
    ).to(backbone.device)

    with torch.no_grad():
        generated = backbone.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )

    new_tokens = generated[:, inputs["input_ids"].shape[1] :]
    return backbone.processor.batch_decode(new_tokens, skip_special_tokens=True)[
        0
    ].strip()
