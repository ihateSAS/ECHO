from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np

from model.qwen_backbone import downmix_to_mono

BACKBONE_MODEL_ID = "moonshotai/Kimi-Audio-7B-Instruct"
DEFAULT_QUESTION = "Where is this instrument positioned, and why?"

STEREO_PREAMBLE = (
    "This is a stereo recording. Listen to the left and right channels and "
    "judge where the instrument sits in the stereo image."
)
DEFAULT_STEREO_QUESTION = (
    "From the audio alone, where is the instrument positioned in the "
    "stereo image? Return its side (left, right, or centre) and, if you "
    "can, an azimuth in degrees."
)

SAMPLING_PARAMS: dict[str, float | int] = {
    "audio_temperature": 0.8,
    "audio_top_k": 10,
    "text_temperature": 0.0,
    "text_top_k": 5,
    "audio_repetition_penalty": 1.0,
    "audio_repetition_window_size": 64,
    "text_repetition_penalty": 1.0,
    "text_repetition_window_size": 16,
}


@dataclass
class Backbone:
    model: Any
    model_id: str = BACKBONE_MODEL_ID


def load_backbone(model_id: str = BACKBONE_MODEL_ID) -> Backbone:
    from kimia_infer.api.kimia import KimiAudio

    model = KimiAudio(model_path=model_id, load_detokenizer=True)
    return Backbone(model=model, model_id=model_id)


def _write_wav(audio: np.ndarray, sample_rate: int) -> str:
    import soundfile as sf

    handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(handle.name, np.asarray(audio), sample_rate, subtype="PCM_16")
    return handle.name


def _generate(backbone: Backbone, text: str, wav_path: str) -> str:
    messages = [
        {"role": "user", "message_type": "text", "content": text},
        {"role": "user", "message_type": "audio", "content": wav_path},
    ]
    _, text_output = backbone.model.generate(
        messages, **SAMPLING_PARAMS, output_type="text"
    )
    return text_output.strip()


def answer(
    backbone: Backbone,
    stereo: np.ndarray,
    sample_rate: int,
    cue_text: str,
    question: str = DEFAULT_QUESTION,
) -> str:
    mono = downmix_to_mono(stereo)
    wav_path = _write_wav(mono, sample_rate)
    return _generate(backbone, f"{cue_text}\n{question}", wav_path)


def answer_from_stereo(
    backbone: Backbone,
    stereo: np.ndarray,
    sample_rate: int,
    question: str = DEFAULT_STEREO_QUESTION,
) -> str:
    wav_path = _write_wav(stereo, sample_rate)
    return _generate(backbone, f"{STEREO_PREAMBLE}\n{question}", wav_path)
