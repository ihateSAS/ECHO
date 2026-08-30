from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.kimi_backbone import (  # noqa: E402
    STEREO_PREAMBLE,
    Backbone,
    answer,
    answer_from_stereo,
)

SAMPLE_RATE = 48_000


def _stereo(samples: int = 4800) -> np.ndarray:
    t = np.arange(samples) / SAMPLE_RATE
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.1 * np.sin(2 * np.pi * 440 * t)
    return np.column_stack((left, right))


class _FakeKimiModel:
    def __init__(self, reply: str = "The instrument is on the left.") -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def generate(self, messages, output_type, **sampling_params):
        self.calls.append(
            {"messages": messages, "output_type": output_type, "params": sampling_params}
        )
        return None, self.reply

    def last_text_message(self) -> str:
        for message in self.calls[-1]["messages"]:
            if message["message_type"] == "text":
                return message["content"]
        raise AssertionError("no text message sent")

    def last_audio_path(self) -> str:
        for message in self.calls[-1]["messages"]:
            if message["message_type"] == "audio":
                return message["content"]
        raise AssertionError("no audio message sent")


def test_cue_condition_sends_mono_audio_and_the_cue_text():
    model = _FakeKimiModel()
    backbone = Backbone(model=model)

    reply = answer(
        backbone, _stereo(), SAMPLE_RATE, "Level difference: 6.0 dB, left louder.", "Where is it?"
    )

    assert reply == "The instrument is on the left."
    assert "6.0 dB" in model.last_text_message()
    assert "Where is it?" in model.last_text_message()
    data, rate = sf.read(model.last_audio_path())
    assert rate == SAMPLE_RATE
    assert data.ndim == 1


def test_audio_only_condition_sends_the_real_stereo_file_and_no_cue_numbers():
    model = _FakeKimiModel()
    backbone = Backbone(model=model)

    answer_from_stereo(backbone, _stereo(), SAMPLE_RATE, "Where is the instrument?")

    assert STEREO_PREAMBLE in model.last_text_message()
    assert "dB" not in model.last_text_message()
    data, rate = sf.read(model.last_audio_path())
    assert rate == SAMPLE_RATE
    assert data.ndim == 2 and data.shape[1] == 2


def test_generate_is_called_with_output_type_text_not_both():
    model = _FakeKimiModel()
    backbone = Backbone(model=model)

    answer_from_stereo(backbone, _stereo(), SAMPLE_RATE)

    assert model.calls[-1]["output_type"] == "text"
