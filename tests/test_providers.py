from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.providers import (  # noqa: E402
    AudioFlamingo3LLM,
    AudioLLM,
    FewShotExample,
    GeminiAudioLLM,
    KimiAudioLLM,
    OpenAIAudioLLM,
    QwenAudioLLM,
    available_api_providers,
    get_api_provider,
    stereo_to_wav_bytes,
)

SAMPLE_RATE = 48_000


def _stereo(samples: int = 4800) -> np.ndarray:
    t = np.arange(samples) / SAMPLE_RATE
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.1 * np.sin(2 * np.pi * 440 * t)
    return np.column_stack((left, right))


def _wav_channels(wav_bytes: bytes) -> int:
    import soundfile as sf

    info = sf.info(io.BytesIO(wav_bytes))
    return info.channels


def test_wav_bytes_roundtrip_preserves_shape_and_rate():
    import soundfile as sf

    stereo = _stereo()
    data, rate = sf.read(io.BytesIO(stereo_to_wav_bytes(stereo, SAMPLE_RATE)))
    assert rate == SAMPLE_RATE
    assert data.shape == stereo.shape


def test_wav_bytes_accepts_mono():
    assert _wav_channels(stereo_to_wav_bytes(np.zeros(1000), SAMPLE_RATE)) == 1


def test_registry_lists_and_builds_known_providers():
    names = available_api_providers()
    assert "gpt-4o-audio" in names
    assert "gemini" in names
    assert isinstance(get_api_provider("gpt-4o-audio"), OpenAIAudioLLM)
    assert isinstance(get_api_provider("gemini-pro"), GeminiAudioLLM)


def test_unknown_provider_raises_with_the_known_list():
    with pytest.raises(KeyError, match="unknown provider"):
        get_api_provider("not-a-real-model")


class _FakeOpenAIClient:
    def __init__(self, reply: str = "The instrument is on the left."):
        self.reply = reply
        self.calls: list[dict] = []

        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)

                class _Msg:
                    content = outer.reply

                class _Choice:
                    message = _Msg()

                class _Resp:
                    choices = [_Choice()]

                return _Resp()

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()

    def sent_audio_channels(self) -> int:
        part = self.calls[-1]["messages"][0]["content"][1]
        wav = base64.b64decode(part["input_audio"]["data"])
        return _wav_channels(wav)

    def sent_text(self) -> str:
        return self.calls[-1]["messages"][0]["content"][0]["text"]

    def sent_messages(self) -> list[dict]:
        return self.calls[-1]["messages"]


def test_openai_cue_condition_sends_mono_audio_and_the_cue_text():
    client = _FakeOpenAIClient()
    model = OpenAIAudioLLM(client=client)

    reply = model.answer_with_cues(
        _stereo(), SAMPLE_RATE, "Level difference: 6.0 dB, left louder.", "Where is it?"
    )

    assert isinstance(reply, str) and reply
    assert client.sent_audio_channels() == 1
    assert "6.0 dB" in client.sent_text()
    assert "Where is it?" in client.sent_text()


def test_openai_audio_only_sends_stereo_and_no_cue_numbers():
    client = _FakeOpenAIClient()
    model = OpenAIAudioLLM(client=client)

    model.answer_audio_only(_stereo(), SAMPLE_RATE, "Where is the instrument?")

    assert client.sent_audio_channels() == 2
    assert "dB" not in client.sent_text()


def test_openai_adapter_satisfies_the_protocol():
    assert isinstance(OpenAIAudioLLM(client=_FakeOpenAIClient()), AudioLLM)


def test_openai_fewshot_sends_examples_as_real_turns_then_the_real_query():
    client = _FakeOpenAIClient()
    model = OpenAIAudioLLM(client=client)
    examples = [
        FewShotExample(_stereo(), SAMPLE_RATE, "left"),
        FewShotExample(_stereo(), SAMPLE_RATE, "right"),
        FewShotExample(_stereo(), SAMPLE_RATE, "center"),
    ]

    reply = model.answer_audio_only_fewshot(
        examples, _stereo(), SAMPLE_RATE, "Where is the instrument?"
    )

    assert isinstance(reply, str) and reply
    messages = client.sent_messages()
    assert len(messages) == 7
    assert [m["role"] for m in messages] == [
        "user", "assistant", "user", "assistant", "user", "assistant", "user",
    ]
    assert [m["content"] for m in messages[1::2]] == ["left", "right", "center"]
    for message in messages[0::2]:
        part = message["content"][1]
        wav = base64.b64decode(part["input_audio"]["data"])
        assert _wav_channels(wav) == 2


class _FakeGeminiClient:
    def __init__(self, reply: str = "It is centred."):
        self.reply = reply
        self.calls: list[list] = []
        self.models = self

    def generate_content(self, model, contents):
        self.calls.append(contents)

        class _Resp:
            text = self.reply

        return _Resp()

    def sent_audio_channels(self) -> int:
        for part in self.calls[-1]:
            if hasattr(part, "inline_data") and part.inline_data is not None:
                return _wav_channels(part.inline_data.data)
        raise AssertionError("no audio part was sent")

    def sent_text(self) -> str:
        return next(p for p in self.calls[-1] if isinstance(p, str))


def test_gemini_cue_condition_sends_mono_and_text():
    client = _FakeGeminiClient()
    model = GeminiAudioLLM(client=client)

    model.answer_with_cues(
        _stereo(), SAMPLE_RATE, "Level difference: 6.0 dB, left louder.", "Where?"
    )

    assert client.sent_audio_channels() == 1
    assert "6.0 dB" in client.sent_text()


def test_gemini_audio_only_sends_stereo():
    client = _FakeGeminiClient()
    model = GeminiAudioLLM(client=client)

    model.answer_audio_only(_stereo(), SAMPLE_RATE, "Where?")

    assert client.sent_audio_channels() == 2
    assert "dB" not in client.sent_text()


def test_gemini_fewshot_sends_examples_as_audio_question_answer_triples():
    client = _FakeGeminiClient()
    model = GeminiAudioLLM(client=client)
    examples = [
        FewShotExample(_stereo(), SAMPLE_RATE, "left"),
        FewShotExample(_stereo(), SAMPLE_RATE, "right"),
        FewShotExample(_stereo(), SAMPLE_RATE, "center"),
    ]

    reply = model.answer_audio_only_fewshot(
        examples, _stereo(), SAMPLE_RATE, "Where is the instrument?"
    )

    assert isinstance(reply, str) and reply
    contents = client.calls[-1]
    assert len(contents) == 11
    audio_parts = [
        p for p in contents if hasattr(p, "inline_data") and p.inline_data is not None
    ]
    assert len(audio_parts) == 4
    for part in audio_parts:
        assert _wav_channels(part.inline_data.data) == 2
    plain_strings = [p for p in contents if isinstance(p, str)]
    assert "left" in plain_strings
    assert "right" in plain_strings
    assert "center" in plain_strings


class _FakeReplicateClient:
    def __init__(self, reply: str = "The instrument is on the left.") -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def run(self, model, input):
        self.calls.append({"model": model, "input": input})
        return {"json_str": self.reply}

    def sent_audio_channels(self) -> int:
        wav_bytes = self.calls[-1]["input"]["audio"].getvalue()
        return _wav_channels(wav_bytes)

    def sent_text(self) -> str:
        return self.calls[-1]["input"]["prompt"]


def test_kimi_cue_condition_sends_mono_audio_and_the_cue_text():
    client = _FakeReplicateClient()
    model = KimiAudioLLM(client=client)

    reply = model.answer_with_cues(
        _stereo(), SAMPLE_RATE, "Level difference: 6.0 dB, left louder.", "Where is it?"
    )

    assert reply == "The instrument is on the left."
    assert client.sent_audio_channels() == 1
    assert "6.0 dB" in client.sent_text()
    assert "Where is it?" in client.sent_text()


def test_kimi_audio_only_sends_stereo_and_no_cue_numbers():
    client = _FakeReplicateClient()
    model = KimiAudioLLM(client=client)

    model.answer_audio_only(_stereo(), SAMPLE_RATE, "Where is the instrument?")

    assert client.sent_audio_channels() == 2
    assert "dB" not in client.sent_text()


def test_kimi_adapter_satisfies_the_protocol():
    assert isinstance(KimiAudioLLM(client=_FakeReplicateClient()), AudioLLM)


def test_kimi_extract_text_handles_plain_string_and_list_output():
    assert KimiAudioLLM._extract_text("plain reply") == "plain reply"
    assert KimiAudioLLM._extract_text({"text": "dict reply"}) == "dict reply"
    assert KimiAudioLLM._extract_text(["list reply"]) == "list reply"


class _FakeReplicateStringClient:
    def __init__(self, reply: str = "The instrument is on the left.") -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def run(self, model, input):
        self.calls.append({"model": model, "input": input})
        return self.reply

    def sent_audio_channels(self) -> int:
        wav_bytes = self.calls[-1]["input"]["audio"].getvalue()
        return _wav_channels(wav_bytes)

    def sent_text(self) -> str:
        return self.calls[-1]["input"]["prompt"]


def test_flamingo3_cue_condition_sends_mono_audio_and_the_cue_text():
    client = _FakeReplicateStringClient()
    model = AudioFlamingo3LLM(client=client)

    reply = model.answer_with_cues(
        _stereo(), SAMPLE_RATE, "Level difference: 6.0 dB, left louder.", "Where is it?"
    )

    assert reply == "The instrument is on the left."
    assert client.sent_audio_channels() == 1
    assert "6.0 dB" in client.sent_text()
    assert "Where is it?" in client.sent_text()


def test_flamingo3_audio_only_sends_stereo_and_no_cue_numbers():
    client = _FakeReplicateStringClient()
    model = AudioFlamingo3LLM(client=client)

    model.answer_audio_only(_stereo(), SAMPLE_RATE, "Where is the instrument?")

    assert client.sent_audio_channels() == 2
    assert "dB" not in client.sent_text()


def test_flamingo3_thinking_is_off_by_default():
    client = _FakeReplicateStringClient()
    model = AudioFlamingo3LLM(client=client)

    model.answer_audio_only(_stereo(), SAMPLE_RATE, "Where is the instrument?")

    assert client.calls[-1]["input"]["enable_thinking"] is False


def test_flamingo3_adapter_satisfies_the_protocol():
    assert isinstance(AudioFlamingo3LLM(client=_FakeReplicateStringClient()), AudioLLM)


def test_qwen_adapter_delegates_to_backbone_functions(monkeypatch):
    calls = {}

    def fake_answer(backbone, stereo, sr, cue_text, question, max_new_tokens):
        calls["cues"] = (cue_text, question, max_new_tokens)
        return "cue reply"

    def fake_answer_from_stereo(backbone, stereo, sr, question, max_new_tokens):
        calls["audio_only"] = (question, max_new_tokens)
        return "audio reply"

    monkeypatch.setattr("model.qwen_backbone.answer", fake_answer)
    monkeypatch.setattr("model.qwen_backbone.answer_from_stereo", fake_answer_from_stereo)

    model = QwenAudioLLM(backbone=object(), max_new_tokens=99)
    assert model.name == "qwen2-audio-7b"

    assert model.answer_with_cues(_stereo(), SAMPLE_RATE, "cue", "q?") == "cue reply"
    assert calls["cues"] == ("cue", "q?", 99)

    assert model.answer_audio_only(_stereo(), SAMPLE_RATE, "q?") == "audio reply"
    assert calls["audio_only"] == ("q?", 99)
