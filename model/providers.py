from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class FewShotExample:
    stereo: np.ndarray
    sample_rate: int
    side: str


def stereo_to_wav_bytes(stereo: np.ndarray, sample_rate: int) -> bytes:
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(stereo), sample_rate, subtype="PCM_16", format="WAV")
    return buffer.getvalue()


@runtime_checkable
class AudioLLM(Protocol):
    name: str

    def answer_with_cues(
        self, stereo: np.ndarray, sample_rate: int, cue_text: str, question: str
    ) -> str:
        ...

    def answer_audio_only(
        self, stereo: np.ndarray, sample_rate: int, question: str
    ) -> str:
        ...


class QwenAudioLLM:
    name = "qwen2-audio-7b"

    def __init__(self, backbone: Any, max_new_tokens: int = 200) -> None:
        self._backbone = backbone
        self._max_new_tokens = max_new_tokens

    def answer_with_cues(
        self, stereo: np.ndarray, sample_rate: int, cue_text: str, question: str
    ) -> str:
        from model.qwen_backbone import answer

        return answer(
            self._backbone,
            stereo,
            sample_rate,
            cue_text,
            question=question,
            max_new_tokens=self._max_new_tokens,
        )

    def answer_audio_only(
        self, stereo: np.ndarray, sample_rate: int, question: str
    ) -> str:
        from model.qwen_backbone import answer_from_stereo

        return answer_from_stereo(
            self._backbone,
            stereo,
            sample_rate,
            question=question,
            max_new_tokens=self._max_new_tokens,
        )


STEREO_PREAMBLE = (
    "This is a stereo recording. Listen to the left and right channels and "
    "judge where the instrument sits in the stereo image."
)


class OpenAIAudioLLM:
    def __init__(
        self,
        model: str = "gpt-audio",
        client: Any = None,
        name: str | None = None,
    ) -> None:
        self.model = model
        self.name = name or model
        self._client = client

    def _ensure_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    def _complete(self, wav_bytes: bytes, text: str) -> str:
        audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
        response = self._ensure_client().chat.completions.create(
            model=self.model,
            modalities=["text"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "input_audio",
                            "input_audio": {"data": audio_b64, "format": "wav"},
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content.strip()

    def answer_with_cues(
        self, stereo: np.ndarray, sample_rate: int, cue_text: str, question: str
    ) -> str:
        from model.qwen_backbone import downmix_to_mono

        mono = downmix_to_mono(stereo)
        return self._complete(
            stereo_to_wav_bytes(mono, sample_rate), f"{cue_text}\n{question}"
        )

    def answer_audio_only(
        self, stereo: np.ndarray, sample_rate: int, question: str
    ) -> str:
        return self._complete(
            stereo_to_wav_bytes(stereo, sample_rate),
            f"{STEREO_PREAMBLE}\n{question}",
        )

    def answer_audio_only_fewshot(
        self,
        examples: list[FewShotExample],
        stereo: np.ndarray,
        sample_rate: int,
        question: str,
    ) -> str:
        messages: list[dict[str, Any]] = []
        for example in examples:
            audio_b64 = base64.b64encode(
                stereo_to_wav_bytes(example.stereo, example.sample_rate)
            ).decode("ascii")
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{STEREO_PREAMBLE}\n{question}"},
                        {
                            "type": "input_audio",
                            "input_audio": {"data": audio_b64, "format": "wav"},
                        },
                    ],
                }
            )
            messages.append({"role": "assistant", "content": example.side})
        audio_b64 = base64.b64encode(stereo_to_wav_bytes(stereo, sample_rate)).decode(
            "ascii"
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{STEREO_PREAMBLE}\n{question}"},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": "wav"},
                    },
                ],
            }
        )
        response = self._ensure_client().chat.completions.create(
            model=self.model, modalities=["text"], messages=messages
        )
        return response.choices[0].message.content.strip()


class GeminiAudioLLM:
    def __init__(
        self,
        model: str = "gemini-flash-latest",
        client: Any = None,
        name: str | None = None,
    ) -> None:
        self.model = model
        self.name = name or model
        self._client = client

    def _ensure_client(self) -> Any:
        if self._client is None:
            from google import genai

            self._client = genai.Client()
        return self._client

    def _generate(self, wav_bytes: bytes, text: str) -> str:
        from google.genai import types

        response = self._ensure_client().models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
                text,
            ],
        )
        return response.text.strip()

    def answer_with_cues(
        self, stereo: np.ndarray, sample_rate: int, cue_text: str, question: str
    ) -> str:
        from model.qwen_backbone import downmix_to_mono

        mono = downmix_to_mono(stereo)
        return self._generate(
            stereo_to_wav_bytes(mono, sample_rate), f"{cue_text}\n{question}"
        )

    def answer_audio_only(
        self, stereo: np.ndarray, sample_rate: int, question: str
    ) -> str:
        return self._generate(
            stereo_to_wav_bytes(stereo, sample_rate),
            f"{STEREO_PREAMBLE}\n{question}",
        )

    def answer_audio_only_fewshot(
        self,
        examples: list[FewShotExample],
        stereo: np.ndarray,
        sample_rate: int,
        question: str,
    ) -> str:
        from google.genai import types

        contents: list[Any] = []
        for example in examples:
            contents.append(
                types.Part.from_bytes(
                    data=stereo_to_wav_bytes(example.stereo, example.sample_rate),
                    mime_type="audio/wav",
                )
            )
            contents.append(f"{STEREO_PREAMBLE}\n{question}")
            contents.append(example.side)
        contents.append(
            types.Part.from_bytes(
                data=stereo_to_wav_bytes(stereo, sample_rate), mime_type="audio/wav"
            )
        )
        contents.append(f"{STEREO_PREAMBLE}\n{question}")
        response = self._ensure_client().models.generate_content(
            model=self.model, contents=contents
        )
        return response.text.strip()


class KimiAudioLLM:
    def __init__(
        self,
        model: str = (
            "zsxkib/kimi-audio-7b-instruct:"
            "7500b32387695e89da3d09271850319ba027969f0c714dfc226361609ff29f2b"
        ),
        client: Any = None,
        name: str | None = None,
    ) -> None:
        self.model = model
        self.name = name or "kimi-audio"
        self._client = client

    def _ensure_client(self) -> Any:
        if self._client is None:
            import replicate

            self._client = replicate
        return self._client

    @staticmethod
    def _extract_text(output: Any) -> str:
        if isinstance(output, str):
            return output.strip()
        if isinstance(output, dict):
            for key in ("json_str", "text", "output"):
                if key in output and output[key]:
                    return str(output[key]).strip()
        if isinstance(output, list) and output:
            return KimiAudioLLM._extract_text(output[0])
        return str(output).strip()

    def _complete(self, wav_bytes: bytes, text: str) -> str:
        import io

        output = self._ensure_client().run(
            self.model,
            input={
                "audio": io.BytesIO(wav_bytes),
                "prompt": text,
                "output_type": "text",
                "return_json": True,
                "text_temperature": 0.0,
            },
        )
        return self._extract_text(output)

    def answer_with_cues(
        self, stereo: np.ndarray, sample_rate: int, cue_text: str, question: str
    ) -> str:
        from model.qwen_backbone import downmix_to_mono

        mono = downmix_to_mono(stereo)
        return self._complete(
            stereo_to_wav_bytes(mono, sample_rate), f"{cue_text}\n{question}"
        )

    def answer_audio_only(
        self, stereo: np.ndarray, sample_rate: int, question: str
    ) -> str:
        return self._complete(
            stereo_to_wav_bytes(stereo, sample_rate),
            f"{STEREO_PREAMBLE}\n{question}",
        )


class AudioFlamingo3LLM:
    def __init__(
        self,
        model: str = (
            "zsxkib/audio-flamingo-3:"
            "419bdd5ed04ba4e4609e66cc5082f6564e9d2c0836f9a286abe74bc20a357b84"
        ),
        client: Any = None,
        name: str | None = None,
    ) -> None:
        self.model = model
        self.name = name or "audio-flamingo-3"
        self._client = client

    def _ensure_client(self) -> Any:
        if self._client is None:
            import replicate

            self._client = replicate
        return self._client

    def _complete(self, wav_bytes: bytes, text: str) -> str:
        import io

        output = self._ensure_client().run(
            self.model,
            input={
                "audio": io.BytesIO(wav_bytes),
                "prompt": text,
                "temperature": 0.0,
                "enable_thinking": False,
            },
        )
        if isinstance(output, list):
            return "".join(str(part) for part in output).strip()
        return str(output).strip()

    def answer_with_cues(
        self, stereo: np.ndarray, sample_rate: int, cue_text: str, question: str
    ) -> str:
        from model.qwen_backbone import downmix_to_mono

        mono = downmix_to_mono(stereo)
        return self._complete(
            stereo_to_wav_bytes(mono, sample_rate), f"{cue_text}\n{question}"
        )

    def answer_audio_only(
        self, stereo: np.ndarray, sample_rate: int, question: str
    ) -> str:
        return self._complete(
            stereo_to_wav_bytes(stereo, sample_rate),
            f"{STEREO_PREAMBLE}\n{question}",
        )


_API_PROVIDERS = {
    "openai": OpenAIAudioLLM,
    "gpt-4o-audio": lambda **kw: OpenAIAudioLLM(model="gpt-audio", **kw),
    "gpt-audio-mini": lambda **kw: OpenAIAudioLLM(model="gpt-audio-mini", **kw),
    "gemini": GeminiAudioLLM,
    "gemini-flash": lambda **kw: GeminiAudioLLM(model="gemini-flash-latest", **kw),
    "gemini-pro": lambda **kw: GeminiAudioLLM(model="gemini-pro-latest", **kw),
    "gemini-flash-lite": lambda **kw: GeminiAudioLLM(model="gemini-flash-lite-latest", **kw),
    "kimi-audio": KimiAudioLLM,
    "audio-flamingo-3": AudioFlamingo3LLM,
}


def available_api_providers() -> tuple[str, ...]:
    return tuple(sorted(_API_PROVIDERS))


def get_api_provider(name: str, **kwargs: Any) -> AudioLLM:
    if name not in _API_PROVIDERS:
        raise KeyError(
            f"unknown provider {name!r}; known: {', '.join(available_api_providers())}"
        )
    return _API_PROVIDERS[name](**kwargs)
