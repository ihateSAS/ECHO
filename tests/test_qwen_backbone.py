from __future__ import annotations

import math
import os
import re
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from benchmark.measure import measure_render
from benchmark.schemas import CueMeasurements
from model.qwen_backbone import (
    DEFAULT_DISCLOSURE,
    BACKBONE_MODEL_ID,
    BACKBONE_SAMPLE_RATE,
    build_messages,
    build_prompt,
    build_stereo_prompt,
    describe_cues,
    downmix_to_mono,
    prepare_audio,
    probe_channel_handling,
    resample_for_backbone,
)
from spatialize.amplitude import gains_for_azimuth, render_amplitude_pan

ANALYSIS_RATE = 48_000
CELLO_FILE = Path(__file__).with_name("AuSep_2_vc_01_Jupiter.wav")
AZIMUTH_TOLERANCE_DEG = 5.0
RUN_BACKBONE = os.environ.get("BINAURALCOT_RUN_BACKBONE") == "1"


def _example_cues(**overrides) -> CueMeasurements:
    values = dict(
        ild_db=6.2,
        high_frequency_ild_db=6.3,
        ild_flatness_db=0.2,
        itd_us=0.0,
        itd_peak_strength=0.8,
        coherence=0.94,
        estimated_azimuth_deg=10.9,
        estimated_regime="amplitude",
        active_bins=500,
    )
    values.update(overrides)
    return CueMeasurements(**values)


def _read_cello() -> tuple[np.ndarray, int]:
    if not CELLO_FILE.exists():
        pytest.skip("optional URMP cello fixture not present")
    return sf.read(CELLO_FILE, dtype="float64")


def _load_processor():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.Qwen2AudioProcessor.from_pretrained(
            BACKBONE_MODEL_ID, local_files_only=True
        )
    except Exception as error:
        pytest.skip(f"Qwen2-Audio processor unavailable: {error}")


def test_downmix_leaves_only_the_sum_that_carries_no_angle():
    mono = np.sin(2.0 * np.pi * 220.0 * np.arange(ANALYSIS_RATE) / ANALYSIS_RATE)
    left_gain, right_gain = gains_for_azimuth(15.0)

    downmixed = downmix_to_mono(render_amplitude_pan(mono, 15.0))

    assert np.allclose(downmixed, 0.5 * (left_gain + right_gain) * mono)


def test_downmixes_of_mirrored_angles_are_identical():
    mono = np.sin(2.0 * np.pi * 220.0 * np.arange(ANALYSIS_RATE) / ANALYSIS_RATE)

    left = downmix_to_mono(render_amplitude_pan(mono, 15.0))
    right = downmix_to_mono(render_amplitude_pan(mono, -15.0))

    assert np.allclose(left, right)


def test_downmix_passes_mono_through_untouched():
    mono = np.arange(10, dtype=np.float64)

    assert np.array_equal(downmix_to_mono(mono), mono)


def test_downmix_rejects_a_three_dimensional_array():
    with pytest.raises(ValueError, match="samples"):
        downmix_to_mono(np.zeros((4, 2, 2)))


def test_resampling_reaches_the_backbone_rate_and_keeps_the_pitch():
    seconds = 1.0
    times = np.arange(round(seconds * ANALYSIS_RATE)) / ANALYSIS_RATE
    tone = np.sin(2.0 * np.pi * 440.0 * times)

    resampled = resample_for_backbone(tone, ANALYSIS_RATE)

    assert resampled.dtype == np.float32
    assert resampled.size == pytest.approx(seconds * BACKBONE_SAMPLE_RATE, rel=0.01)
    spectrum = np.abs(np.fft.rfft(resampled.astype(np.float64)))
    peak_hz = np.fft.rfftfreq(resampled.size, 1.0 / BACKBONE_SAMPLE_RATE)[
        int(np.argmax(spectrum))
    ]
    assert peak_hz == pytest.approx(440.0, abs=2.0)


def test_resampling_a_signal_already_at_the_backbone_rate_is_a_no_op():
    signal = np.linspace(-1.0, 1.0, 1_000)

    assert np.allclose(resample_for_backbone(signal, BACKBONE_SAMPLE_RATE), signal)


def test_prepare_audio_downmixes_before_resampling():
    stereo = np.column_stack(
        (np.ones(ANALYSIS_RATE), np.full(ANALYSIS_RATE, 3.0))
    )

    prepared = prepare_audio(stereo, ANALYSIS_RATE)

    assert prepared.ndim == 1
    assert prepared.size == pytest.approx(BACKBONE_SAMPLE_RATE, rel=0.01)
    assert float(np.median(prepared)) == pytest.approx(2.0, abs=1e-3)


def test_resampling_rejects_stereo_and_bad_rates():
    with pytest.raises(ValueError, match="one-dimensional"):
        resample_for_backbone(np.zeros((100, 2)), ANALYSIS_RATE)
    with pytest.raises(ValueError, match="positive"):
        resample_for_backbone(np.zeros(100), 0)


def test_cue_text_reports_the_numbers_it_was_given():
    text = describe_cues(_example_cues())

    assert "6.2 dB" in text
    assert "Coherence: 0.94" in text
    assert "0 microseconds" in text


def test_cue_text_names_the_side_the_sign_convention_means():
    left = describe_cues(_example_cues(ild_db=6.2, estimated_azimuth_deg=10.9))
    right = describe_cues(_example_cues(ild_db=-6.2, estimated_azimuth_deg=-10.9))

    assert "left louder" in left
    assert "right louder" in right


def test_the_default_prompt_withholds_both_conclusions():
    text = describe_cues(_example_cues(estimated_azimuth_deg=10.9))

    assert "Measured regime" not in text
    assert "Azimuth implied" not in text
    assert "10.9 degrees" not in text
    assert "6.2 dB" in text
    assert "Coherence: 0.94" in text


def test_the_default_prompt_states_the_law_it_withholds_the_answer_to():
    evidence = _example_cues(ild_db=19.5, estimated_azimuth_deg=25.0)
    text = describe_cues(evidence)
    assert "arctan" in text and "tan 30 degrees" in text

    ratio = 10.0 ** (evidence.ild_db / 20.0)
    d = (ratio - 1.0) / (ratio + 1.0)
    derived = math.degrees(math.atan(d * math.tan(math.radians(30.0))))
    assert derived == pytest.approx(evidence.estimated_azimuth_deg, abs=0.1)


def test_the_conclusions_prompt_still_reproduces_the_part_6_wording():
    left = describe_cues(
        _example_cues(ild_db=6.2, estimated_azimuth_deg=10.9), disclosure="conclusions"
    )
    right = describe_cues(
        _example_cues(ild_db=-6.2, estimated_azimuth_deg=-10.9),
        disclosure="conclusions",
    )

    assert "Measured regime: amplitude." in left
    assert "10.9 degrees left" in left
    assert "10.9 degrees right" in right
    assert "arctan" not in left


def test_the_ablation_prompt_has_no_numbers_left_in_it():
    text = describe_cues(
        _example_cues(ild_db=6.2, itd_us=381.0), disclosure="no_numbers"
    )

    assert "left channel is louder" in text
    assert "right channel arrives later" in text
    assert not re.search(r"\d", text), text


def test_coherence_travels_with_a_word_not_just_a_decimal():
    assert "a single compact source" in describe_cues(_example_cues(coherence=1.0))
    assert "partly spread" in describe_cues(_example_cues(coherence=0.7))
    assert "widely spread" in describe_cues(_example_cues(coherence=0.1))


def test_cue_text_names_which_channel_is_late():
    assert "right channel later" in describe_cues(_example_cues(itd_us=381.0))
    assert "left channel later" in describe_cues(_example_cues(itd_us=-381.0))


def test_a_centred_source_never_reads_as_minus_zero_microseconds():
    text = describe_cues(_example_cues(itd_us=-0.02))

    assert "-0 microseconds" not in text
    assert "0 microseconds." in text
    assert "channel later" not in text


def test_cue_text_can_carry_the_part_five_numbers():
    text = describe_cues(_example_cues(), instrument="cello", f0_hz=146.8)

    assert "Instrument: cello." in text
    assert "146.8 Hz" in text


def test_cue_text_leaves_out_a_fundamental_it_does_not_have():
    assert "fundamental" not in describe_cues(_example_cues())


def test_messages_reject_empty_input():
    with pytest.raises(ValueError, match="cue_text"):
        build_messages("   ")
    with pytest.raises(ValueError, match="question"):
        build_messages("cues", "  ")


@pytest.mark.parametrize("azimuth_deg", [-25.0, -15.0, 15.0, 25.0])
def test_the_prompt_carries_the_angle_that_was_planted(azimuth_deg: float):
    mono, sample_rate = _read_cello()
    assert sample_rate == ANALYSIS_RATE

    stereo = render_amplitude_pan(mono, azimuth_deg)
    text = describe_cues(measure_render(stereo, sample_rate), disclosure="conclusions")

    match = re.search(r"([\d.]+) degrees (left|right)", text)
    assert match is not None, text
    recovered = float(match.group(1)) * (1.0 if match.group(2) == "left" else -1.0)
    assert recovered == pytest.approx(azimuth_deg, abs=AZIMUTH_TOLERANCE_DEG)


def test_a_centred_source_is_described_as_centred():
    mono, sample_rate = _read_cello()

    text = describe_cues(
        measure_render(render_amplitude_pan(mono, 0.0), sample_rate),
        disclosure="conclusions",
    )

    match = re.search(r"([\d.]+) degrees", text)
    assert match is not None
    assert float(match.group(1)) < AZIMUTH_TOLERANCE_DEG


def test_the_processor_drops_the_second_channel():
    handling = probe_channel_handling(_load_processor())

    assert handling.feature_extractor == "WhisperFeatureExtractor"
    assert handling.expects_sample_rate == BACKBONE_SAMPLE_RATE
    assert len(handling.mono_feature_shape) == 3
    assert handling.mono_feature_shape[0] == 1
    assert not handling.keeps_stereo
    assert handling.channels_became_batch_entries
    assert "drops the second channel" in handling.summary()


def test_the_prompt_the_template_builds_still_has_an_audio_slot():
    processor = _load_processor()

    prompt = build_prompt(processor, "Coherence: 0.94.", "Where is it?")

    assert "<|AUDIO|>" in prompt
    assert "Coherence: 0.94." in prompt
    assert "Where is it?" in prompt


def test_feature_extractor_preserves_level_difference():
    processor = _load_processor()
    fe = processor.feature_extractor
    rate = int(fe.sampling_rate)
    tone = np.sin(2.0 * np.pi * 440.0 * np.arange(rate) / rate).astype(np.float32)
    text = "<|audio_bos|><|AUDIO|><|audio_eos|>\n"

    def features(scale: float):
        return processor(
            text=text,
            audio=(scale * tone).astype(np.float32),
            sampling_rate=rate,
            return_tensors="pt",
        )["input_features"]

    loud = features(0.5)
    quiet = features(0.05)
    assert float((loud - quiet).abs().max()) > 0.1


def test_stereo_prompt_has_two_audio_slots_and_no_cue_numbers():
    processor = _load_processor()

    prompt = build_stereo_prompt(processor)

    assert prompt.count("<|AUDIO|>") == 2
    assert "left channel" in prompt
    assert "right channel" in prompt
    assert "dB" not in prompt
    assert "degrees" not in prompt or "azimuth in degrees" in prompt


def test_the_stereo_generation_path_runs_end_to_end():
    from model.qwen_backbone import answer_from_stereo

    processor = _load_processor()
    backbone = _tiny_backbone(processor)
    mono = np.sin(2.0 * np.pi * 220.0 * np.arange(ANALYSIS_RATE) / ANALYSIS_RATE)
    stereo = render_amplitude_pan(mono, 15.0)

    reply = answer_from_stereo(backbone, stereo, ANALYSIS_RATE, max_new_tokens=8)

    assert isinstance(reply, str)
    assert "<|AUDIO|>" not in reply


def test_stereo_path_rejects_a_mono_array():
    from model.qwen_backbone import answer_from_stereo

    processor = _load_processor()
    backbone = _tiny_backbone(processor)
    mono = np.sin(2.0 * np.pi * 220.0 * np.arange(ANALYSIS_RATE) / ANALYSIS_RATE)

    with pytest.raises(ValueError, match="stereo"):
        answer_from_stereo(backbone, mono, ANALYSIS_RATE, max_new_tokens=4)


def test_dither_stereo_channels_adds_independent_noise_per_channel():
    from model.qwen_backbone import dither_stereo_channels

    left = np.zeros(1000)
    right = np.zeros(1000)
    rng = np.random.default_rng(0)

    dithered_left, dithered_right = dither_stereo_channels(left, right, rng=rng)

    assert dithered_left.shape == left.shape
    assert dithered_right.shape == right.shape
    assert not np.array_equal(dithered_left, dithered_right)
    assert not np.allclose(dithered_left, 0.0)
    assert not np.allclose(dithered_right, 0.0)


def test_dither_stereo_channels_matches_the_dual_beats_formula():
    from model.qwen_backbone import dither_stereo_channels

    left = np.full(100, 0.3)
    right = np.full(100, -0.3)
    amplitude = 0.05

    rng = np.random.default_rng(42)
    expected_noise_left = np.random.default_rng(42).standard_normal(left.shape)
    dithered_left, _ = dither_stereo_channels(left, right, amplitude=amplitude, rng=rng)

    np.testing.assert_allclose(
        dithered_left, left + expected_noise_left * amplitude
    )


def test_dither_stereo_channels_default_amplitude_matches_the_paper():
    from model.qwen_backbone import DUAL_BEATS_DITHER_AMPLITUDE

    assert DUAL_BEATS_DITHER_AMPLITUDE == 0.05


def test_the_stereo_generation_path_with_dithering_runs_end_to_end():
    from model.qwen_backbone import answer_from_stereo

    processor = _load_processor()
    backbone = _tiny_backbone(processor)
    mono = np.sin(2.0 * np.pi * 220.0 * np.arange(ANALYSIS_RATE) / ANALYSIS_RATE)
    stereo = render_amplitude_pan(mono, 15.0)

    reply = answer_from_stereo(
        backbone, stereo, ANALYSIS_RATE, max_new_tokens=8, dither=True
    )

    assert isinstance(reply, str)
    assert "<|AUDIO|>" not in reply


def _tiny_backbone(processor):
    torch = pytest.importorskip("torch")
    from transformers import (
        Qwen2AudioConfig,
        Qwen2AudioEncoderConfig,
        Qwen2AudioForConditionalGeneration,
        Qwen2Config,
    )

    from model.qwen_backbone import Backbone

    config = Qwen2AudioConfig(
        audio_config=Qwen2AudioEncoderConfig(
            d_model=32,
            encoder_attention_heads=2,
            encoder_ffn_dim=64,
            encoder_layers=2,
            num_mel_bins=128,
            max_source_positions=1500,
            scale_embedding=False,
        ),
        text_config=Qwen2Config(
            vocab_size=len(processor.tokenizer),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=4096,
        ),
        audio_token_index=processor.tokenizer.convert_tokens_to_ids("<|AUDIO|>"),
    )
    torch.manual_seed(0)
    return Backbone(
        model=Qwen2AudioForConditionalGeneration(config).eval(),
        processor=processor,
        model_id="tiny-random-qwen2-audio",
        device="cpu",
    )


def test_the_generation_path_runs_end_to_end():
    from model.qwen_backbone import answer

    processor = _load_processor()
    backbone = _tiny_backbone(processor)
    mono = np.sin(2.0 * np.pi * 220.0 * np.arange(ANALYSIS_RATE) / ANALYSIS_RATE)
    stereo = render_amplitude_pan(mono, 15.0)
    cue_text = describe_cues(measure_render(stereo, ANALYSIS_RATE))

    reply = answer(backbone, stereo, ANALYSIS_RATE, cue_text, max_new_tokens=8)

    assert isinstance(reply, str)
    assert "Coherence" not in reply
    assert "<|AUDIO|>" not in reply


def test_generation_is_deterministic_with_greedy_decoding():
    from model.qwen_backbone import answer

    processor = _load_processor()
    backbone = _tiny_backbone(processor)
    mono = np.sin(2.0 * np.pi * 220.0 * np.arange(ANALYSIS_RATE) / ANALYSIS_RATE)
    stereo = render_amplitude_pan(mono, 15.0)

    first = answer(backbone, stereo, ANALYSIS_RATE, "Coherence: 0.94.", max_new_tokens=8)
    second = answer(backbone, stereo, ANALYSIS_RATE, "Coherence: 0.94.", max_new_tokens=8)

    assert first == second


@pytest.mark.skipif(
    not RUN_BACKBONE, reason="set BINAURALCOT_RUN_BACKBONE=1 to load the 7B model"
)
def test_the_model_answers_using_the_numbers_in_the_prompt():
    from model.qwen_backbone import answer, load_backbone

    mono, sample_rate = _read_cello()
    excerpt = mono[: 5 * sample_rate]
    stereo = render_amplitude_pan(excerpt, 25.0)
    cue_text = describe_cues(measure_render(stereo, sample_rate), instrument="cello")

    backbone = load_backbone()
    reply = answer(
        backbone, stereo, sample_rate, cue_text, max_new_tokens=120
    )

    assert reply.strip()
    assert math.isfinite(len(reply))
    assert "left" in reply.lower()


def test_the_command_line_exposes_every_disclosure_and_defaults_to_evidence(
    monkeypatch: pytest.MonkeyPatch,
):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_qwen_backbone",
        Path(__file__).resolve().parent.parent / "scripts" / "run_qwen_backbone.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr("sys.argv", ["run_qwen_backbone.py", "clip.wav"])
    assert module.parse_args().disclosure == DEFAULT_DISCLOSURE == "evidence"

    for value in ("evidence", "conclusions", "no_numbers"):
        monkeypatch.setattr(
            "sys.argv", ["run_qwen_backbone.py", "clip.wav", "--disclosure", value]
        )
        assert module.parse_args().disclosure == value

    monkeypatch.setattr(
        "sys.argv", ["run_qwen_backbone.py", "clip.wav", "--disclosure", "everything"]
    )
    with pytest.raises(SystemExit):
        module.parse_args()
