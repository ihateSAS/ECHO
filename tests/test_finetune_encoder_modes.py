from __future__ import annotations

import pytest

from model.finetune_config import ENCODER_MODES, adapter_dir, format_trainable, trainable_by_component

peft = pytest.importorskip("peft", reason="peft is only needed to check LoRA targeting")

from peft import LoraConfig  # noqa: E402
from peft.tuners.tuners_utils import check_target_module_exists  # noqa: E402

ENCODER_ATTENTION = [
    f"audio_tower.layers.{layer}.self_attn.{projection}"
    for layer in (0, 15, 31)
    for projection in ("q_proj", "k_proj", "v_proj")
]
ENCODER_OTHER = [
    f"audio_tower.layers.{layer}.self_attn.out_proj" for layer in (0, 15, 31)
] + [f"audio_tower.layers.{layer}.{name}" for layer in (0, 31) for name in ("fc1", "fc2")]
LANGUAGE_MODEL = [
    f"language_model.model.layers.{layer}.self_attn.{projection}"
    for layer in (0, 27)
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
] + [
    f"language_model.model.layers.{layer}.mlp.{projection}"
    for layer in (0, 27)
    for projection in ("gate_proj", "up_proj", "down_proj")
]


def adapted(mode: str, keys: list[str]) -> list[str]:
    config = LoraConfig(r=16, target_modules=ENCODER_MODES[mode])
    return [key for key in keys if check_target_module_exists(config, key)]


@pytest.mark.parametrize("mode", sorted(ENCODER_MODES))
def test_every_mode_adapts_the_whole_language_model(mode: str) -> None:
    assert adapted(mode, LANGUAGE_MODEL) == LANGUAGE_MODEL


def test_the_default_mode_reaches_into_the_audio_encoder() -> None:
    assert adapted("qkv", ENCODER_ATTENTION) == ENCODER_ATTENTION
    assert adapted("qkv", ENCODER_OTHER) == []


def test_none_is_the_frozen_encoder_the_old_runs_only_claimed_to_be() -> None:
    assert adapted("none", ENCODER_ATTENTION) == []
    assert adapted("none", ENCODER_OTHER) == []


def test_all_reaches_every_linear_layer_in_the_encoder() -> None:
    assert adapted("all", ENCODER_ATTENTION) == ENCODER_ATTENTION
    assert adapted("all", ENCODER_OTHER) == ENCODER_OTHER


def test_full_leaves_the_encoder_to_direct_training_not_lora() -> None:
    assert adapted("full", ENCODER_ATTENTION) == []
    assert adapted("full", ENCODER_OTHER) == []


def test_the_projector_is_never_lora_adapted() -> None:
    for mode in ENCODER_MODES:
        assert adapted(mode, ["multi_modal_projector.linear"]) == []


def test_the_default_mode_keeps_the_existing_adapter_path() -> None:
    assert adapter_dir("audio_only", "qkv") == "audio_only"
    assert adapter_dir("cues", "qkv") == "cues"


def test_new_modes_do_not_overwrite_an_existing_adapter() -> None:
    paths = {adapter_dir("audio_only", mode) for mode in ENCODER_MODES}
    assert len(paths) == len(ENCODER_MODES)
    assert "audio_only_enc-none" in paths


def test_an_unknown_mode_is_rejected_rather_than_silently_freezing() -> None:
    with pytest.raises(ValueError, match="unknown encoder mode"):
        adapter_dir("cues", "encoder")


class _FakeParameter:
    def __init__(self, count: int, requires_grad: bool) -> None:
        self._count = count
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self._count


class _FakeModel:
    def __init__(self, parameters: dict[str, tuple[int, bool]]) -> None:
        self._parameters = parameters

    def named_parameters(self):
        for name, (count, grad) in self._parameters.items():
            yield name, _FakeParameter(count, grad)


def test_the_report_separates_the_encoder_from_the_language_model() -> None:
    model = _FakeModel(
        {
            "audio_tower.layers.0.self_attn.q_proj.lora_A.weight": (100, True),
            "audio_tower.layers.0.self_attn.q_proj.base_layer.weight": (900, False),
            "language_model.model.layers.0.self_attn.q_proj.lora_A.weight": (50, True),
            "language_model.model.layers.0.self_attn.q_proj.base_layer.weight": (450, False),
            "multi_modal_projector.linear.weight": (200, False),
        }
    )
    counts = trainable_by_component(model)
    assert counts["audio_tower"] == (100, 1000)
    assert counts["language_model"] == (50, 500)
    assert counts["multi_modal_projector"] == (0, 200)
    rendered = format_trainable(counts)
    assert "audio_tower" in rendered and "10.00%" in rendered


def test_the_projector_mode_leaves_the_encoder_alone() -> None:
    assert adapted("projector", ENCODER_ATTENTION) == []
    assert adapted("projector", ENCODER_OTHER) == []


def test_only_the_direct_modes_train_real_weights() -> None:
    from model.finetune_config import directly_trained

    assert directly_trained("qkv") == ()
    assert directly_trained("none") == ()
    assert directly_trained("all") == ()
    assert directly_trained("full") == ("audio_tower", "multi_modal_projector")
    assert directly_trained("projector") == ("multi_modal_projector",)


def test_the_projector_is_trained_by_exactly_one_mode_directly() -> None:
    from model.finetune_config import DIRECTLY_TRAINED, ENCODER_MODES

    for mode in ENCODER_MODES:
        assert adapted(mode, ["multi_modal_projector.linear"]) == []
    trains_projector = {
        mode for mode, parts in DIRECTLY_TRAINED.items()
        if "multi_modal_projector" in parts
    }
    assert trains_projector == {"full", "projector"}


def test_an_unknown_mode_is_rejected_by_the_direct_lookup_too() -> None:
    from model.finetune_config import directly_trained

    with pytest.raises(ValueError, match="unknown encoder mode"):
        directly_trained("encoder")
