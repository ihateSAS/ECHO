from __future__ import annotations

LLM_PROJECTIONS = "q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj"
ENCODER_PROJECTIONS = "q_proj|k_proj|v_proj|out_proj|fc1|fc2"


ENCODER_MODES: dict[str, list[str] | str] = {


    "qkv": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],

    "none": rf"language_model\..*\.({LLM_PROJECTIONS})",

    "all": (
        rf"(language_model\..*\.({LLM_PROJECTIONS}))"
        rf"|(audio_tower\..*\.({ENCODER_PROJECTIONS}))"
    ),


    "full": rf"language_model\..*\.({LLM_PROJECTIONS})",


    "projector": rf"language_model\..*\.({LLM_PROJECTIONS})",
}


DIRECTLY_TRAINED: dict[str, tuple[str, ...]] = {
    "qkv": (),
    "none": (),
    "all": (),
    "full": ("audio_tower", "multi_modal_projector"),
    "projector": ("multi_modal_projector",),
}


def directly_trained(encoder: str) -> tuple[str, ...]:
    if encoder not in ENCODER_MODES:
        raise ValueError(
            f"unknown encoder mode {encoder!r}; pick one of {sorted(ENCODER_MODES)}"
        )
    return DIRECTLY_TRAINED[encoder]


def adapter_dir(condition: str, encoder: str) -> str:
    if encoder not in ENCODER_MODES:
        raise ValueError(f"unknown encoder mode {encoder!r}; pick one of {sorted(ENCODER_MODES)}")
    return condition if encoder == "qkv" else f"{condition}_enc-{encoder}"


def trainable_by_component(model) -> dict[str, tuple[int, int]]:
    counts: dict[str, list[int]] = {}
    for name, parameter in model.named_parameters():
        if "audio_tower" in name:
            part = "audio_tower"
        elif "multi_modal_projector" in name:
            part = "multi_modal_projector"
        else:
            part = "language_model"
        slot = counts.setdefault(part, [0, 0])
        slot[1] += parameter.numel()
        if parameter.requires_grad:
            slot[0] += parameter.numel()
    return {part: (values[0], values[1]) for part, values in counts.items()}


def format_trainable(counts: dict[str, tuple[int, int]]) -> str:
    lines = ["trainable parameters by component:"]
    for part in sorted(counts):
        trainable, total = counts[part]
        share = 100.0 * trainable / total if total else 0.0
        lines.append(f"  {part:22s} {trainable:>12,} / {total:>13,}  ({share:5.2f}%)")
    return "\n".join(lines)
