from __future__ import annotations

from dataclasses import dataclass

from benchmark.process import expected_after_processing
from benchmark.schemas import ProcessingStep


ON_GRID_EPSILON_DEG = 0.75


def trained_answer_angles(
    grid_deg: tuple[float, ...], midside_width: float | None = 1.2
) -> tuple[float, ...]:
    answers = {float(angle) for angle in grid_deg}
    if midside_width is not None:
        steps = (ProcessingStep(kind="midside", params={"width": midside_width}),)
        for angle in grid_deg:
            expected = expected_after_processing(float(angle), steps)
            if expected.azimuth_deg is not None:
                answers.add(round(expected.azimuth_deg, 4))
    return tuple(sorted(answers))


def distinct_answer_angles(config) -> tuple[float, ...]:
    if config.rendering == "binaural":
        return tuple(sorted({float(angle) for angle in config.angles_degrees}))

    answers: set[float] = set()
    for angle in config.angles_degrees:
        for recipe in ((), *config.processing_recipes):
            expected = expected_after_processing(float(angle), recipe)
            if expected.azimuth_deg is not None:
                answers.add(round(expected.azimuth_deg, 4))
    return tuple(sorted(answers))


def snap_to_trained(azimuth_deg: float, trained: tuple[float, ...]) -> float:
    return min(trained, key=lambda candidate: abs(candidate - azimuth_deg))


def on_grid_rate(
    predictions: list[float | None],
    trained: tuple[float, ...],
    epsilon_deg: float = ON_GRID_EPSILON_DEG,
) -> float:
    answered = [value for value in predictions if value is not None]
    if not answered:
        return 0.0
    hits = sum(
        1
        for value in answered
        if any(abs(value - angle) <= epsilon_deg for angle in trained)
    )
    return hits / len(answered)


def choose_eval_angles(
    trained: tuple[float, ...],
    count: int,
    limit_deg: float = 25.0,
    decimals: int = 1,
) -> tuple[float, ...]:
    inside = sorted(angle for angle in trained if abs(angle) <= limit_deg)
    gaps = [
        ((low + high) / 2.0, (high - low) / 2.0)
        for low, high in zip(inside, inside[1:])
        if high > low
    ]


    positive = sorted(
        (abs(centre), clearance) for centre, clearance in gaps if centre > 0
    )
    ranked = sorted(positive, key=lambda item: (-item[1], item[0]))
    chosen: list[float] = []
    for centre, _ in ranked:
        if len(chosen) >= count:
            break
        chosen.extend([round(-centre, decimals), round(centre, decimals)])
    return tuple(sorted(chosen[:count]))


@dataclass(frozen=True)
class Hypothesis:
    name: str
    within_tolerance: float
    mean_abs_error_deg: float
    on_grid_rate: float

    def summary(self) -> str:
        return (
            f"{self.name:24s} within 5 deg {self.within_tolerance:6.1%}  "
            f"MAE {self.mean_abs_error_deg:5.2f} deg  "
            f"answers on trained grid {self.on_grid_rate:6.1%}"
        )


@dataclass(frozen=True)
class Discriminability:
    eval_angles_deg: tuple[float, ...]
    trained_angles_deg: tuple[float, ...]
    memorises: Hypothesis
    computes: Hypothesis

    @property
    def separates_on_tolerance(self) -> bool:
        return abs(self.memorises.within_tolerance - self.computes.within_tolerance) > 0.05

    @property
    def separates_on_error(self) -> bool:
        return abs(self.memorises.mean_abs_error_deg - self.computes.mean_abs_error_deg) > 1.0

    @property
    def separates_on_grid_rate(self) -> bool:
        return abs(self.memorises.on_grid_rate - self.computes.on_grid_rate) > 0.5

    def to_dict(self) -> dict[str, object]:
        return {
            "eval_angles_deg": list(self.eval_angles_deg),
            "trained_angles_deg": list(self.trained_angles_deg),
            "hypotheses": {
                h.name: {
                    "within_tolerance": h.within_tolerance,
                    "mean_abs_error_deg": h.mean_abs_error_deg,
                    "on_grid_rate": h.on_grid_rate,
                }
                for h in (self.memorises, self.computes)
            },
            "separates_on": {
                "within_tolerance": self.separates_on_tolerance,
                "mean_abs_error": self.separates_on_error,
                "on_grid_rate": self.separates_on_grid_rate,
            },
        }

    def summary(self) -> str:
        lines = [
            f"eval angles:    {', '.join(f'{a:+g}' for a in self.eval_angles_deg)}",
            f"trained angles: {len(self.trained_angles_deg)} distinct",
            "",
            self.memorises.summary(),
            self.computes.summary(),
            "",
            "this eval set separates the two hypotheses on:",
            f"  within-tolerance rate  {'YES' if self.separates_on_tolerance else 'NO'}",
            f"  mean absolute error    {'YES' if self.separates_on_error else 'NO'}",
            f"  answers on grid        {'YES' if self.separates_on_grid_rate else 'NO'}",
        ]
        return "\n".join(lines)


def discriminability(
    eval_angles_deg: tuple[float, ...],
    trained_angles_deg: tuple[float, ...],
    tolerance_deg: float = 5.0,
) -> Discriminability:
    snapped = [snap_to_trained(angle, trained_angles_deg) for angle in eval_angles_deg]
    errors = [abs(pred - true) for pred, true in zip(snapped, eval_angles_deg)]
    memorises = Hypothesis(
        name="memorised the table",
        within_tolerance=sum(1 for e in errors if e <= tolerance_deg) / len(errors),
        mean_abs_error_deg=sum(errors) / len(errors),
        on_grid_rate=on_grid_rate(list(snapped), trained_angles_deg),
    )
    computes = Hypothesis(
        name="learned the panning law",
        within_tolerance=1.0,
        mean_abs_error_deg=0.0,
        on_grid_rate=on_grid_rate(list(eval_angles_deg), trained_angles_deg),
    )
    return Discriminability(
        eval_angles_deg=tuple(eval_angles_deg),
        trained_angles_deg=tuple(trained_angles_deg),
        memorises=memorises,
        computes=computes,
    )
