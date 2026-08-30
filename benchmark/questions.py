from __future__ import annotations

from typing import Literal

from benchmark.process import expected_after_processing
from benchmark.schemas import QuestionRecord, RenderPlan


def side_for_angle(angle_deg: float) -> Literal["left", "center", "right"]:
    if angle_deg > 0:
        return "left"
    if angle_deg < 0:
        return "right"
    return "center"


def answerable_azimuth_deg(plan: RenderPlan) -> float:
    if plan.regime == "binaural":
        return plan.azimuth_deg

    expected = expected_after_processing(plan.azimuth_deg, plan.processing)
    if expected.azimuth_deg is None:
        raise ValueError(
            f"{plan.render_id} has no recoverable azimuth after its processing "
            f"({'; '.join(expected.notes)}); validate_render should have "
            "rejected it before a question was built"
        )
    return expected.azimuth_deg


def make_absolute_localization_question(plan: RenderPlan) -> QuestionRecord:
    azimuth_deg = answerable_azimuth_deg(plan)
    return QuestionRecord(


        question_id=f"{plan.published_id}_absolute",
        question_type="absolute_localization",
        question=(
            f"Where is the {plan.instrument} positioned in the stereo image? "
            "Return its side and azimuth in degrees."
        ),
        target_instrument=plan.instrument,
        answer_side=side_for_angle(azimuth_deg),
        answer_azimuth_deg=azimuth_deg,
    )
