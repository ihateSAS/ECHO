from __future__ import annotations

from benchmark.questions import side_for_angle
from benchmark.schemas import MixtureQuestionRecord, MixtureRenderPlan


def make_mixture_localization_question(plan: MixtureRenderPlan) -> MixtureQuestionRecord:
    azimuths = tuple(sorted((stem.azimuth_deg for stem in plan.stems), reverse=True))
    return MixtureQuestionRecord(
        question_id=f"{plan.published_id}_mixture",
        question_type="mixture_localization",
        question=(
            "This mix contains several instruments panned to different "
            "positions in the stereo image. How many distinct positions can "
            "you hear, and what is the azimuth in degrees of each, ordered "
            "from left to right?"
        ),
        answer_count=len(azimuths),
        answer_azimuths_deg=azimuths,
        answer_sides=tuple(side_for_angle(angle) for angle in azimuths),
    )
