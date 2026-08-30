# StereoMusicQA v0.1: the first numbers

## What this is

The benchmark had 1,529 items and no way to be used as one. Nothing read
`test_public.jsonl`, ran anything over it, and checked the result against
`test_private_labels.jsonl`. So there was no floor, no ceiling, and no
scale on which any future model result could be read.

This is that scale. Four reference conditions on the 209-item held-out
split, no GPU and no model involved.

```bash
python scripts/score_benchmark.py \
    --json experiments/2026-08-09-stereomusicqa-baselines/result.json
```

## Results, 209 held-out items

| Condition | Within 5° | Side | MAE (answered) | Coverage | Inference passed |
|---|---|---|---|---|---|
| no answer | 0.0% | 0.0% | — | 0% | 0.0% |
| always centre | **27.3%** | 9.1% | 13.64° | 100% | 18.2% |
| random from grid | 24.4% | 37.3% | 19.26° | 100% | 19.6% |
| front end (ceiling) | **100.0%** | 100.0% | 0.00° | 100% | 100.0% |

**27.3% is the floor.** A model that predicts "centre" every time, having
listened to nothing, lands inside the 5° tolerance on 27.3% of items. That
is not a quirk of this sample — it is arithmetic. The eleven angles are
uniform over ±25° in steps of 5, so three of them (−5, 0, +5) are within 5°
of centre, and 3/11 = 27.3%. Its mean error is 150/11 = 13.64°. Any result
at or below those numbers means the audio and the cues went unused.

**Random guessing from the published grid scores 24.4%**, slightly *worse*
than always-centre, and this is the deliberately strong form of random: it
already knows which eleven angles the benchmark uses. Note its side
accuracy is much better than always-centre's (37.3% vs 9.1%) while its
azimuth accuracy is slightly worse. Reporting only one of those two would
rank these two conditions in opposite orders, which is why both are in the
table.

**100% is the ceiling, and it is not a model result.** The front-end
condition answers with `cues/level.py`'s own measurement. On dry amplitude
pans it is exact by construction — both channels are the same waveform at
two gains, so the level ratio inverts to the planted angle at machine
precision. Its value is as a control: it confirms the labels and the audio
agree on every item, and it says what score is available to a model that
uses the numbers in its prompt perfectly.

That ceiling is also the honest framing of what v0.1 tests. The difficulty
is not in the signal processing — that is solved and exact. It is entirely
in whether a model can use cues it has been handed. A v0.2 with reverb,
mixtures, or genuine binaural rendering would move the ceiling below 100%
and start testing the front end too.

## The two scoring rules, and why they are not arbitrary

**Unanswered items count as wrong.** `within_tolerance_rate` is over every
item, not over the ones the model chose to answer. Compute it over answered
items and a model that replies to the three easy items and refuses the rest
scores 100%. There is a test for exactly that case
(`test_answering_only_the_easy_items_does_not_buy_a_good_score`): flawless
mean error, 3/11 coverage, and a benchmark score of 3/11.

**Mean error is over answered items only, and is never printed without
coverage beside it.** Both halves are needed; either alone can be made to
look good by sacrificing the other. The "no answer" row exists to prove the
degenerate case behaves: 0.0% within tolerance, not a NaN that could sort
above a real result.

## Faithfulness is a separate column on purpose

`inference_passed` is the verifier's rate over the two claims a model has
to derive rather than copy — the regime, and the azimuth the panning law
converts a level difference into. It is reported beside accuracy and never
folded into it, because the two come apart in both directions.

The baselines show one of those directions. Always-centre scores 18.2%
inference-passed while being wrong 72.7% of the time: it is right about the
items that genuinely are near centre, and being right for no reason still
counts as passing when the answer happens to match. The other direction — a
model that reasons correctly and answers wrongly — needs a real model to
observe, and is the thing the first real evaluation should look for.

Two caveats on those numbers. The baselines emit a single sentence, so most
of their transcripts make no checkable claim beyond the angle itself; a real
model's transcript will exercise far more of the verifier. And the baselines
were scored under the `"conclusions"`-era assumption that a stated angle is
a claim — under the current default prompt a model is not handed the angle
at all, which is the whole point of that change and will make the
inference column mean considerably more.

## What is still missing

A real model has not been run over this. That needs one generation pass
over 209 items, which `docs/compute.md` prices at well under a dollar of
rented GPU and 60 days of laptop CPU. The scoring half is done, costs
nothing, and is what was actually blocking the first evaluation.
