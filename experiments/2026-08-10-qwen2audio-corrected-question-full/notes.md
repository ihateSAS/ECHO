# The real zero-shot score: strong on direction, blind to arithmetic

## What this is

The full 1,045-item zero-shot run, done right. It supersedes
`experiments/2026-08-10-qwen2audio-zero-shot-first-score/`, which had a
bug: it asked the generic `DEFAULT_QUESTION` ("Where is this instrument
positioned, and why?") instead of each item's actual benchmark question
("...Return its side and azimuth in degrees."). That one-line fix
(`scripts/modal_generate.py` now uses `item["question"]`) took coverage
from 7.5% to 97.5%. This is the corrected number, over the whole held-out
set, and it is read the way the 2026-08-10 metric decision says to read it:
**side/coverage/inference as the primary axes, exact-degree as a
front-end/fine-tuning concern, not the headline.**

Run on Modal's free tier (L4), bf16, greedy, `"evidence"` disclosure.
Cost: $0. The run dropped its gRPC connection twice mid-way (free-tier
flakiness); `scripts/modal_generate.py`'s per-10-item checkpointing meant
each resume picked up exactly where it left off and no item was ever
regenerated -- 488 -> 861 -> 1045 across three invocations, same volume
file.

```bash
python scripts/score_benchmark.py \
    --labels artifacts/stereomusicqa_v0.2/test_private_labels.jsonl \
    --answers experiments/2026-08-10-qwen2audio-corrected-question-full/answers.jsonl \
    --baselines \
    --json experiments/2026-08-10-qwen2audio-corrected-question-full/result.json
```

## Results, 1,045 held-out items

| Condition | Side | Coverage | Within 5° | Inference |
|---|---|---|---|---|
| **Qwen2-Audio-7B (zero-shot)** | **71.8%** | **97.5%** | 0.3% | 0.3% |
| no answer | 0.0% | 0% | 0.0% | 0.0% |
| always centre | 9.1% | 100% | 23.6% | 14.5% |
| random from grid | 40.5% | 100% | 23.5% | 18.9% |
| front end (ceiling) | 100% | 100% | 100% | 100% |

**Read side accuracy first, not within-5°.** On the axis that measures
whether the model reasons about *where* the source is, it scores 71.8% --
well above the always-centre floor (9.1%) and the strong random baseline
(40.5%, which already knows the angle grid). The model is genuinely
reading the primary spatial cue (which channel is louder) and turning it
into a direction, 97.5% of the time producing a checkable answer. That is
real signal, and it is the thing StereoMusicQA is for.

## But 71.8% hides a systematic rightward bias -- report the breakdown

The aggregate is not uniform competence. Broken down by the true side
(confusion over all 1,045 items):

| True side | n | Model correct | Where the errors go |
|---|---|---|---|
| **right** | 475 | **96.6%** | 15 unanswered, 1 called left |
| **left** | 475 | **61.3%** | 177 (37%) called *right*, 7 unanswered |
| **center** | 95 | **0.0%** | all 91 answered called *right* |

The model reliably detects a right-louder signal (96.6%), is much weaker
when the signal points left (61.3%, misfiling more than a third of
left-panned sources as right), and never once identifies a centred
source -- it calls all of them right. This is not a prompt artifact: the
centre items measure ~0.0 dB and their prompt text mostly even says "left
louder" (85 of 95 have a hair-positive level difference), and the model
says "right" anyway. It is a genuine default-to-right bias that takes over
whenever the level-difference cue is weak or points left.

So the honest one-sentence characterization is: **the zero-shot model
reads the louder-channel cue and is near-perfect when it points right,
degrades when it points left, and defaults to right when it is
ambiguous** -- real spatial reasoning with a strong, specific bias, not
uniform 72% skill.

## Why within-5° is 0.3% and why that's a footnote, not the story

Exact-azimuth accuracy is near-zero for the reason
`experiments/2026-08-10-qwen2audio-zero-shot-first-score/` and
`...-question-and-chain-of-thought/` already established and this run
confirms at scale: producing the number requires evaluating
`theta = arctan(d * tan 30 degrees)` on the measured level difference, and
a 7B model cannot do that arithmetic reliably -- it snaps to the panning
law's stated boundary (30° or, confusing the regime, 90°) instead of
computing. The verifier catches this directly, which is why `inference`
tracks `within-5°` almost exactly (0.3% each): the same failed conversion
that misses the tolerance also fails the faithfulness check.

This is a limitation of the model as a *calculator*, orthogonal to whether
it can *reason about stereo audio*, and it is exactly the gap fine-tuning
(`docs/compute.md` Phase 3, QLoRA on the 5,332 real training items) is for.
The exact degree is also already available at 100% from the front end
(`cues/level.py`), by construction -- so nothing about the benchmark or the
pipeline is blocked on the model learning to do arctan. The research
question the number answers is "can a zero-shot 7B convert the cue to a
degree in its head", and the answer is a clean no.

## The framing this run supports

For a status update: *the zero-shot audio model demonstrably reads the
primary spatial cue and identifies direction far above chance (71.8% side
vs. 40.5% for a strong random baseline), with a characterised rightward
bias; it cannot yet produce exact azimuths, which is an arithmetic
limitation the verifier catches and which motivates the fine-tuning
phase.* That is a real, defensible first result with a clear next step,
not a "the model scores 0%" dead end.

## Caveats

- **One prompt design, one disclosure (`"evidence"`), greedy decoding.**
  The rightward bias and the arithmetic failure are both stable across
  1,045 items, so they are not sampling noise, but they are specific to
  this configuration.
- **The rightward bias deserves its own follow-up.** Whether it is a
  sign-convention confusion (the prompt states "positive azimuth means
  left", the opposite of the compass intuition), a token-level prior, or
  something in how the cue text is phrased, is not established here. A
  prompt that flips or restates the convention would test it cheaply.
- **Mixture items (2 in the test split) are not in this run.** Single-source
  pipeline only, same as every model run so far.
