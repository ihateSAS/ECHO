# The first real model score, and it's below the floor

## What this is

Every number in this project before today was a baseline, a front-end
measurement, or a scoring-pipeline sanity check -- never a model actually
answering. This is that: Qwen2-Audio-7B-Instruct, zero-shot, bf16, run
over all 1,045 single-source held-out test items from StereoMusicQA v0.2,
under the default `"evidence"` disclosure (the model is given the
measured cues and the panning law, and has to derive the regime and the
azimuth itself -- see `model/qwen_backbone.py::describe_cues`).

Run on Modal (`scripts/modal_generate.py`), an L4 GPU, on the program's
free monthly credit -- not a rented A100, and not a training job, per
`docs/compute.md`'s 2026-08-10 update on why. Cost: $0. Wall clock: ~40
minutes for all 1,045 items after the model was warm (~1.5-4s/item), far
under the CPU-based worst case (`docs/compute.md`'s 60-day estimate) and
also far under the original GPU estimate (4-11 hours) -- generation on a
real GPU is simply much faster than the pessimistic planning numbers
assumed, which were bracketing an unmeasured throughput on purpose.

```bash
python scripts/score_benchmark.py \
    --labels artifacts/stereomusicqa_v0.2/test_private_labels.jsonl \
    --answers experiments/2026-08-10-qwen2audio-zero-shot-first-score/answers.jsonl \
    --baselines \
    --json experiments/2026-08-10-qwen2audio-zero-shot-first-score/result.json
```

## Results, 1,045 held-out items

| Condition | Within 5° | Side | MAE (answered) | Coverage | Inference passed |
|---|---|---|---|---|---|
| **Qwen2-Audio-7B (zero-shot)** | **0.6%** | **5.6%** | 64.80° | **7.5%** | 0.6% |
| no answer | 0.0% | 0.0% | — | 0% | 0.0% |
| always centre | 23.6% | 9.1% | 14.13° | 100% | 14.5% |
| random from grid | 23.5% | 40.5% | 18.77° | 100% | 18.9% |
| front end (ceiling) | 100.0% | 100.0% | 0.00° | 100% | 100.0% |

**The model scores below every floor, including "no answer" is not true --
it scores below always-centre and below random-from-grid, which is a
stronger statement than "worse than expected."** Side accuracy (5.6%) is
below chance on a mostly-binary question and far below random-from-grid's
40.5%. Coverage -- whether the model stated anything a parser could grade
at all -- is 7.5%: 92.5% of the time it either didn't answer the question
in a checkable form or its answer didn't survive the same extraction path
the verifier uses.

## Why: the model doesn't interpolate, it recognizes extremes

This is the actual finding, and it is not noise. Every one of the 78
items that did extract to a parseable azimuth landed on one of five
values:

| Extracted value | Count | What it means |
|---|---|---|
| -90.0° | 39 | "90 degrees" attributed to the right |
| +30.0° | 20 | "30 degrees" attributed to the left |
| +90.0° | 17 | "90 degrees" attributed to the left |
| 68.9°, 60.0° | 1 each | Stray, no clear pattern |

Not one of the 78 stated an intermediate angle -- nothing like "18
degrees" or "22.4 degrees" ever appears, despite the prompt handing over
the exact formula (`theta = arctan(d * tan 30 degrees)`) and the exact
numbers needed to evaluate it. A representative transcript
(`smq_test_00860_absolute`, true azimuth +25°):

> "The instrument is positioned facing towards the left, as indicated by
> the positive azimuth value of 30 degrees. This is because the
> inter-channel level difference is 19.5 dB and the left channel is
> louder..."

19.5 dB is the correct, real level difference for a source at -25°/+25°
magnitude (matches `cues/level.py::ild_db_for_azimuth(25.0)` to the tenth
of a dB) -- the front end handed over the right number. The model reads it
correctly (louder channel, rough magnitude), recognizes "this is a strong
pan," and reaches for **30°, the hard boundary the prompt's own panning
law states** (`|theta| <= 30 degrees`), rather than doing the arithmetic
that would actually put it at 25.0°. It is pattern-matching to "this is
about as panned as amplitude panning gets" instead of evaluating the
formula it was given.

The 90° cases are a different and more specific confusion. A
representative transcript (`smq_test_00479_absolute`, true azimuth -15°,
an amplitude pan, not delayed):

> "The instrument is positioned facing towards the right, as indicated by
> the positive azimuth value of 90 degrees. This is because the
> inter-channel level difference is negative on the right side (-8.7 dB)..."

Two things are wrong here, both diagnostic. First, 90° is not this
project's amplitude-panning limit (30°) -- it's the *binaural* limit
(`spatialize/binaural.py`'s Woodworth relation, `CLAUDE.md`: "Range here
is ±90°, unlike amplitude panning's ±30°"), and this item was never
delayed; the model reached for the wrong regime's boundary. Second, "the
positive azimuth value of 90 degrees" attached to *"facing towards the
right"* directly contradicts the sign convention stated two sentences
earlier in its own prompt ("Positive azimuth means left") -- the model
states the rule and then violates it in the same breath. This is exactly
the kind of self-contradiction `verifier/` was built to catch, and did:
`inference_passed` on this item is 0.

## What this means for the project, not just the number

This is the first time the `"evidence"` vs. `"conclusions"` disclosure
distinction (`model/qwen_backbone.py::describe_cues`) has been tested with
real generation at scale rather than argued for in a docstring. The
prediction it was built to test -- that handing over conclusions lets a
model pass by transcribing its input, while withholding them measures
whether it can actually do the conversion -- holds up: zero-shot, it
mostly cannot. That's a real, negative, and useful result, not a failure
of the experiment. It says the benchmark is measuring something real (a
capability gap), not something trivial (a formatting exercise), and it
gives fine-tuning an actual target: right now the model has the right
*numbers* (the 19.5 dB and -8.7 dB readings above are both correct reads
of the prompt) and the right *qualitative direction* most of the time, and
still fails the quantitative step almost completely.

## Caveats

- **This is one prompt design, one disclosure condition, one decoding
  setting** (greedy, `max_new_tokens=200`). Whether `"conclusions"` mode
  (which states the answer) or a few-shot prompt changes this is unknown
  and untested here.
- **Coverage this low partly reflects the extractor, not only the model.**
  `verifier/extract.py::extract_azimuth` needs an explicit "N degrees"
  claim; a model that correctly identifies "far left" without ever
  converting to a number is graded as unanswered, same as one that said
  nothing. That's the right scoring rule (an unchecked claim isn't a
  graded claim), but it means 7.5% coverage is a floor on "the model tried
  to be quantitative," not a ceiling on "the model understood the
  question."
- **Mixture items (2 in the test split) were not included in this run** --
  `scripts/modal_generate.py` only covers the single-source pipeline so
  far; `eval/mixture_score.py` exists but has no model answers to score
  yet.
