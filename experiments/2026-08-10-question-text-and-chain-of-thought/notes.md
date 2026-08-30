# Fixing the question mattered a lot. Chain-of-thought made it worse.

## What this is

Two things tested against `experiments/2026-08-10-qwen2audio-zero-shot-first-score/`'s
finding (model reaches for a boundary value -- 30° or 90° -- instead of
computing the panning-law formula), on the same 30-item deterministic
sample (`scripts/modal_generate.py`'s `--limit 30`, seed fixed at 0 so all
three runs below share the identical items):

1. **A real bug in the first run**: it asked `DEFAULT_QUESTION` ("Where is
   this instrument positioned, and why?") instead of each item's actual
   benchmark question ("...Return its side and azimuth in degrees."). Now
   fixed in `scripts/modal_generate.py` to use `item["question"]`.
2. **Chain-of-thought prompting** (`--chain-of-thought`): appends explicit
   step-by-step instructions -- compute r, then d, then the arctan -- to
   the (now-correct) question, and raises `max_new_tokens` to 400 to give
   the derivation room to complete.

```bash
modal run scripts/modal_generate.py --limit 30 --answers-path answers/x.jsonl
modal run scripts/modal_generate.py --limit 30 --chain-of-thought --answers-path answers/y.jsonl
```

`labels_30item_sample.jsonl` is the matching label subset (scoring these
30 answers against the full 1,045-item label file dilutes coverage to
near-zero with ~1,015 items that were never attempted -- worth naming
explicitly since it is an easy mistake to make when reusing
`scripts/score_benchmark.py` on a partial answer set).

## Results, all three on the identical 30 items

| Condition | Within 5° | Side | MAE (answered) | Coverage |
|---|---|---|---|---|
| Original (generic question) | 0.0% | 10.0% | 24.35° | 10.0% |
| **Fixed question only** | 0.0% | **56.7%** | 72.35° | **96.7%** |
| Fixed question + chain-of-thought | 0.0% | 0.0% | 131.39° | 33.3% |

**Fixing the question was the single biggest lever pulled in this
project's model evaluation so far.** Coverage went from 10% to 96.7% and
side accuracy from 10% to 56.7%, from changing four words of prompt
plumbing, not the model or the evidence. The original run's 7.5%
system-wide coverage (`qwen2audio-zero-shot-first-score`) was measuring,
substantially, whether the model happened to volunteer a number when
never actually asked to give one in degrees -- not a stable property of
its capability.

**Chain-of-thought is not a fix here -- it is a regression**, on every
axis measured: coverage down (96.7% to 33.3%), side accuracy down (56.7%
to 0%), MAE roughly doubled. The hypothesis going in
(`experiments/2026-08-10-qwen2audio-zero-shot-first-score/notes.md`'s
"what this means" section) was that CoT would move the model off
boundary-snapping and toward real evaluation of the formula. It does move
it off boundary-snapping -- but not toward the right answer.

## Why: the model doesn't compute, it recalls "nice" trig facts

The chain-of-thought transcripts do show real, structured, step-labeled
work (`answers_fixed_question_cot.jsonl`) -- this part succeeded, the
model follows the requested three-step structure faithfully every time.
The arithmetic inside those steps is wrong in a specific, recognizable
way: it converges on textbook arctan values that have nothing to do with
the actual input.

> "azimuth = arctan(1 \* tan 30 degrees) = arctan(1 \* -1.732) = **-53.13
> degrees**"

tan(30°) is +0.577, not -1.732 (that's -tan(60°)); arctan(-1.732) is -60°,
not -53.13°. But **53.13° is arctan(4/3)**, the angle in a 3-4-5 right
triangle -- one of the handful of arctan values that appear constantly in
textbooks and is almost certainly overrepresented in training data
relative to arbitrary decimal inputs like this project's actual measured
values.

> "azimuth = arctan(-0.9921 \* tan(30)) = **63.43 degrees**"

appears for two *different* items with two *different* (but similarly
close-to-1) d values, both landing on exactly 63.43° -- **arctan(2)**,
another textbook constant, and mathematically not what arctan(-0.99 x
0.577) ≈ arctan(-0.57) ≈ -29.7° actually is. And for one item:

> "arctan(-499.5 \* tan 30 degrees) = ... = **228.6 degrees**"

which is not just wrong, it's impossible: arctan's range is (-90°, 90°)
for any real input, by definition. The model produced a number outside
the function's own range while showing a step explicitly labeled as
computing that function.

Put together with the first experiment's boundary-snapping finding, the
picture is consistent: **the model is not evaluating the arctan formula
on the actual numbers in front of it, whether asked for a direct answer
or asked to show its work.** Direct-answer mode reaches for the nearest
*physical* boundary (30° or 90°, values salient because the prompt states
them as limits). Chain-of-thought mode reaches for the nearest
*mathematically memorable* value (53.13°, 63.43°) instead. Neither is
computing; both are retrieving something adjacent to the real answer from
whatever the model associates with "arctan of a number near this
magnitude."

## What this means for the fix priority

This changes the plan from the previous experiment's notes:

1. **Ship the question-text fix regardless of anything else.** It is a
   real correctness fix (the benchmark's own question asks for degrees;
   the model should be asked that question), not a research trick, and it
   alone recovers most of the "does the model even attempt a checkable
   answer" gap.
2. **Chain-of-thought, at least this version of it, is not the next
   experiment worth funding effort into.** It made the measured failure
   mode worse, not better, and did so in an interesting way (memorized
   constants standing in for computation) that a different CoT phrasing
   might not fix, because the underlying issue -- unreliable multi-digit
   division and exponentiation inside free-text generation -- is a known,
   general LLM weakness, not specific to this prompt's wording.
3. **This sharpens, rather than removes, the case for fine-tuning
   (`docs/compute.md` Phase 3).** A model that can follow structured
   instructions (it did label every CoT step correctly) but cannot
   evaluate `10^(x/20)` reliably is a case QLoRA on real StereoMusicQA
   examples is suited for -- the derivation structure is already there;
   what is missing is calibration to the actual numbers, which is exactly
   what supervised examples teach and prompting cannot.

## Caveats

- **n=30, one sample, one seed.** Large enough to see a real, consistent
  effect (56.7% vs 0% side accuracy is not sampling noise at this
  magnitude), too small to report exact percentages as stable estimates.
  The question-text fix should be re-run over the full 1,045-item test
  set before being treated as the new baseline number.
- **`max_new_tokens=400` for CoT vs. 200 for direct-answer** -- checked
  that this isn't truncation (sampled transcripts complete their
  derivation and state a final number well under the limit), but a still
  higher limit or a different stopping criterion was not tried.
- **One CoT phrasing, one temperature (greedy), one instruction style.**
  This rules out *this* chain-of-thought prompt, not the entire idea of
  prompted step-by-step reasoning.
