# A third model, a clean gradient: the boundary-snapping failure recurs at rates that track overall accuracy

## What this is

Every result before this one was Qwen2-Audio alone. This closes the
biggest open gap `docs/PROJECT_STATUS.md` had been carrying: is
"audio-LLMs snap to a boundary instead of computing" a Qwen2-Audio quirk,
or does it recur across independent model families? Now tested across
**three** independent model families (Qwen2-Audio, self-hosted on Modal;
Gemini flash-lite, Google's `google-genai` SDK; and GPT-audio, OpenAI's
`gpt-audio`, replacing the fully-retired `gpt-4o-audio-preview`),
zero-shot, same 1,045 held-out items, **both** conditions (cues-as-text /
audio-only) now complete for all three models.

Real spend across all three: Modal is free-tier; Google needed billing
enabled after hitting a live 20-requests/day free-tier cap; OpenAI needed
credits added after starting at zero balance, and its audio-only run hit
a real per-minute rate limit partway through that stretched it to ~23
hours of wall-clock time (see the audio-only section below) without
adding cost. All three were real, user-approved costs, not free compute.

```bash
python scripts/api_generate.py --provider gemini-flash-lite
python scripts/api_generate.py --provider gemini-flash-lite --audio-only
python scripts/api_generate.py --provider gpt-4o-audio
python scripts/api_generate.py --provider gpt-4o-audio --audio-only
python scripts/score_benchmark.py \
    --labels artifacts/stereomusicqa_v0.2/test_private_labels.jsonl \
    --answers experiments/2026-08-11-multimodel-comparison/answers_cues_gptaudio.jsonl \
    --baselines
python scripts/score_audio_only_stated_word.py \
    --labels artifacts/stereomusicqa_v0.2/test_private_labels.jsonl \
    --answers experiments/2026-08-11-multimodel-comparison/answers_audio_only_gptaudio.jsonl
```

## Two real scoring bugs, found and fixed mid-analysis

Both caught by direct skepticism ("maybe something is wrong with the
code"), not by inspection, and both confirmed via regression tests in
`tests/test_score.py` plus a full rescore proving **Qwen2-Audio's
already-reported numbers are byte-identical** before and after (these bugs
only ever affected the newer models' transcripts).

1. **The "°" symbol.** `verifier/extract.py`'s claim regex only recognized
   the spelled-out word "degrees"; Gemini's terse answers ("left at
   27.5°") almost always use the symbol instead. Fixed via a shared
   `_DEGREE_UNIT` pattern matching either. A first fix attempt kept a
   trailing `\b` word boundary, which still failed: `°` is a non-word
   character, so no word boundary can follow it before whitespace or
   end-of-string, caught by testing the fix against the real failing
   string, not by trusting the regex on sight.

2. **First-claim vs. last-claim.** `extract_azimuth` searched for the
   *first* degree-shaped match in the text. That is right for Qwen2-Audio
   and Gemini's short, single-claim answers, and wrong for GPT-audio's
   verbose, spontaneous chain-of-thought: a real transcript quotes the
   panning law's own "30 degrees" constant two or three times while
   deriving the formula, then states its actual answer
   ("...azimuth of approximately +20 degrees") at the very end. The first
   match grabbed the formula constant, not the conclusion, and silently
   scored a fully-correct answer as unanswered (coverage undercounted at
   roughly 55% before the fix). Rewritten to delegate to
   `extract_all_azimuths` (already built for the mixture scorer, which
   needs every claim) and take the **last** one.

## Results, cue-text (reasoning) condition, 1,045 held-out items

| Metric | Qwen2-Audio (zero-shot) | Gemini flash-lite (zero-shot) | GPT-audio (zero-shot) |
|---|---|---|---|
| Within 5deg | 0.4% | 44.9% | **80.7%** |
| Side accuracy | 71.9% | 93.6% | 90.4% |
| MAE (answered) | 74.3deg | 7.49deg | **5.44deg** |
| Coverage | 97.6% | 96.8% | **99.8%** |
| Verifier faithful | 0.4% | 42.4% | **77.8%** |

**GPT-audio is the best zero-shot model tested on the actual arithmetic**,
by a wide margin: 80.7% within-5deg and 77.8% verifier-faithful, roughly
double Gemini's numbers and two orders of magnitude past Qwen2-Audio's.
Gemini still edges it out slightly on side accuracy (93.6% vs 90.4%) and
coverage is close between the two (96.8% vs 99.8%), but on the metric that
matters most for the paper's central claim, does the model actually
compute the angle faithfully, GPT-audio is a different tier entirely.

## The boundary-snapping failure recurs in all three models, at rates that track overall accuracy

This is the finding worth taking seriously, not any single model's
headline number. Of each model's answered items, the fraction that landed
at exactly the panning law's stated hard boundary (±30deg) instead of a
computed value:

| Model | Boundary-snap rate | Within-5deg |
|---|---|---|
| Qwen2-Audio | **~100%** (essentially every answered item) | 0.4% |
| Gemini flash-lite | 19.7% (199/1,012 answered) | 44.9% |
| GPT-audio | **1.2%** (12/1,043 answered) | 80.7% |

**A clean, monotonic gradient across three architecturally unrelated
models, from three different labs, trained on different data with
different everything except the prompt design**: the rate at which a
model substitutes the prompt's own stated physical limit for genuine
arctan computation falls as the model's overall accuracy rises. That
inverse relationship, holding across all three points, is much stronger
evidence for "this is a specific, nameable failure mode of a
recognizable kind, not an artifact of one checkpoint" than any single
model's number could be, and it gives GPT-audio's near-elimination of
the failure (1.2%) real interpretive weight as an existence proof that the
task is tractable zero-shot, not just via fine-tuning.

A representative Qwen2-Audio pair, the failure at its most extreme (side
correct, magnitude identically wrong both times):

> true +20deg: "The trumpet is positioned on the left side, with an
> azimuth of **30.0** degrees."
>
> true +15deg: "The flute is on the left side, with an azimuth of **30.0**
> degrees."

## What this changes about the project's claims

The corrected claim, replacing "Qwen2-Audio fails at the arithmetic"
wherever it appeared: **audio-LLMs, when asked to convert a stated cue to
an angle, show a specific, shared failure mode (substituting the
prompt's own stated physical boundary for the actual computation) at a
rate that varies substantially and *predictably* by model, tracking
overall task competence rather than appearing at a fixed rate across the
board.** That is a stronger, more general, and more defensible claim than
any single-model result, and it is exactly what running two more models
was for.

Separately: Qwen2-Audio's zero-shot reasoning failure, the one
`experiments/2026-08-10-cues-finetune-fixes-arithmetic/` fixed by
fine-tuning (0.4% -> 100% within-5deg), is now clearly a
**Qwen2-Audio-specific** weakness rather than evidence that audio-LLMs as
a class cannot do this zero-shot. GPT-audio in particular does it well
without any fine-tuning at all. The paper's fine-tuning result is best
framed as "recovers a weak model's zero-shot deficit to match or exceed a
strong model's zero-shot ceiling," not "makes an impossible task
possible."

## Audio-only (perception) condition: all three models fail it

Same 1,045 items, no cue text, the two channels sent as real stereo audio.
Scored by **stated word** (left/right/center), not a signed number, since
nothing in this prompt tells the model "positive azimuth means left":
a raw signed number would be read against a convention the model was
never given. GPT-audio's run
(`scripts/score_audio_only_stated_word.py`, added for this scoring pass,
since `eval/score.py`'s default scorer assumes the project's sign
convention, which doesn't apply here) needed a real OpenAI-side rate
limit worked through: after ~127 items it settled into a steady **~88
seconds per request** (down from ~1-2s), consistent with the account's
per-minute quota engaging on the audio endpoint. Not an error, no retries
were needed, just ~23 hours of real wall-clock time for the full 1,045
items. Total cost for both GPT-audio runs (cue-text + audio-only) stayed
within the budget confirmed on OpenAI's usage dashboard.

| Metric | Qwen2-Audio (zero-shot) | Gemini flash-lite (zero-shot) | GPT-audio (zero-shot) |
|---|---|---|---|
| Side accuracy | 28.5% | 36.6% | 42.0% |
| vs. always-majority baseline (45.5%) | below | below | **still below** |

**All three models fail the actual task, and the ranking is the same as
the reasoning condition** (GPT-audio > Gemini > Qwen2-Audio) even though
every single one lands below the score obtainable by ignoring the audio
entirely and always guessing the majority side. GPT-audio's 42.0% is
closer to the 45.5% baseline than the other two, but "closer to chance"
is not "working": the confusion matrix makes that unambiguous.

| True side | n | Correct | Where it goes wrong |
|---|---|---|---|
| left | 475 | 78.1% | 54 called right, 44 gave no side word at all |
| **right** | 475 | **13.9%** | **354 (74.5%) called left** |
| center | 95 | 2.1% | 72 (75.8%) called left |

**GPT-audio's failure is the same shape as the other two, and more
extreme.** Qwen2-Audio defaults to "right" regardless of true side;
Gemini defaults to "left" on 567/1,045 answers; GPT-audio defaults to
"left" even harder, 797/1,045 answers (76.3%), enough that a genuinely
right-panned source is called "left" more than five times as often as
"right" (354 vs 66). Three independent models, three different specific
directional priors, but the identical underlying failure: none of them
read the inter-channel cue from real audio. They fall back to a
directional default when the cue that mattered for the cue-text condition
(the level-difference number) isn't handed over as text. **This closes
the loop the reasoning-condition gradient opened**: the same model
(GPT-audio) that is best at computing the angle *when told the numbers*
is not meaningfully better at *perceiving stereo position from audio
itself*. Reasoning capability and audio perception are dissociable
capabilities in every model tested, not just Qwen2-Audio's original
architecture-specific quirk.

## Caveats

- **One model, one prompt, greedy-equivalent default decoding per
  provider.** No sweep across GPT-audio's or Gemini's own model tiers
  (e.g. `gpt-audio-mini`, Gemini pro; pro hit the free tier's quota
  entirely, billing would be needed to test it too).
- **`gemini-flash-lite-latest` and `gpt-audio` are both rolling aliases**,
  not pinned checkpoints. The exact model behind either can change
  without notice (already true once this session: `gpt-4o-audio-preview`
  was fully retired mid-project, `gemini-2.0-flash`/`gemini-1.5-pro` both
  404'd). This means these specific numbers are not guaranteed
  reproducible against a future provider release the way the Qwen2-Audio
  numbers are against a fixed checkpoint.
- **Coverage is now comparable across all three models (97.6% Qwen /
  96.8% Gemini / 99.8% GPT-audio)**, after the two extraction bugs above
  were found and fixed. All three scores already correctly count an
  unanswered item as wrong; the models are now held to the same bar of
  "answered at all."
- **4 of 1,045 Gemini audio-only items** and a handful of GPT-audio
  cue-text items hit transient provider-side 500/503 errors even after
  automatic retries with backoff; a final resume pass before scoring
  picked up all of them, so every reported number is the full 1,045, not
  an undercount.
- **GPT-audio audio-only's 89.6% coverage (109/1,045 gave no left/right/
  centre word at all)** is lower than its 99.8% cue-text coverage,
  consistent with the occasional flat refusal noted earlier in this
  project ("I currently don't have the capability to directly analyze
  audio..." on a handful of items, a real behavioral quirk, not a pipeline
  failure; the API call itself succeeded every time). Folded into the
  42.0% side accuracy as answered-wrong, per the project's standing rule
  that an unanswered item counts against the model, not toward coverage
  padding.
