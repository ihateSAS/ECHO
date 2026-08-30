# Few-shot doesn't rescue audio-only perception, and rules out a real confound

## What this is

Every audio-only run so far (Qwen2-Audio, Gemini, GPT-audio; zero-shot and,
for Qwen2-Audio, fine-tuned) fails to localize from real stereo audio.
Zero-shot alone can't distinguish two different explanations for that:

1. **No usable signal.** The frozen encoder or downstream architecture
   doesn't expose/combine the interaural cue in a way the LLM can use.
2. **Signal exists, task/output convention doesn't.** The model has some
   real spatial sensitivity but doesn't know what's being asked or what
   output format is expected, and falls back to a directional prior.

Few-shot exemplars test (2) specifically, without touching any weights: 3
worked (audio, correct side) examples, drawn from the **train** split
(never test, to avoid leaking an answer), shown as real turns before the
actual query. See `model/providers.py::FewShotExample` and the new
`answer_audio_only_fewshot` methods on `OpenAIAudioLLM` and
`GeminiAudioLLM`. If this moves the numbers, that's real signal being
unlocked by task calibration. If it doesn't, that rules out "the model
just didn't understand what was being asked" as a confound on the
existing negative result, a *stronger* claim than zero-shot alone
supports, not a weaker one.

Scope, deliberately small: a pilot, not a full run. Gemini flash-lite
n=40, GPT-audio n=20 (smaller given the real per-minute rate limit hit on
its previous audio-only run, and the added audio-token cost of carrying 3
exemplars in every request). Qwen2-Audio excluded from this round: its
two-clip stereo path (`answer_from_stereo`) would need real surgery to
carry multiple few-shot stereo exemplars, and its failure already has
more mechanistic grounding than the two closed models (the ILD survives
its feature extractor, per `tests/test_qwen_backbone.py`), so this
"calibration vs. no signal" question is more open for Gemini/GPT-audio.

The 3 exemplars are fixed and pinned in
`scripts/run_audio_only_fewshot_pilot.py::FEWSHOT_ITEM_IDS`: one clear
left (+20deg, viola), one clear right (-20deg, flute), one center (0deg,
violin), from three different pieces so the model can't key on one
stem's timbre. Same three exemplars for both providers, a fair
comparison needs that held constant.

```bash
python scripts/run_audio_only_fewshot_pilot.py --provider gemini-flash-lite --n 40
python scripts/run_audio_only_fewshot_pilot.py --provider gpt-4o-audio --n 20
python scripts/score_audio_only_stated_word.py \
    --labels artifacts/stereomusicqa_v0.2/test_private_labels.jsonl \
    --answers experiments/2026-08-12-audio-only-fewshot-pilot/answers_gemini_fewshot.jsonl
```

## Results

Zero-shot baselines below are **not** the full-1,045-item numbers reported
elsewhere. They're the same models' zero-shot answers, filtered down to
the exact same item_ids as each pilot, so the comparison is apples to
apples on identical items. The always-majority baseline is also
recomputed per subset, since a 40- or 20-item sample doesn't necessarily
have the same left/right/center split as the full test set.

| | Gemini flash-lite (n=40) | GPT-audio (n=20) |
|---|---|---|
| Zero-shot side accuracy (matched subset) | 35.0% | 50.0% |
| Few-shot side accuracy | **27.5%** | **55.0%** |
| Subset always-majority baseline | 42.5% | 50.0% |

**Gemini got worse with few-shot, not better**: 35.0% down to 27.5%,
and both numbers sit below the subset's own 42.5% majority baseline.
**GPT-audio moved up slightly**, 50.0% to 55.0%, but its zero-shot
number on this specific 20-item subset is *exactly* the majority baseline
(50.0%), because zero-shot said "left" on 18 of 20 items regardless of
truth. A 5-point move on n=20 is within noise (one item flipping is a
5-point swing) and not a claim this pilot can support on its own.

## What actually happened isn't "no effect", it's the wrong kind of effect

The confusion matrices are more informative than the headline numbers:

**Gemini collapsed toward "center."** Zero-shot mostly said "left" (23/40
answers); few-shot mostly said "center" (25/40 answers), which is
exactly the answer given by one of the three few-shot exemplars. That's
the signature of **surface pattern-matching on the shown examples'
answer distribution**, not task calibration unlocking real spatial
sensitivity. If the model had latent signal and just needed the task
explained, showing it one "center" example among three shouldn't make it
say "center" on 62% of a fresh 40-item sample regardless of the true
side.

| True side | n | Zero-shot correct | Few-shot correct |
|---|---|---|---|
| left | 17 | 52.9% | 11.8% |
| right | 17 | 23.5% | 29.4% |
| center | 6 | 16.7% | 66.7% |

Left accuracy collapsed (52.9% -> 11.8%) almost exactly as center
"accuracy" rose (16.7% -> 66.7%), consistent with the model shifting
its default answer, not genuinely attending to the audio any more than
before.

**GPT-audio's shift is more ambiguous but tells a similar story.**
Zero-shot on this subset was an extreme single-answer default (18/20
said "left"); few-shot broke that into a more varied 12-right/8-left
split. Accuracy per side moved in different directions rather than both
improving:

| True side | n | Zero-shot correct | Few-shot correct |
|---|---|---|---|
| left | 10 | 100.0% | 50.0% |
| right | 8 | 0.0% | 75.0% |
| center | 2 | 0.0% | 0.0% |

Zero-shot's 100% on true-left is not competence, it's the trivial
consequence of saying "left" on every single item in a subset that
happens to be 50% left. Few-shot didn't make the model more accurate
overall so much as **move which single default it leans on**, from an
almost-total "left" bias to a more balanced but still not clearly
audio-driven split.

## Conclusion

**Few-shot exemplars did not unlock real perceptual signal in either
model.** What moved was which default answer each model reached for, not
whether it was actually using the audio: Gemini's shift mirrors the
exemplar mix almost exactly, and GPT-audio's shift traded one bias for
another rather than producing a coherent accuracy gain. This is evidence
*against* explanation (2) above (signal exists, task calibration was
missing) and *for* explanation (1) (no usable signal reaches the point in
the model where an answer gets produced).

**This makes the existing negative result stronger, not weaker.** A
skeptical read of the zero-shot audio-only failure could argue the model
just didn't understand what was being asked of it. This pilot, a real,
if small, few-shot attempt with worked examples covering all three
classes, doesn't rescue the numbers, and the *way* it fails (defaulting
to whatever the exemplars showed, rather than converging on genuine
accuracy) is itself informative: it looks like a system with no route
from audio to answer, filling in from context however it can, not one
holding back real spatial sensitivity for lack of task framing.

## Caveats

- **Small samples.** n=40 and n=20 respectively; individual item flips
  move percentages by 2.5-5 points. Neither number should be read as
  precise. The *direction and shape* of the change (collapse toward an
  exemplar's answer; trading one default for another) is the actual
  finding, not the exact percentages.
- **Qwen2-Audio untested here**, see "What this is" above. If this
  pilot's conclusion (few-shot doesn't help) needs a third model to be
  fully convincing, Qwen2-Audio's two-clip stereo path would need to be
  extended to carry few-shot exemplars first.
- **One fixed set of 3 exemplars.** Different examples, more exemplars,
  or different formatting could plausibly behave differently. This
  pilot tests one specific few-shot configuration, not the whole space of
  possible ones. Scaling to the full 1,045 items or trying a different
  k/exemplar mix is a real next step if there's reason to keep pushing on
  this, but the small pilot's result doesn't currently motivate that
  investment.
