# Can the model hear stereo? No: audio-only localization is below chance

## The question this settles

Every model run before this one handed the model the spatial cues **as
text** (`describe_cues`) and the audio **as a mono downmix**. So the 71.8%
side accuracy in `experiments/2026-08-10-qwen2audio-corrected-question-full/`
measured the model *reasoning from numbers it was given*, not *perceiving
space from audio*. This experiment removes the text cues entirely and gives
the model the actual two channels as audio, to test the stronger claim:
**can Qwen2-Audio localize a source from the stereo signal itself?**

The answer is no, clearly and at scale.

## How it was run

Qwen2-Audio's Whisper feature extractor has no channel axis, so an
interleaved `(samples, 2)` stereo array cannot be passed (it gets read as
garbage, `probe_channel_handling`). The only way the real per-channel
audio reaches the model is as **two separate audio clips**: left as Audio
1, right as Audio 2, with a one-line preamble saying which is which and
**no measured cue values at all** (`model/qwen_backbone.py::answer_from_stereo`,
`scripts/modal_generate.py --audio-only`).

This is a fair test of what the architecture allows, and the level cue does
survive the encoder: a 20 dB amplitude difference is a clean offset in the
log-mel features, not normalized away
(`tests/test_qwen_backbone.py::test_feature_extractor_preserves_level_difference`).
So the information the model would need is present in what it receives; the
question is only whether it can use it. Full 1,045 held-out items, bf16,
greedy, on Modal's free tier ($0).

```bash
modal run scripts/modal_generate.py --audio-only \
    --answers-path answers/test_answers_audio_only.jsonl
```

## Results, 1,045 held-out items

| Condition | Side accuracy | Coverage |
|---|---|---|
| **Audio-only (no cue text)** | **28.5%** | 98.1% |
| always-guess-majority-side (reference) | 45.5% | 100% |
| Cue-text condition (same model, same items) | **71.8%** | 97.5% |

**28.5% is below the majority-class baseline (45.5%).** The model answers
almost every item (98.1% coverage); it is not refusing, it is
confidently wrong. Removing the text cues does not just lower the score, it
drops it *below what you get by ignoring the audio and always guessing the
most common side.* The 71.8% was the text cues doing the work; the audio
contributes nothing usable.

## The confusion matrix is the real evidence: it can't tell left from right

By true side, what the model said:

| True side | n | -> left | -> right | -> center | correct |
|---|---|---|---|---|---|
| left | 475 | 194 | 63 | 202 | 40.8% |
| **right** | 475 | **203** | **65** | 204 | **13.7%** |
| center | 95 | 42 | 13 | 39 | 41.1% |

The decisive row is **true-right**: the model calls a right-panned source
"left" (203) *three times more often* than "right" (65). It is not merely
at chance on right sources, it is **anti-correlated**, actively wronger
than a coin flip. Its stated-side distribution over all items is centre
445 / left 439 / **right 141**: it almost never says "right" at all (13%
of answers), though 45% of the items truly are. The model is not reading
inter-channel level; it is defaulting to left/centre and guessing, and the
guess is uncorrelated with (slightly opposed to) the truth.

For contrast, the cue-text condition got 96.6% on exactly these right
sources, because it was told "N dB, right louder" in words. Hand it the
number: 96.6%. Make it listen: 13.7%. That gap is the whole finding.

## What this means: perception failure, cleanly separated from reasoning failure

This is the missing half of a two-part characterization that StereoMusicQA's
front-end-plus-verifier design makes measurable and that these experiments
now support end to end:

1. **Perceptual failure (this experiment).** The audio-LLM cannot recover
   stereo position from the audio. Two independent reasons, both shown:
   the encoder architecturally collapses stereo (mono feature extractor,
   `probe_channel_handling`), and even via the two-clip workaround that
   does preserve the level cue, localization is below chance (28.5%,
   anti-correlated on right).
2. **Reasoning failure (the earlier runs).** Given perfect cues *as text*,
   the model gets direction largely right (71.8% side) but cannot produce
   the exact azimuth: it fabricates arithmetic, snapping to boundary or
   memorized-constant values the verifier catches
   (`experiments/2026-08-10-qwen2audio-corrected-question-full/`,
   `...-question-text-and-chain-of-thought/`).

Separating these two is the contribution. "The model scores badly on
spatial audio" conflates an encoder that throws away the signal with a
reasoner that can't do arithmetic; this benchmark pulls them apart and
says which one failed, per item.

## Honest caveats

- **One model, one method, one prompt.** The two-clip framing is forced by
  the mono encoder and asks the model to compare two separately-encoded
  clips, which is harder than ingesting a true stereo stream, but no
  audio-LLM we can run ingests true stereo, which is itself part of the
  point. A different prompt might help at the margin; a below-majority,
  anti-correlated result is not a near-miss that prompt-tuning rescues.
- **This is Qwen2-Audio specifically.** The claim that *audio-LLMs in
  general* can't localize needs the multi-model sweep (GPT-4o-audio,
  Gemini, SALMONN, Audio-Flamingo via the cohort's API access). That sweep
  is the single change that turns this from "one model can't" into a
  field-level finding, and it is the next experiment worth running.
- **Not a claim that the task is impossible.** The front end recovers the
  angle exactly from the same audio; the information is there. This is a
  claim about what today's audio-LLMs, as shipped, actually do with it.
