# Dithering at inference alone doesn't rescue perception (as expected, and informative anyway)

## What this is

A test of whether Dual-BEATs' (arXiv:2607.08800) dithering trick,
applied purely at inference time to an already-trained Qwen2-Audio,
changes anything about the audio-only perception failure. Dual-BEATs
adds independent Gaussian noise to each stereo channel before encoding,
intended to survive normalization layers that would otherwise erase the
inter-channel variance a spatial cue depends on:

```
W_L' = W_L + N_L * 0.05,  W_R' = W_R + N_R * 0.05,  N_L, N_R ~ N(0, 1) independent
```

Implemented exactly as `model/qwen_backbone.py::dither_stereo_channels`
and wired into `answer_from_stereo(dither=True)` and
`scripts/modal_generate.py --dither`. Qwen2-Audio's own audio-only path
already routes left and right independently through the same encoder as
two separate clips, the same shape Dual-BEATs' own dual-encoder setup
uses, so applying their exact intervention here is a direct test, not an
approximation.

```bash
modal run scripts/modal_generate.py --audio-only --dither --limit 50
```

## Result: no real change in accuracy, a real change in which default bias the model falls back to

Scored by stated word, matched to the same 50 held-out items on both
sides (the zero-shot number below is the existing
`experiments/2026-08-10-audio-only-stereo-perception/` run filtered down
to these exact 50 item_ids, not the full-1,045 number, for a fair
apples-to-apples comparison).

| | Zero-shot (undithered) | Dithered |
|---|---|---|
| Side accuracy (n=50) | 32.0% | 36.0% |

A 4-point difference on n=50 is within noise (one item flipping moves the
number by 2 points), so this alone doesn't support "dithering helped."
The confusion matrix is more informative:

| | Zero-shot | Dithered |
|---|---|---|
| Stated-side distribution | centre=19, left=25, right=6 | left=13, **right=29**, centre=6 |
| True-left correct | 50.0% | 20.0% |
| True-right correct | 17.4% | **60.9%** |
| True-centre correct | 28.6% | 0.0% |

The model didn't start genuinely perceiving stereo position. It flipped
which single direction it defaults to: zero-shot leans left/centre,
dithered leans right almost exclusively (29/50 answers). True-right
accuracy jumped because it now guesses "right" most of the time; true-left
and true-centre accuracy dropped correspondingly. This is the same shape
of result as the earlier few-shot pilot
(`experiments/2026-08-12-audio-only-fewshot-pilot/`): an intervention
that changes which default bias the model reaches for, not one that
unlocks real use of the audio signal.

## Why this doesn't refute Dual-BEATs, and what it does establish

Dual-BEATs' reported 97.2% accuracy is very plausibly from models
*trained or fine-tuned* with dithering present throughout, using it as a
regularizer the model learns to exploit, not from a frozen pretrained
model seeing dithered input for the first time at inference. Qwen2-Audio
was never trained with this signal preserved through its normalization
layers, so there's no reason to expect it already knows what to do with
the extra variance dithering exposes, even if that variance is now
technically present in what reaches the encoder.

What this pilot does establish: the negative audio-only perception
result is *not* fixed by a cheap, training-free intervention applied at
inference. Combined with the earlier few-shot result, this is now two
independent interventions (in-context examples, input-level dithering)
that both fail to rescue zero-shot perception the same way, without
touching the model's weights. That's further evidence for "no usable
signal reaches the point an answer is produced" over any explanation
that a small nudge at inference time should have been enough.

## What's next

The real test of Dual-BEATs' actual claim is dithering applied
*throughout fine-tuning*, not just at inference, so the model has the
chance to learn to use the preserved variance the way Dual-BEATs' own
models presumably did. See
`experiments/2026-08-13-dual-beats-dithering-finetune/` (once it exists)
for that result.

## Caveats

- n=50, not the full 1,045 -- a pilot, not a final number, same
  discipline as every other "measure before committing" step in this
  project.
- One dithering amplitude (DA=0.05, the paper's own default). Different
  amplitudes weren't swept.
- Free-tier Modal compute (L4, zero-shot, no adapter); no real cost
  incurred by this pilot.
