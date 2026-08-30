# Dithering during fine-tuning produces a real, above-baseline signal, with severe output collapse alongside it

## What this is

The real test of Dual-BEATs' (arXiv:2607.08800) claim, following up on
`experiments/2026-08-13-dual-beats-dithering-inference/`, which found
that applying their dithering trick purely at inference to an already-
trained Qwen2-Audio didn't rescue audio-only perception. Dual-BEATs'
own reported results are much more likely to come from training with
dithering present throughout, so the model has a chance to learn to use
the variance the noise preserves, not from a frozen model seeing it for
the first time. This experiment applies dithering during fine-tuning
itself, using the same QLoRA setup as
`experiments/2026-08-11-audio-only-finetune/`, plus the exact Dual-BEATs
formula (`model/qwen_backbone.py::dither_stereo_channels`, DA=0.05,
independent Gaussian noise per channel) applied to both channels before
each training example is encoded.

```bash
modal run scripts/modal_finetune.py --condition audio_only --dither --limit 50   # validation
modal run scripts/modal_finetune.py --condition audio_only --dither             # full run, 5,332 examples
modal run scripts/modal_generate.py --audio-only --dither \
    --adapter adapters/audio_only_dithered \
    --answers-path answers/test_answers_audio_only_dithered_finetuned.jsonl
```

## A real correction found while investigating this: the encoder was never actually frozen

Before trusting this result, it's worth being clear about something this
investigation surfaced that changes how every prior fine-tuning
experiment in this project should be read.
`scripts/modal_finetune.py`'s LoRA `target_modules` list (`q_proj`,
`k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) is a
list of bare layer names, not full module paths. Qwen2-Audio's audio
encoder attention layers happen to use the identical names
`q_proj`/`k_proj`/`v_proj` that the language-model decoder's attention
layers use (only the encoder's output projection, named `out_proj`, and
its MLP, differently named than the decoder's gated MLP, are distinct).
PEFT matches by name across the whole model, not just the intended
submodule.

Verified directly, not inferred: constructed the real
`Qwen2AudioForConditionalGeneration` architecture (a small mock config,
no need for the full 8B weights) and ran the exact `LoraConfig` used in
`scripts/modal_finetune.py` through `get_peft_model`, then inspected
which modules actually received adapters:

```
Encoder projections that got LoRA adapters: ['k_proj', 'q_proj', 'v_proj']
```

So every fine-tuning run so far in this project, including the original
audio-only failure documented in
`experiments/2026-08-11-audio-only-finetune/` (now corrected), has been
adapting the encoder's attention Q/K/V the whole time. Only the encoder's
output projection and MLP stayed frozen. The claim "with a frozen
encoder... fine-tuning does not teach perception" was factually wrong as
stated, and "unfreezing the encoder" was never really an untested next
step; it was already partially happening.

**This makes today's result more informative, not less.** The original
non-dithered audio-only fine-tune had this same partial encoder access
and still collapsed to chance. That rules out "the encoder just needed
to be trainable" as the explanation for what changes below. Something
else had to be different, and the only thing that changed is the
dithering.

## Result: real above-baseline side accuracy, alongside near-total output collapse

Full 1,045 held-out items, dithered at both train and eval time
(consistent application, matching Dual-BEATs' own methodology, since a
model trained on dithered audio should also be evaluated on dithered
audio).

| Metric | Zero-shot | Non-dithered fine-tune | **Dithered fine-tune** |
|---|---|---|---|
| Side accuracy | 28.5% | 44.4% (chance in disguise) | **75.6%** |
| vs. 45.5% majority baseline | below | ~= (collapsed to "right" 953/1045) | **well above** |

75.6% is a real, large jump past the always-majority baseline. But the
raw output is severely degenerate:

```
716x  "The cello is at 5.0 degrees to the left."
329x  "The cello is at 5.0 degrees to the right."
```

**Every one of the 1,045 answers is one of exactly these two literal
strings.** The instrument name ("cello") is wrong for the vast majority
of items (the test set spans many instruments), the stated magnitude
(5.0 degrees) never varies and is essentially always wrong, and the
model never once says "centre".

## Why this isn't the same kind of collapse as the non-dithered fine-tune's fake 44.4%

The earlier audio-only fine-tune's 44.4% was chance in disguise because
it collapsed to a *single* answer ("right") regardless of the true side,
landing exactly where always-guessing-the-majority-class would. This
result is different in a way that matters: the model alternates between
exactly two answers, and *which one* it picks is clearly correlated with
the true side, not just the base rate.

| True side | n | Correct | Where it goes wrong |
|---|---|---|---|
| left | 475 | **99.6%** | 2 called right |
| right | 475 | **66.7%** | 158 called left |
| centre | 95 | 0.0% | 85 called left, 10 called right |

If this were pure default-guessing, one side would dominate regardless
of truth, the way "right" did in the non-dithered run. Instead: true-left
items get called "left" almost perfectly, and 317 of 475 true-right items
correctly get called "right", well above what saying "right" on 31.5% of
items at random would produce if the choice were arbitrary. That is a
real, above-chance signal driving the binary choice.

Centre items get no real handling: they default overwhelmingly to "left"
(85/95), consistent with a model that learned "louder channel wins,
default to left when the signal is weak or ambiguous" -- a sensible
degenerate strategy for a system that apparently never learned to
represent "centre" as a real output option at all.

## The honest interpretation

Fine-tuning with dithering appears to have let a genuine, if crude,
binary left/right signal reach the model's actual output, the exact
mechanism Dual-BEATs argues for (preserving inter-channel variance
through training so the model has something to learn from). At the same
time, something about this training run collapsed the model's broader
generative diversity entirely, instrument identity, exact magnitude, and
the centre option all vanished into a fixed two-sentence template. Both
things are true simultaneously, and the second doesn't erase the first:
a system that reliably says "left" for left-panned sources and "right"
for right-panned ones, even if it can say nothing else correctly, has
learned *something* real about the signal that the non-dithered model
never did.

This is real, independent, cross-architecture support for Dual-BEATs'
causal claim (they tested Gemma-3-1B/OLMo-3-7B with a "decoupled" dual
encoder; this is Qwen2-Audio's Whisper-style encoder, a different
architecture entirely), not just an adjacent finding. It falls well
short of their reported 97.2% ceiling, and comes with a side effect their
own reported metric (directional accuracy) might not surface: this
project's full-sentence-generation task exposes an instrument-identity
and magnitude collapse that a pure classification-style evaluation
wouldn't catch.

## What's not yet established

- **One training run, one random dithering seed.** Before this is
  trustworthy as a real, reproducible effect rather than a lucky draw,
  it needs a repeat with a different seed.
- **Why the output collapsed the way it did.** Whether this is specific
  to dithering interacting with this project's full-sentence generation
  target format, an artifact of only 1 epoch, or something else, is
  untested. A held-out validation split during training, or a shorter/
  longer schedule, might change this independently of the side-accuracy
  effect.
- **Whether the reasoning (cues) condition is affected.** This run only
  touched the audio-only condition. Dithering the (already mono)
  cues-condition audio as a control, to confirm reasoning stays
  unaffected the way it should if this is genuinely perception-specific,
  hasn't been run.
- **Generalization to unseen, continuous pan positions.** This test used
  the same held-out grid every other zero-shot/fine-tuned run in this
  project uses, not new continuous positions outside the trained angle
  set.

## Cost

Full fine-tuning run (A100-40GB, 5,332 examples, 1 epoch) plus full
1,045-item held-out evaluation (L4). Same ballpark as the original
audio-only fine-tune (~$10-12).

## Follow-up: unfreezing the rest of the encoder is a regression, not an improvement

The correction above raised an obvious next question: the encoder's
attention Q/K/V was already being adapted by accident, but its output
projection, MLP, and normalization layers (the exact layers Dual-BEATs
argues erase inter-channel variance) were still fully frozen. Would
deliberately unfreezing those too, on top of dithering, let the model do
more with the preserved variance?

```bash
modal run scripts/modal_finetune.py --condition audio_only --dither --unfreeze-encoder --limit 50   # validation
modal run scripts/modal_finetune.py --condition audio_only --dither --unfreeze-encoder             # full run
```

`--unfreeze-encoder` extends the LoRA `target_modules` to also cover the
encoder's `out_proj`, `fc1`, `fc2` (verified not to collide with any
decoder layer name, unlike q/k/v; see
`tests/test_finetune_lora_targeting.py`), and explicitly sets
`requires_grad=True` on the encoder's LayerNorm parameters directly
(LoRA doesn't apply to LayerNorm, which isn't a plain linear layer). This
took trainable parameters from 43.9M to 51.9M, confirmed via the training
log (`also unfroze 166,400 encoder LayerNorm params`).

**Result: total collapse. Worse than the partial-encoder run, and the
same failure shape as the original non-dithered fine-tune.**

| | Zero-shot | Fine-tune (no dither) | **Dither, partial encoder** | **Dither, encoder unfrozen** |
|---|---|---|---|---|
| Side accuracy | 28.5% | 44.4% (chance) | **75.6%** | **45.5% (chance)** |
| Unique answers (of 1,045) | many | 2 (mostly "right") | 2 ("cello... left/right") | **1** ("cello... right", always) |

Every single one of the 1,045 answers is the identical string, "The
cello is at 5.0 degrees to the right." Not left-or-right depending on
anything, not even the crude but real 99.6%/66.7% asymmetric split the
partial-encoder run showed, just one constant answer regardless of truth.
Side accuracy (45.5%) lands exactly on the always-majority baseline,
because that is exactly what happened: pure default-guessing, the same
shape of failure as the original non-dithered fine-tune, just even more
extreme (100% constant vs. that run's 91%).

**This is a real regression, not a null result.** The partial-encoder
run's binary left/right signal was genuinely above chance. Giving the
model more trainable capacity, on the exact same 5,332-example, 1-epoch
training budget, didn't refine that signal into something richer
(correct instrument, correct magnitude) -- it destroyed it, collapsing
all the way back to constant-output guessing. More capacity without more
data or a longer schedule made this worse, not better.

**What this narrows down, from the three live hypotheses going in:**

1. *"No real spatial signal reaches the output, dithering is inherently
   crude"* -- still plausible for why instrument identity and magnitude
   never recovered in either dithered run, but doesn't explain why the
   *directional* signal specifically got worse with more capacity, since
   dithering itself didn't change between these two runs.
2. *"Decoder-side habit, not encoder-limited"* -- gets real support: the
   identical literal template across both dithered runs (differing only
   in whether "left" ever appears) suggests the language-model side had
   already settled into a fixed response shape that the encoder's
   capacity wasn't the binding constraint on.
3. *"Regression from broader adaptation on too little data"* -- this is
   what the numbers actually show happened. More trainable parameters
   with the same small budget didn't help the model do more; it made
   collapse to a single constant answer *easier* to fall into, not
   harder.

The honest reading: dithering unlocked a real, crude, above-baseline
signal in a narrowly-adapted model. Unfreezing more of the encoder on
the same training budget didn't build on that; it erased it. Whether
more data, more epochs, or a smaller learning rate would let the larger
trainable surface actually help is untested and would need a real
follow-up, not assumed.

## Second follow-up: 3x the training budget doesn't recover it either

The obvious next question the regression above raised: was the collapse
specifically because the larger trainable surface (encoder Q/K/V + out_proj
+ MLP + LayerNorms) was undertrained relative to its size, and would more
epochs on the full training set let it actually use that capacity?

```bash
modal run scripts/modal_finetune.py --condition audio_only --dither --unfreeze-encoder --limit 200 --epochs 3   # validation
modal run scripts/modal_finetune.py --condition audio_only --dither --unfreeze-encoder --epochs 3               # full run, 15,996 steps
```

The cheap 200-example validation was genuinely encouraging: a clean,
mostly monotonic loss decline from 1.90 to 0.15 over 600 steps, nothing
like the bouncing every other audio-only run (dithered or not) has shown.
That pattern did not hold at full scale. The full run (5,332 examples,
3 epochs, 15,996 steps) went back to bouncing for most of its length
(roughly 0.17-1.08 through the visible middle and end of training), the
same qualitative shape as every 1-epoch run.

**Result: still no real signal, though the collapse looks less extreme
on the surface.**

| Configuration | Side accuracy | Unique answers (of 1,045) | Real signal? |
|---|---|---|---|
| Zero-shot | 28.5% | many | No |
| Fine-tune, no dither | 44.4% | 2 (mostly "right") | No |
| **Dither, partial encoder (accidental)** | **75.6%** | 2 | **Yes** |
| Dither, encoder unfrozen, 1 epoch | 45.5% | 1 | No |
| Dither, encoder unfrozen, 3 epochs, full data | 46.4% | 2 | No |

The last row looks superficially better than the row above it (two
templates instead of one, closer-sounding numbers), but the confusion
matrix rules that out: the model said "right" 71.9% of the time overall,
and true-right items were "correct" 72.6% of the time, true-left items
29.5% of the time -- both essentially identical to what the model's own
marginal answer rate predicts on its own, independent of the actual
input. That is the signature of a model whose output doesn't depend on
the audio at all, just dressed up as two templates instead of one. Contrast
with the partial-encoder run's real signal, where true-left hit 99.6% and
true-right hit 66.7%, both far from what the model's own ~68.5%-left
answer rate would predict by chance -- that one was genuinely
discriminative; this one isn't.

## What this settles, across every configuration tried

The only setup that produced a real, above-chance, input-dependent signal
was the original, accidental one: dithering, with just the encoder's
attention Q/K/V trainable (not by design; a naming collision). Every
deliberate attempt to improve on it, unfreezing more of the encoder at
the same training budget, unfreezing more of the encoder *and* tripling
the training budget, made it worse, not better, landing back at
chance-level performance dressed up in different degenerate output
shapes each time.

**The honest conclusion:** the dithering effect is real (independent,
cross-architecture support for Dual-BEATs' claim, confirmed once), but
fragile -- sensitive to exactly how much of the model is trainable in a
way that isn't well understood and doesn't respond straightforwardly to
"more capacity" or "more training." This matters for how strongly the
75.6% result should be cited: as a genuine existence proof that the
mechanism works, not as a stable, reproducible recipe. Reproducing it
even with a different random seed on the *original* configuration (never
done) remains the most direct way to know whether 75.6% itself is solid
ground, before any further variation on top of it.

## Cost, both follow-ups

1-epoch encoder-unfrozen run: ~$10-12 (same ballpark as prior full runs).
3-epoch encoder-unfrozen run (200-example validation plus the full
5,332-example, 15,996-step run, plus a second full 1,045-item held-out
evaluation): a further ~$25-35. Total real spend across the whole
dithering + encoder-unfreeze arc, all three fine-tuning attempts plus
their evaluations, is in the ~$45-60 range.

## Seed and eval-noise replication, added 2026-08-13

The open question at the end of the section above: does 75.6% hold on a
different training seed, and does it depend on the specific (unseeded)
noise draw each evaluation used? Both were untested until now.

Repeated the original configuration exactly (`--condition audio_only
--dither`, no `--unfreeze-encoder`), fixing `--dither-seed 1` for the
training-time noise. `scripts/modal_generate.py` did not previously
accept a seed at all -- each item's dithering noise was drawn fresh via
`np.random.default_rng()` with no way to reproduce it, so two evaluations
of the same adapter would each see different dithered audio. Added
`--dither-seed` there too, threading one `Generator` through every item
in order, so a fixed seed reproduces the identical noise sequence.

Full run: 5,332 examples, 1 epoch, 5,332 steps, saved to
`adapters/audio_only_dithered_seed1`. Evaluated both this adapter and the
original `adapters/audio_only_dithered` against the same held-out 1,045
items, under the same fixed evaluation-noise seed (101) -- only the
training seed differs between them.

| | Original | Seed 1 |
|---|---|---|
| Side accuracy | 75.6% | **77.1%** |
| True-left correct | 99.8% | 98.7% |
| True-right correct | 66.5% | 70.9% |
| True-center correct | 0.0% | 0.0% |
| Output templates | 2 | 3 (one stray "centred") |

Same shape under a different training seed: near-perfect on true-left, real
but weaker signal on true-right, complete failure on centre, the same
fixed "cello, 5.0 degrees" template regardless of the true instrument or
angle magnitude, only the side word tracking the input.

**This settles the open question.** 75.6% is not a fluke of one lucky
training run or one lucky unseeded noise draw; it reproduces under a
different training seed and under fixed evaluation noise, with the same
asymmetric confusion pattern both times.
