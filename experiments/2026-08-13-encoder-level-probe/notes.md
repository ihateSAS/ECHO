# Unfreezing the audio encoder is probably not what's missing

## What this is

`experiments/2026-08-11-audio-only-finetune/` failed to teach Qwen2-Audio
to hear stereo position and named "unfreeze the audio encoder" as the next
step, on the reasoning that a frozen encoder "simply doesn't expose a
usable signal to the layers the LoRA adapter can touch."

This tests that reasoning directly, for the price of ~30 minutes of laptop
CPU and no GPU at all, before anyone pays for a training run. **It does not
hold.** The cue survives the encoder. Two other things turned up on the
way, one of which invalidates how the previous run was described.

```bash
python experiments/2026-08-13-encoder-level-probe/feature_extractor_check.py
python experiments/2026-08-13-encoder-level-probe/probe.py
python scripts/make_figures.py --only encoder
```

## Finding 1: the level difference survives the log-mel front end, exactly

The first suspect was the feature extractor. Whisper's front end takes
`log10` of the mel power spectrogram and applies a dynamic-range floor
relative to *that clip's own maximum* — a per-clip relative operation — and
`WhisperFeatureExtractor` also carries a `zero_mean_unit_var_norm` that
would remove an absolute level outright. If the level difference died
there, the encoder would never see the only cue an amplitude pan has, and
unfreezing it could not possibly help.

It does not die. Whisper scales its log-spectrogram by 1/4, so a power
ratio `r` should appear as a uniform additive offset of `log10(r)/4` — a
prediction with no free parameters. Measured through the exact two-clip
call `scripts/modal_finetune.py` makes, at every level difference the v0.2
benchmark actually presents:

| planted ILD | predicted offset | measured offset |
|---|---|---|
| 2.7 dB | 0.0675 | 0.0675 |
| 8.7 dB | 0.2175 | 0.2175 |
| 19.5 dB | 0.4875 | 0.4875 |
| 36.1 dB | 0.9025 | 0.9025 |

Worst deviation across all ten: **1.3e-7**. The cue reaches the encoder
intact.

## Finding 2: it survives the encoder too, which is the point

So the real question is whether the *encoder* discards it. An encoder
trained for speech recognition has every reason to: absolute loudness is a
nuisance variable for transcription, and learning to be invariant to it is
correct behaviour for the task it was trained on.

Method: load the audio tower alone (637M parameters — it lives entirely in
shard 1 of 5, so this needs a 4 GB download, not the full 17 GB), encode
the left and right channels of real held-out URMP excerpts panned to known
angles, mean-pool each, and try to linearly decode the planted azimuth from
the difference of the two. Leave-one-excerpt-out, so the probe has to
generalise across musical content rather than memorise excerpts. The same
probe on the log-mel features is the positive control; the same probe on
labels shuffled within each excerpt is the null.

| Probe input | MAE | R² | within 5° |
|---|---|---|---|
| log-mel features (encoder input) | 1.87° | 0.983 | 100% |
| **encoder output (what reaches the LLM)** | **2.59°** | **0.953** | **89.1%** |
| encoder output, shuffled labels | 15.36° | **−0.375** | 20.9% |

**The encoder preserves the cue.** A *linear* readout of its output
recovers the planted azimuth to 2.6°, inside the project's own 5° tolerance,
generalising to excerpts it was not fitted on. The null collapses to worse
than predicting the mean, so this is signal and not a 1,280-feature probe
fitting 99 rows of noise.

The honest conclusion: **unfreezing the audio encoder targets the wrong
stage.** The information is already there, and already linear. Whatever is
failing is downstream of it.

### The caveat that matters

The probe is handed `encoder(left) − encoder(right)`. The model is not. It
receives the two channels as two separately-encoded clips, as two token
sequences, and would have to form that comparison itself across a long
context.

So the precise claim is: *the information survives the encoder and is
linearly available once the two channels are differenced*. What remains
untested is whether the language model can perform that differencing at
all. That is a much more specific failure than "the encoder loses it", and
it points somewhere different.

## Finding 3: the previous run's encoder was never frozen

Found while writing the `--encoder` flag this experiment was meant to
motivate. PEFT matches a plain `target_modules` list by **suffix**, against
every module in the model. Qwen2-Audio's audio encoder names its attention
projections `q_proj`/`k_proj`/`v_proj` exactly as the language model does.
So the config every fine-tuning run in this project has used:

```python
target_modules=["q_proj", "k_proj", "v_proj", "o_proj", ...]
```

silently attached adapters to **96 of the audio tower's 192 linear layers**
— the Q, K and V of all 32 encoder layers. Verified against PEFT's own
`check_target_module_exists`, not by reading the regex, and pinned in
`tests/test_finetune_encoder_modes.py`.

Both `experiments/2026-08-11-audio-only-finetune/notes.md` and
`docs/PROJECT_STATUS.md` described those runs as "LoRA only on the
language layers". Both are corrected. The effect on that experiment's
conclusion is to **strengthen** it — adapting most of the encoder's
attention had already failed to teach perception — but it means the
frozen-encoder condition has never actually been run, and "unfreeze the
encoder" was a narrower step than it sounded.

`scripts/modal_finetune.py --encoder {qkv,none,all,full,projector}` now
makes the choice explicit, with `qkv` the default so every previous run
reproduces byte-for-byte, and the trainable-parameter report broken down by
component so this class of error cannot recur silently.

## What to do instead

The projector. It is the single component **no run in this project has ever
trained**: it is a plain `Linear`, so no suffix in any target list has ever
matched it, and nothing has unfrozen it. It sits exactly between the
surviving cue and the language model that is not using it.

`--encoder projector` trains it with the encoder frozen. That is a far
cheaper run than unfreezing 637M encoder parameters, and it is what the
measurements above point at rather than what intuition suggested.

Beyond it, `docs/spatial_audio_models.md` arrives at the same place from
the literature: SPUR's result is that a **frozen** encoder plus a small
adapter fed inter-channel covariance is enough for spatial understanding.
For stereo, the 2x2 per-band inter-channel covariance *is* the complete
description of an amplitude pan, and this project's front end already
computes all of it. Handing the model the difference, rather than hoping it
forms one, is the intervention both this experiment and that survey point
to.

## Methodological note: the first null control was a no-op

Worth recording because it nearly shipped. The first version of the
shuffled-label control was:

```python
shuffled = targets.copy()
for group in np.unique(group_ids):
    mask = group_ids == group
    rng.shuffle(shuffled[mask])       # wrong
```

`shuffled[mask]` with a boolean mask returns a **copy**, so `shuffle`
permuted a temporary and discarded it. The "null" ran on unshuffled labels
and returned results byte-identical to the real probe — which is the only
reason it was caught. A subtler no-op would have produced a plausible
null and laundered an unvalidated result as a validated one.

It is now a tested function (`tests/test_encoder_probe_control.py`,
including an end-to-end check that the null fails on data the real probe
fits). `probe.py` also writes `embeddings.npz` beside itself, so a new
control can be tried in seconds instead of re-encoding for half an hour;
it is gitignored (regenerable, and this directory is for numbers and
conclusions, not arrays).
