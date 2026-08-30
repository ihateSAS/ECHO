# Fine-tuning fixes reasoning completely. It does not touch perception.

## What this is

The keystone experiment for whether BinauralCoT's positive thesis
("audio-LLMs can be trained to produce faithful spatial reasoning")
extends to genuine audio perception, not just reasoning from cues handed
over as text. Same QLoRA setup as
`experiments/2026-08-10-cues-finetune-fixes-arithmetic/` (4-bit NF4 base,
LoRA r=16 on the LLM's attention/MLP projections, 1 epoch, Modal
A100-40GB), trained on the **audio_only** condition instead: no cue text,
target is just the correct placement (`model/finetune_data.py::audio_only_target`),
audio is the two channels as two clips (`answer_from_stereo`).

```bash
modal run scripts/modal_finetune.py --condition audio_only
modal run scripts/modal_generate.py --audio-only --adapter adapters/audio_only \
    --answers-path answers/test_answers_audio_only_finetuned.jsonl
```

## The training loss already told the story

The cues fine-tune's loss collapsed smoothly to ~0.0001 within a few
hundred steps. This one never converged. It dropped from 1.59 to a
plateau around 0.18-0.20 in the first few hundred steps and then bounced
(0.18 to 0.94) for the rest of the run, never settling. That is the
signature of a model that cannot fit the training targets from the given
input, not one that is slowly learning them. (A partial confound: the
audio-only target names the instrument, e.g. "The trumpet is at 25.0
degrees to the right," while the audio-only *prompt* never tells the model
which instrument it's hearing, so some of that irreducible loss is the
model having no way to predict the instrument name, not the position. It
does not explain a loss that never trends down at all.)

## Results, 1,045 held-out items

| Metric | Zero-shot audio-only | Fine-tuned audio-only | Always-majority baseline |
|---|---|---|---|
| Side accuracy | 28.5% | **44.4%** | 45.5% |

**44.4% looks like an improvement over the zero-shot 28.5% and is not
one.** It is statistically indistinguishable from always guessing the
majority class (45.5%), because that is exactly what the fine-tuned
model collapsed to doing:

| True side | n | Stated "right" | Stated "left" |
|---|---|---|---|
| left | 475 | 437 (92%) | 38 |
| right | 475 | 426 (90%) | 49 |
| center | 95 | 90 (95%) | 5 |

**953 of 1,045 answers say "right."** The model did not learn to
distinguish left from right, up to or even toward chance. It learned
that "right" is the training target's single most common word (their
distribution is genuinely close to a coin flip by construction, so this
isn't even a class-imbalance artifact) and emits it almost unconditionally
for every real input distribution. This is a stronger, more diagnostic
failure than the zero-shot run's anti-correlation
(`experiments/2026-08-10-audio-only-stereo-perception/`): that one at
least varied its answer with *something* in the input; this one collapsed
to a near-constant.

## What this establishes, and what it doesn't

**Established:** with this specific setup -- the two channels as two
independently-encoded clips, QLoRA, 5,332 examples -- fine-tuning does not
teach Qwen2-Audio to perceive stereo position zero-shot. The failure is not
"needs more epochs" (the loss plateau is the tell) or "needs more data"
(5,332 examples was enough to fully solve the structurally similar
reasoning task).

> **Correction, 2026-08-13.** The sentence above originally read "a frozen
> Whisper-style audio encoder, LoRA applied only to the language-model
> layers". That was wrong, and the error was in the code, not the prose.
> PEFT matches a plain `target_modules` list by **suffix** across the whole
> model, and Qwen2-Audio's audio encoder names its attention projections
> `q_proj`/`k_proj`/`v_proj` exactly as the language model does. This run
> therefore adapted **96 of the audio tower's 192 linear layers** -- the Q,
> K and V of all 32 encoder layers -- alongside the language model.
> Verified directly by constructing the real model architecture and
> inspecting which modules `get_peft_model` actually wrapped (only the
> encoder's output projection and MLP, differently named, stayed frozen);
> pinned in `tests/test_finetune_encoder_modes.py`.
>
> This makes the negative **stronger**: adapting most of the encoder's
> attention had already failed. It also means the frozen-encoder condition
> this experiment is cited for has never actually been run. See
> `experiments/2026-08-13-encoder-level-probe/`.

**Not established then, and partly resolved since:** whether the task is
unlearnable by any fine-tuning approach. It isn't entirely -- see
`experiments/2026-08-13-dual-beats-dithering-finetune/notes.md`: the same
partially-adapted (`qkv`) encoder, given dithered audio that preserves
inter-channel variance through training, produces a real, above-baseline
side-accuracy signal (75.6%, replicated at 77.1% under a different
training seed), though with its own serious caveats (severe output-template
collapse, and deliberately unfreezing more of the encoder on top of
dithering only regressed it back to chance, twice). That regression
pattern has an explanation:
`experiments/2026-08-13-encoder-level-probe/` shows the level difference
survives the log-mel front end exactly and survives the encoder too (a
linear probe on its output recovers the planted azimuth at MAE 2.6deg, R2
0.95), so the encoder was never where the cue was lost, and unfreezing
more of it was probably never the right fix. `scripts/modal_finetune.py
--encoder {qkv,none,all,full,projector}` makes the choice explicit now,
`qkv` reproducing this run exactly; the indicated next target is
`projector`, the one component no run has ever trained.

The other next step this motivates: **a model with a genuinely
stereo-aware front end.** Surveyed in `docs/spatial_audio_models.md`, and
the answer is that no such model exists for stereo. Every general-purpose
audio LLM is mono at the front end; the three 2026 systems built for
spatial audio (SPUR, Spatial-Omni, The World is Not Mono) are all
First-Order Ambisonics and explicitly do not accept stereo -- and FOA
cannot represent an amplitude pan anyway, since a phantom image has no
direction of arrival. The transferable idea is SPUR's *architecture*: a
frozen encoder plus a small adapter fed inter-channel covariance, which is
exactly what this project's front end already computes.

## The honest paper framing

This is the second half of a complete, symmetric result:
**reasoning is fully teachable (0.4% -> 100%, 98.2% verifier-faithful);
perception, under this architecture, is not (28.5% -> 44.4%, which is
chance in disguise).** That symmetry, one clean success and one clean,
mechanistically-explained failure, from the identical method applied to
the two halves of the same task, is a stronger result than either half
alone, and it is exactly the kind of finding StereoMusicQA's verifier and
two-condition design exist to produce.

## Cost

Training + held-out eval, Modal A100-40GB. Landed in the same ballpark as
the cues fine-tune (~$10-12): comparable step count, similar per-item
eval cost.
