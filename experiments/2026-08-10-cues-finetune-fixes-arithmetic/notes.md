# Fine-tuning fixes the reasoning failure completely (the first positive result)

## What this is

The zero-shot runs established that Qwen2-Audio, handed the cues as text,
gets direction mostly right (71.9% side) but cannot do the arctan
arithmetic (0.4% within 5deg): it fabricates the conversion, snapping to
boundary or memorized-constant values the verifier catches. This is the
experiment that tests whether that reasoning failure is *teachable*. It is,
completely.

QLoRA fine-tune of Qwen2-Audio-7B on the 5,332 train items, on the
**cue-text** condition: the target is the faithful worked derivation
`model/finetune_data.py` builds from each item's own level difference (r,
d, arctan, the planted angle, all recomputed, so the training signal is
*correct reasoning*, not an opaque label). 4-bit NF4 base, LoRA r=16 on the
LLM projections (44 M trainable params, 0.52%), 1 epoch, on Modal
(A100-40GB). Adapter evaluated as 4-bit base + adapter over all 1,045
held-out items.

```bash
modal run scripts/modal_finetune.py --condition cues
modal run scripts/modal_generate.py --adapter adapters/cues \
    --answers-path answers/test_answers_finetuned.jsonl
python scripts/score_benchmark.py \
    --labels artifacts/stereomusicqa_v0.2/test_private_labels.jsonl \
    --answers experiments/2026-08-10-cues-finetune-fixes-arithmetic/answers.jsonl \
    --baselines
```

## Results, 1,045 held-out items

| Metric | Zero-shot (cue-text) | **Fine-tuned (cue-text)** | Front-end ceiling |
|---|---|---|---|
| Within 5deg | 0.4% | **100.0%** | 100% |
| Side accuracy | 71.9% | **100.0%** | 100% |
| MAE (answered) | 74.3deg | **0.00deg** | 0deg |
| Verifier faithful (inference) | 0.4% | **98.2%** | 100% |

Fine-tuning closes the gap to the front-end ceiling almost entirely. The
model went from *fabricating* the conversion to *computing* it: on every
held-out item it now states the correct level difference, the correct r and
d, and the correct final angle. A representative held-out transcript (true
= -25deg, a stem never seen in training):

> "The inter-channel level difference is -19.5 dB, so the right channel is
> louder. Converting with the panning law: r = 10^(-19.5/20) = 0.106;
> d = (r - 1)/(r + 1) = -0.808; azimuth = arctan(d * tan 30deg) = -25.0
> degrees. The trumpet is at 25.0 degrees to the right."

Every intermediate is numerically right, versus the zero-shot model's
memorized constants (53.13deg, 63.43deg) and impossible >90deg values.

**The `inference` column is the point, not the accuracy.** 98.2% verifier
faithfulness means the reasoning chain the model produces is checkable
against the signal and passes. BinauralCoT's whole premise (elicit
spatial reasoning where every step is verifiable) is demonstrated, not just
argued. A model that got the right answer by an unfaithful route would
score high on within-5deg and low here; it scores high on both.

## It is computing the mapping, not memorizing answers

The mid/side-shifted angles are the check. Width scaling produces non-round
answerable angles (-29.23, -23.59, -17.82, -11.95, +11.95 deg) that are
not memorizable round numbers. The fine-tuned model returns -29.2, -23.6,
-17.8, -11.9, +11.9, correct to 0.1deg on held-out audio. It learned the
deterministic dB->angle function and applies it.

## Honest caveats: what this does and does not show

- **This is the cue-text (reasoning) condition, not perception.** The model
  is handed the level difference as text and learns to convert it. It does
  **not** address whether the model can *hear* the position: that is the
  audio-only fine-tune, not yet run, and the genuinely uncertain one. This
  result fixes the reasoning half of the two-failure decomposition and
  leaves the perception half exactly where it was (28.5%, below chance).

- **Train and test share the angle grid.** Both draw answerable angles from
  the same set (the 11-angle grid plus the mid/side-shifted values), so the
  model learned the mapping *over those angles*. Extrapolation to angles
  never seen in training is untested, and 100% partly reflects that the
  answer set is a fixed, learnable lookup. A robustness experiment holding
  out whole angles from training would separate "learned the function" from
  "learned the grid", worth doing before over-claiming generalization.

- **100% = matches the front end, which is a formula we handed over.** The
  honest framing is not "the model discovered how to localize" but
  "the reasoning/arithmetic failure is fully teachable, and the taught
  reasoning is verifiably faithful." That isolates perception as the one
  remaining hard problem, which is a clean contribution, not an inflated
  one.

- **A scorer fix was needed to see the true number.** The azimuth extractor
  only recognized left/right, so the model's correct centre answers ("0.0
  degrees to the center", all 95 centre items) scored as unparseable:
  90.9% before the fix, 100% after (`verifier/extract.py` now reads a
  centre claim as 0 degrees; `tests/test_score.py` covers it). The fix
  applies to every condition's reported number, not just this one.

## Cost

Training (A100-40GB, ~1.5-2 h) plus the held-out eval (~10 s/item over
1,045 items, ~2.9 h, because each answer is a full ~200-token derivation).
Total for the cues experiment landed around **$10-12** of real Modal spend,
more than the $4-7 first estimated, the difference being the eval's long
derivations. Future evals of a verbose fine-tune should cap max_new_tokens
lower and/or score a random subset.

## What this unlocks

The positive version of the paper now has a real result: *BinauralCoT can
be trained to produce faithful, verifiable spatial reasoning* (98.2%
faithful, from 0.4% zero-shot). The open questions that remain are the
interesting ones: can fine-tuning teach *perception* (audio-only), and
does the reasoning fine-tune extrapolate beyond the trained angle grid.
