# The fine-tune's 100% cannot distinguish learning the law from memorising 21 answers

## What this is

`experiments/2026-08-10-cues-finetune-fixes-arithmetic/` is the project's
positive result: QLoRA takes Qwen2-Audio from 0.4% to 100% within-5deg on
held-out items. `docs/PROJECT_STATUS.md` read that as "it computes rather
than memorizes".

This is the check of that reading, prompted by the direct question "does
the fine-tuned reasoning still work on angles it never saw during
training?". The answer so far is that **nobody knows, including us**, and
the reason is a property of the benchmark rather than of the model.

Nothing here needed a GPU or a model run. It is all arithmetic and a
re-reading of transcripts already committed to this repo.

## The benchmark contains 21 distinct questions, not 1,045

A dry amplitude pan is exactly invertible, so the measured level difference
is a **deterministic function of the planted angle** -- the same angle on a
trumpet and on a cello produces the identical number. `describe_cues`
states it to one decimal place. So across the whole of v0.2:

| | |
|---|---|
| held-out items | 1,045 |
| distinct level differences in their prompts | **21** |
| distinct azimuths those map to | **21** |
| training examples | 5,332 |
| distinct (cue, answer) pairs available to learn from | **21** |

Read off the real transcripts, every one of the 1,045 held-out answers is
one of 21 (level difference -> azimuth) pairs, and the fine-tune gets all
21 exactly right:

```
-36.1 dB -> -29.2 deg      +0.0 dB ->  +0.0 deg     +12.9 dB -> +20.0 deg
-19.5 dB -> -25.0 deg      +2.7 dB ->  +5.0 deg     +17.2 dB -> +23.6 deg
-17.2 dB -> -23.6 deg      +3.2 dB ->  +6.0 deg     +19.5 dB -> +25.0 deg
      ... and the twelve others, symmetric about centre ...
```

**A model that learned `arctan(d * tan 30deg)` and a model that memorised a
21-row table produce byte-identical output on every item this benchmark
contains.** The 100% is real and it is not evidence for either one.

Drawn as `figures/finetune_answers_are_a_lookup_table.png`
(`python scripts/make_figures.py --only lookup`): the answers sit exactly
on the panning-law curve, at 21 points, with nothing in between ever asked.

### The existing argument for "it computes" does not hold

`docs/PROJECT_STATUS.md` says it "computes rather than memorizes (correct
to 0.1deg on the non-round mid/side-shifted angles)". Those angles --
29.2, 23.6, 17.8, 11.9, 6.0 -- come from mid/side width 1.2 applied to the
grid, and **the train split contains mid/side items**. They are rows 12-21
of the same table, not held-out values. Being correct on them shows the
table was learned completely, not that it was derived.

That sentence in `PROJECT_STATUS.md` has been corrected.

## The obvious experiment is a trap, twice over

**Trap 1: the headline metric cannot detect memorisation at all.** The
trained answers are at most 5deg apart, and the azimuth tolerance *is*
5deg, so a model that snapped every novel cue to the nearest angle it knew
would still score **100% within-5deg** on any eval set built inside
+-25deg. Reporting only the usual number would make the experiment
unfalsifiable. There is a test pinning this
(`tests/test_generalization.py::test_tolerance_cannot_detect_memorisation_anywhere_in_range`).

**Trap 2: the intuitive angles are contaminated.** "Halfway between the
grid angles" (+-2.5, 7.5, 12.5, 17.5, 22.5) sounds like the off-grid set,
but the trained answers are *not* that grid -- mid/side scattered ten more
between its lines. 17.5deg sits 0.32deg from the trained 17.82deg, and
12.5deg sits 0.55deg from the trained 11.95deg. Four of those ten angles
are effectively trained values, so a model that genuinely computed would
be scored 40% "on the trained grid" and look partly memorised.

`eval/generalization.py::choose_eval_angles` picks from the widest gaps in
the *actual* 21-value trained set instead, symmetrically about centre:

```
-21.8  -16.4  -13.5  -8.0  -2.5  +2.5  +8.0  +13.5  +16.4  +21.8
```

## What the eval can and cannot show, computed before running it

`eval.generalization.discriminability`, no model involved:

| | within 5 deg | MAE | answers on a trained angle |
|---|---|---|---|
| if it memorised the table | 100.0% | 1.84 deg | **100.0%** |
| if it learned the panning law | 100.0% | 0.00 deg | **0.0%** |

So the experiment is worth running, and the number to read is **the share
of answers landing on a trained angle**, with MAE as the secondary. The
within-tolerance rate should be reported and ignored. Drawn as
`figures/offgrid_eval_design.png`.

## The eval set

Built with `benchmark/configs/stereomusicqa_offgrid.json`: the same six
held-out URMP pieces and 19 stems as v0.2's test split, at the ten angles
above, dry pan only (no processing variants, so nothing confounds the
angle). Same seed, so the piece assignment is identical to v0.2's and the
only new variable is the angle.

```bash
python scripts/build_benchmark.py --config benchmark/configs/stereomusicqa_offgrid.json
```

Built: **1,390 items from 149 stems, 100 rejected** (all `audio_too_quiet`,
the same excerpt-offset issue as every other build), of which **190 are
held-out** — 19 stems x 10 angles, exactly balanced at 19 per angle. No
eval angle sits within 0.75deg of any of the 21 trained answers. Public
identifiers are opaque (`smq_test_00023_absolute`) and the public split
carries no azimuth or evidence field.

### The set is sound: the audio carries the labelled angles

```bash
python scripts/score_benchmark.py \
    --labels artifacts/stereomusicqa_offgrid/test_private_labels.jsonl \
    --baselines --angles=-21.8,-16.4,-13.5,-8,-2.5,2.5,8,13.5,16.4,21.8 \
    --json experiments/2026-08-13-offgrid-generalization/baselines.json
```

| Condition | Within 5° | Side | MAE | Inference |
|---|---|---|---|---|
| no answer | 0.0% | 0.0% | — | 0.0% |
| always centre | 20.0% | 0.0% | 12.44° | 10.0% |
| random from grid | 16.8% | 47.9% | 16.59° | 14.2% |
| front end (ceiling) | **100.0%** | 100.0% | 0.00° | 100.0% |

The front end recovers **100% at every one of the ten angles**, which is
the check that matters here: it confirms the rendered audio and the new
labels agree, so a model that fails on this set is failing at the task and
not at a broken eval. As everywhere else, that 100% is a control and not a
model result — a dry pan is exactly invertible.

The floor is 20.0%, slightly below v0.2's 23.6%, for the same arithmetic
reason: only two of the ten angles (+-2.5deg) sit within 5deg of centre.

## Status

**Established:** the benchmark cannot distinguish the two hypotheses; the
existing "it computes" claim is unsupported; an eval that *can*
distinguish them is designed, built, and its discriminating power is known
in advance.

**Not established:** which hypothesis is true. That needs one generation
pass with the existing cues adapter over the off-grid items, which needs
the Modal credentials this environment does not have. It is a single cheap
run -- the same shape as the eval in
`experiments/2026-08-10-cues-finetune-fixes-arithmetic/`, on 190 items
instead of 1,045.

```bash
modal run scripts/modal_generate.py --adapter adapters/cues \
    --artifacts artifacts/stereomusicqa_offgrid \
    --answers-path answers/test_answers_offgrid.jsonl
```

**This is the highest-value single run left in the project.** If the
answers land on the trained angles, the headline positive result is a
lookup table and the paper's central claim needs restating. If they land
on the true angles, the claim is stronger than it currently is, because it
would then be demonstrated rather than assumed.

## The run, added 2026-08-13: it's a lookup table

```bash
modal run scripts/modal_generate.py --artifacts-root stereomusicqa_offgrid \
    --adapter adapters/cues --answers-path answers/cues_offgrid.jsonl
python scripts/score_benchmark.py \
    --labels artifacts/stereomusicqa_offgrid/test_private_labels.jsonl \
    --answers answers_offgrid.jsonl \
    --angles=-21.8,-16.4,-13.5,-8,-2.5,2.5,8,13.5,16.4,21.8 --baselines
```

(`--artifacts-root` is new in `scripts/modal_generate.py`, added to run this:
every build's held-out items number from the same seeded sequence, so
v0.2's and the off-grid set's `audio_path`s collide byte-for-byte, and
uploading the wrong one to the shared `audio/test/` volume path would
silently read the wrong audio. Non-default roots now upload to
`audio/<root>_test/` and have their manifest's `audio_path` rewritten to
match before the remote call, so this can't happen quietly.)

| | Within 5deg | MAE | Side | On a trained angle |
|---|---|---|---|---|
| **cues adapter, off-grid** | **80.0%** | **4.81deg** | 100.0% | **100.0%** |
| Original benchmark (for reference) | 100.0% | 0.00deg | 100.0% | 100.0% (only 21 exist) |
| If it memorised (predicted in advance) | 100.0% | 1.84deg | -- | 100.0% |
| If it learned the panning law (predicted) | 100.0% | 0.00deg | -- | 0.0% |

**Every one of the 190 answers lands on one of the 21 trained angles.**
Not approximately -- `eval.generalization.on_grid_rate` against the real
predictions extracted from the real transcripts returns exactly 100.0%.
The measured within-5deg (80.0%) and MAE (4.81deg) land off both
predictions above because "snap to the nearest trained angle" isn't quite
what happens; see below for what actually does.

**How it happens, read straight from the transcripts.** The model states
the real, novel off-grid cue correctly in its first sentence, then its own
"step-by-step derivation" silently substitutes the nearest *trained* cue's
arithmetic before computing anything, and the final answer is that trained
cue's exact trained answer. Item `smq_test_00023_absolute`, trumpet, true
azimuth **-21.8deg** (an off-grid angle -- not a trained one):

> The inter-channel level difference is **-14.8 dB**, so the right channel
> is louder. Converting with the panning law: r = 10^(-14.8/20) =
> **0.139**; d = (r - 1)/(r + 1) = -0.756; azimuth = arctan(d * tan 30deg)
> = **-23.6 degrees**. The trumpet is at 23.6 degrees to the right.

`10^(-14.8/20)` is `0.182`, not `0.139`. `0.139` is what `10^(x/20)` gives
for `x = -17.1`, and **-17.2 dB -> -23.6deg is one of the project's 21
trained pairs**, exactly the final answer given. The model never touches
the -14.8 dB it just stated; it does the arithmetic for -17.2 dB, the
trained cue nearest to it, and reports that answer instead. Checked across
all 190 transcripts (regex-extracted `r` and final azimuth, matched
against the 21 trained ILD/azimuth pairs at a `0.3` tolerance): **189/190
(99.5%) state an `r` value that corresponds to a trained cue, not the cue
they just stated**, and 189/190 final azimuths match that trained cue's
trained answer exactly.

This is a stronger and more specific finding than "answers land near
trained values" -- the model produces fabricated intermediate arithmetic
that contradicts its own stated premise, in service of reproducing a
memorised answer. It is not doing the conversion on novel inputs at all;
it is recognising which of 21 rows a cue is closest to and reciting that
row, dressed in the *form* of a derivation.

## Status, resolved

**The paper's central positive claim is a lookup table, not learned
computation.** "0.4% -> 100% within-5deg, it computes rather than
memorizes" (`experiments/2026-08-10-cues-finetune-fixes-arithmetic/`,
`docs/PROJECT_STATUS.md`) needs restating: QLoRA fine-tuning teaches the
model to recognise which of 21 known (cue, answer) pairs a prompt is
closest to and recite the paired answer inside a plausible-looking
derivation, not to compute the panning law. The reasoning fine-tune's
headline result is real in the sense that it reliably produces this
behaviour on held-out *items* -- but every held-out item, by construction,
carries one of only 21 possible cues, so "held-out" never meant "novel" for
this benchmark. On genuinely novel cues it falls to 80.0% within-5deg,
still with 100% side accuracy (the sign of the cue survives even when the
magnitude doesn't) but with a mean error of 4.81deg and a mechanism that is
demonstrably not computing.

Honest framing for the paper: this is not a null result. A model that
reliably recognises the nearest of 21 memorised patterns and produces a
faithful-*looking* derivation for it, entirely undetected by the
benchmark's own headline metric, is itself a finding worth reporting --
about the benchmark design (the tolerance-vs-grid-gap trap this
experiment's own baseline table predicted in advance) as much as about the
model. `eval.generalization.on_grid_rate` and this off-grid set are what
future fine-tuning claims on this benchmark should be checked against
before being reported as computation.

## The verifier didn't catch this either, and now does (added 2026-08-13)

At the time the section above was written, "verifier-passing" was also
true: every one of these 190 transcripts passed `inference_passed`, the
project's own faithfulness gate. That's a real gap in the verifier, not
just the headline accuracy metric. `verifier/checker.py` checked every
claim against ground truth (the stated cue against the real measured ILD,
the final azimuth against the real measured angle) but never checked
whether a transcript's own *intermediate* arithmetic was internally
consistent -- so a transcript that opens with the correct, real cue
(passes the read-out check) and lands within 5deg of the true angle by
the coincidence of the nearest memorised answer (passes the inference
check) sailed through, regardless of what happened in between.

Fixed: `verifier/extract.py::extract_premise_r` reads the `r = 10^(x/20)
= y` step's own two numbers, and `verifier/checker.py` now checks whether
`y` actually equals `10^(x/20)` for the `x` the same formula just wrote
down -- not against ground truth, against the transcript's own stated
premise. This is exactly what catches the substitution: in the -14.8dB
example above, the formula's displayed input is the correct -14.8, but
its result (0.139) is the answer for a different, memorised -17.2, and
`0.139 != 10^(-14.8/20)`.

Re-scored with the fix in place:

| | Before | After |
|---|---|---|
| Off-grid verifier faithfulness (`inference_passed_rate`) | 80.0%* | **0.5%** |
| Original in-distribution verifier faithfulness (unchanged, re-scored to confirm) | 98.2% | 98.2% |

(*before the fix, `inference_passed` had no premise check to fail, so it
tracked the within-5deg/side checks alone -- not a previously-reported
number under that name, shown for contrast.)

The in-distribution number is unchanged because it never needed to be:
memorised answers, reproduced verbatim for a cue the model has actually
seen, are internally consistent by construction (`faithful_derivation`
generates every real target that way). The premise check only fires when
a transcript's own arithmetic doesn't agree with itself, which is
specifically the off-grid substitution's signature, not something that
happens when the model is just reciting a memorised row correctly. 0.5%
(1/190) is close to the 189/190 fabrication rate found by hand-reading
transcripts earlier in this file -- the automated check and the manual
read agree.

## The fix: dense synthetic cue augmentation (added 2026-08-14)

The root cause was never the training procedure; it was that the whole
benchmark's dry pan gives only 21 distinct (cue, answer) pairs to learn
from, and memorising 21 rows is a strictly easier fit than learning
`arctan`. `model/dense_cue_augmentation.py` generates additional
synthetic (cue, answer) pairs by continuous sampling -- each one a real,
correctly-derived example (`cues/level.py::ild_db_for_azimuth`, the exact
inverse of the forward formula) that no real render happens to carry,
reusing an existing real render as audio padding (the cues condition was
always solvable from the text alone, so the audio does not need to match
the stated cue). Off-grid eval angles excluded with margin so none of the
190 held-out points could leak into training by chance rounding.

Retrained `--condition cues --dense-augment 3000`: 5,332 real + 3,000
synthetic = 8,332 examples, one epoch. Loss settled around 0.06-0.08,
not the ~0.0001 the original 21-cue run collapsed to -- the model can no
longer reach near-zero loss by memorising a small closed set, which is
itself a sign something different is happening during training, before
any evaluation.

Re-ran the exact same off-grid eval against the new adapter
(`adapters/cues_denseaug3000`):

| | Original adapter | Dense-augmented adapter |
|---|---|---|
| Within-5deg | 80.0% | **100.0%** |
| MAE | 4.81deg | **0.24deg** |
| Side accuracy | 100.0% | 100.0% |
| Answers on one of the 21 *original* trained angles | 100.0% | **0.0%** |
| Verifier faithfulness (`inference_passed`, premise check included) | 0.5% | **100.0%** |

Every metric that separates memorisation from computation flipped. Read
from a transcript directly -- the same item that fabricated before
(`smq_test_00023_absolute`, true azimuth -21.8deg): previously stated
`r = 10^(-14.8/20) = 0.139` (the value for a different, memorised -17.2dB
cue) and answered -23.6deg; now states `r = 10^(-14.8/20) = 0.187`
(true value 0.182, small ordinary arithmetic imprecision, not a
substitution) and answers -21.5deg, off by 0.3deg. The chain computes
forward from the actual stated cue now, consistently, across all 190
items -- not memorised-table recitation dressed as a derivation.

## Checked against the original in-distribution test too (added 2026-08-14)

Same 1,045-item held-out set the original 100% result was measured on,
same adapter (`cues_denseaug3000`):

| | Original adapter | Dense-augmented adapter |
|---|---|---|
| Within-5deg | 100.0% | 100.0% |
| MAE | 0.00deg | 0.13deg |
| Side accuracy | 100.0% | **91.2%** |
| Verifier faithfulness | 98.2% | **100.0%** |

The side-accuracy number looks like a regression and isn't one, but the
first explanation written here for why was wrong and is corrected below.

All 92 "wrong-side" items -- every single one, no exceptions -- are items
whose *true* azimuth is exactly 0.0deg (92 of the 95 true-centre items in
the set). A real transcript: cue displayed as "0.0 dB", `r =
10^(0.0/20) = 1.007`, giving `d = 0.003`, `azimuth = 0.1 degrees to the
right`. **Correction:** this is not the model recovering a "hidden",
slightly-nonzero true level difference -- checked directly, this item's
true `ild_db` is exactly `0.0`, and the model never sees anything but
the rounded "0.0 dB" text regardless, so there is no unrounded value for
it to recover even in principle. `10^(0/20)` is exactly `1.000`; `1.007`
is a real, if small, inconsistency in the model's own generated text, not
evidence of hidden precision.

The actual source: `model/finetune_data.py::faithful_derivation`, which
generated every training target (real and synthetic alike), computes `r`
from the *unrounded* `ild_db` but displays both `ild_db` and the formula's
own `x` rounded to one decimal --

```python
r = 10.0 ** (ild_db / 20.0)                    # unrounded
f"r = 10^({ild_db:.1f}/20) = {r:.3f}"          # x shown rounded, r is not
```

-- so essentially every training example carries a small, systematic gap
between its displayed premise and its displayed result (worst case
observed elsewhere ~0.4deg from the *r* precision, ~0.06deg here from the
*ild_db* rounding), comfortably inside the verifier's 0.5dB premise
tolerance. The model learned this pattern from thousands of examples that
all have it and reproduces it as generation noise, not as physics. Right
at true-centre, where the correct answer is 0.0deg exactly, that
noise -- entirely ordinary, present on essentially every item, usually
invisible because it does not cross a side boundary -- is large enough
relative to the target that it has roughly even odds of landing a hair to
either side. The original adapter's 100% here came from reciting a
memorised exact "0.0", which was never more informative about whether it
computes; the new adapter's near-misses are the same small noise every
other item also carries, just visible here because centre is the one
place noise this size can flip a sign.

Zero real side errors: every item with a true azimuth away from zero,
across the entire 1,045-item set, has the correct side. MAE moving from
0.00 to 0.13deg is the same story in miniature -- 0.00 was memorised
recitation, 0.13 is the same ordinary generation noise described above,
on cues the model has seen before as well as ones it hasn't.

**Net result: the fix cost nothing on the original benchmark and fixed
generalisation completely.** Within-5deg and verifier faithfulness both
held or improved; the one metric that moved is explained by generation
noise (present on nearly every item, inherited from a rounding-vs-
computation gap in `faithful_derivation`'s own training-target text)
meeting the one point where any noise can flip a sign -- not a capability
regression.

## `faithful_derivation` itself fixed (added 2026-08-14)

The gap above is now closed at the source: `model/finetune_data.py::
faithful_derivation` rounds `ild_db` once and computes `r`, `d`, and the
displayed azimuth (and, since it follows from that azimuth's sign, the
stated side) all forward from that single rounded value, rather than
mixing a rounded display with full-precision arithmetic. The item's true
`azimuth_deg` is now used only as a sanity check that the self-consistent
chain lands within a degree of it (`MAX_GROUND_TRUTH_DRIFT_DEG`), not as
what gets displayed. Regression coverage in `tests/test_finetune_data.py`
parses generated text with no access to the function's internals and
independently recomputes every step from what a reader of the text would
have -- the same thing the verifier's premise check and a training run
both do, so this would have caught the original bug directly.

**The adapter every result in this file was measured on (`cues_denseaug3000`
and the original `adapters/cues`) used the earlier, imperfect generator.**
Nothing here has been rerun against the fix, and nothing needed to be: the
discrepancy it introduced is bounded at 0.05dB (checked numerically,
`experiments/2026-08-13-offgrid-generalization/notes.md`'s own true-centre
section above), far below anything that could move a headline number, and
its one visible effect -- the true-centre near-misses -- is already
identified and explained, not hidden by the fix. A future retrain (e.g. a
second training seed, if one is run to replicate the dense-augmentation
result) should use the corrected generator; the seed replication and a
confirmation that the cleanup has no material effect come from the same
run.
