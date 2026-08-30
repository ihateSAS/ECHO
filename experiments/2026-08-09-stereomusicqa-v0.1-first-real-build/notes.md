# StereoMusicQA v0.1, built for real

## What this is

The benchmark builder had never been run on the actual dataset. Every test
of it used seven synthetic noise stems in a temporary directory, and one
URMP stem happened to be on disk. So the deliverable this project is named
around had never existed as a file.

This is the first real build: all 149 URMP `AuSep_` stems, eleven angles
each, measured back and validated one item at a time.

```bash
python scripts/build_benchmark.py --config benchmark/configs/stereomusicqa_v0.1.json
```

About 30 minutes on this laptop's CPU, 2.5 GB of 24-bit WAV, no GPU. The
output is gitignored; `result.json` next to this file carries every number
below.

## What came out

| | |
|---|---|
| Stems in inventory | 149 |
| Renders planned | 1,639 (149 × 11 angles) |
| Items accepted | **1,529** |
| Items rejected | 110 |
| Train / dev / test | 1,056 / 264 / 209 |
| Pieces per split | 31 / 7 / 6, **zero overlap** |
| Instruments | all 13 URMP codes |
| Items per angle | 139 at every one of the eleven angles |
| f0 tracked | 1,529 of 1,529, spanning 59.0 to 1005.0 Hz |

Splits hold at the piece level, which is the property that stops the same
performance appearing on both sides of a train/test boundary, and the angle
distribution is exactly balanced, so no model can profit from a prior over
which angles are common.

## What it does not prove, which matters more than what it does

Azimuth recovery error across all 1,529 items: **4.44e-15 degrees**.
Coherence: exactly 1.0 on every item. |ITD|: 2.25e-14 microseconds.

Those numbers are not an accuracy result and should never be quoted as
one. A dry amplitude pan multiplies one mono signal by two constants, so
every frequency bin carries the identical level ratio, the median across
bins inverts it exactly, two exactly-proportional channels are coherent by
definition, and GCC-PHAT of a zero-phase cross-spectrum peaks at zero. The
front end could not have failed here whatever the audio was. It is the same
point the real-record survey made about its own dry control: a pass says
the plumbing is sound and says nothing else.

So what this build actually establishes is bookkeeping and scale --
inventory, splits, rendering, measurement, validation, manifest writing and
hashing all work on 1,529 real items without a crash, a NaN, or a silent
`None`. Where the estimators get tested is
`experiments/2026-08-09-real-stereo-front-end-survey/` (real records, no
ground truth) and `tests/test_mixture_level.py` (several sources at once).
Both of those can fail. This cannot.

Worth watching anyway: active bins ran from **73** at the low end to 1,328
at the high, median 395. A clip with 73 usable bins is a quiet sustained
note with little harmonic content, and it is the sort of item where a
future, harder estimator would degrade first.

## Finding 1: the held-out set was publishing its own answers

Found by looking at the output rather than at the code, and it would not
have surfaced any other way.

All 209 public test items were named like
`AuSep_1_tpt_07_GString_amp_neg25_15312ms_absolute`, with the audio file
matching. `neg25` is the answer. A model never had to listen to anything,
and an evaluation harness that read filenames would have posted a perfect
score against a benchmark that had handed over the labels.

The leakage test that existed asserted `answer_azimuth_deg` and
`measured_evidence` were absent from the public record. They were. "The
fields are absent" and "the answer is absent" are different claims and only
the first was being made.

Fixed before this build was kept: test items are now published as
`smq_test_00050`, with the audio file named to match, and the descriptive
`render_id` appears only in `test_private_labels.jsonl`. The numbering is a
shuffle keyed on the build seed rather than a hash of the render_id --
the stem list, angle set and seed are all in the published config, so a
hash would be reversible by enumerating every possible render_id and
matching digests. Re-checked on the rebuild: **0 of 209** public
identifiers or paths encode an angle.

## Finding 2: 110 items lost to a single unlucky excerpt each

Every one of the 110 rejections is `audio_too_quiet`, and they come from
exactly **10 stems × 11 angles**. `make_render_plans` draws one random
start offset per stem and reuses it for all eleven angles, so when that
offset lands on a rest, the whole stem's eleven items die together rather
than one of them.

Trumpet accounts for 44 of the 110 from four stems, which is what you would
expect -- brass parts rest more than string parts do.

That is the acceptance gate working exactly as designed: a six-second
excerpt of near-silence cannot support a question about where the
instrument is. But 6.7% of the benchmark is a lot to lose to one draw, and
the loss is concentrated rather than spread.

**Not changed here, deliberately.** The obvious fix -- resample the start
offset until the excerpt is loud enough -- would bias the benchmark toward
loud passages, and "how well does this work on quiet playing" is a question
someone may want to ask of it later. Choosing sustained-loud excerpts on
purpose is a legitimate design decision and so is keeping the uniform draw;
what is not legitimate is making that choice by accident inside a retry
loop. If it is changed, the retry count belongs in the config next to the
seed, and this paragraph should be updated to say which way it went and
why.

## Cost note

Roughly 30 minutes of CPU, of which most is pitch tracking: `measure_render`
runs pYIN once per item, and the eleven items from one stem share the same
mono content up to a constant gain, so ten of every eleven pYIN calls
recompute an identical answer. Caching on (stem, offset) would cut the
build to a few minutes. Not done -- it is a real speedup with a real
correctness argument behind it (a pan is gain-only and pYIN's silence floor
is relative, so the f0 is genuinely identical), but 30 minutes is not
currently costing anyone anything, and a cache that is subtly wrong would
poison every item's `f0_hz` at once.
