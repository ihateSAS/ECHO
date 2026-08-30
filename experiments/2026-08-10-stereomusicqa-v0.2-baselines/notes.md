# StereoMusicQA v0.2: the floor and ceiling move

## What this is

`experiments/2026-08-09-stereomusicqa-baselines/` set the floor and ceiling
for v0.1 -- 1,529 dry-only items. v0.2 rebuilt the benchmark from the same
149 URMP stems with four of the five processing types turned on
(compression, EQ, mid/side width 1.2, delay-widening 300 us; reverb stays
off -- see `docs/PROJECT_STATUS.md` for why). That changes the item count,
the angle grid, and therefore both baselines, so the v0.1 numbers are not a
valid yardstick for anything scored against v0.2. This re-runs the same
four reference conditions, no GPU and no model involved, against the real
thing.

```bash
python scripts/score_benchmark.py \
    --labels artifacts/stereomusicqa_v0.2/test_private_labels.jsonl \
    --baselines \
    --json experiments/2026-08-10-stereomusicqa-v0.2-baselines/result.json
```

## Results, 1,045 held-out single-source items

(Mixture items -- 28 accepted, held in
`artifacts/stereomusicqa_v0.2/mixture_test_private_labels.jsonl` -- use a
different question schema (`answer_count` / `answer_azimuths_deg`, not a
single `answer_azimuth_deg`) that `eval/score.py` does not score yet. Not
included below; a real gap, not an oversight.)

| Condition | Within 5° | Side | MAE (answered) | Coverage | Inference passed |
|---|---|---|---|---|---|
| no answer | 0.0% | 0.0% | — | 0% | 0.0% |
| always centre | **23.6%** | 9.1% | 14.13° | 100% | 14.5% |
| random from grid | 23.5% | 40.5% | 18.77° | 100% | 18.9% |
| front end (ceiling) | **100.0%** | 100.0% | 0.00° | 100% | 100.0% |

v0.1's numbers for comparison: always-centre 27.3%, random-from-grid 24.4%,
side accuracy 9.1% / 37.3%.

## Why the floor moved and the ceiling didn't

**The floor dropped, 27.3% to 23.6%, because the angle grid got wider.** A
dry pan and three of the four processing types (compression, EQ,
delay-widening) leave the planted angle exactly where it was, but mid/side
width 1.2 rescales it: `d' = width * d` (`benchmark/process.py`), so a stem
panned to +5 degrees answers at +5.99 degrees after widening, not +5. The
21 distinct answerable angles in v0.2 (`within_tolerance_by_angle`'s keys in
`result.json`) are the original 11 plus 10 new ones the width scaling
produces -- and the new ones are the ones furthest from centre, since width
> 1 always pushes |d'| > |d|. More of the distribution now sits outside the
5-degree window around zero, so guessing centre pays off less often. This
is arithmetic, the same way 27.3% was: it says nothing about difficulty in
any interesting sense, only that the answer key spread out.

**The ceiling did not move, and is still not a model result.** Every one of
the 21 angles scores 100% under `within_tolerance_by_angle` for the front
end baseline, mid/side-shifted ones included: `expected_after_processing`
computes the exact post-widening angle as the label, and `cues/level.py`'s
inversion is exact by construction on all four processing types actually
turned on here, same as it was on a dry pan. `docs/PROJECT_STATUS.md`
already flagged that reverb, mixtures, or real binaural rendering would be
what moves the ceiling below 100% -- the single-source v0.2 items don't
include reverb (rejected as unanswerable) or binaural rendering, so this
prediction holds and the ceiling is unchanged. The 28 mixture items are a
different question type this scorer doesn't cover yet, so they're not part
of this ceiling claim either way.

## What is still missing

Same conclusion as v0.1's baselines, restated because it is still true: a
real model has not been run over this. `docs/compute.md` prices that at
~$10-25 of rented GPU and about 60+ days of continuous laptop CPU for the
larger v0.2 test set -- the laptop number is not a slower option, it is not
an option. Scoring the 28 mixture items also needs a scorer extension
(`eval/score.py` currently assumes one `answer_azimuth_deg` per item, not a
list) before either can happen.
