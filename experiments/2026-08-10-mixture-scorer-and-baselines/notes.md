# Mixture items get a scorer, and a floor and ceiling

## What this is

The 28 mixture items (`experiments/2026-08-10-stereomusicqa-v0.2-baselines/`
built them, didn't score them) had no way to be graded: `eval/score.py`
assumes one `answer_azimuth_deg` per item, and a mixture item's answer is a
list. This adds `eval/mixture_score.py`, the counterpart, and runs the same
kind of reference conditions v0.1/v0.2's single-source items already have.

```bash
python scripts/score_mixture_benchmark.py \
    --baselines \
    --json experiments/2026-08-10-mixture-scorer-and-baselines/result.json
```

## Two ways to be wrong, not one

A mixture answer can be wrong on count (heard 2 sources, there were 3) or
on position (heard 3, but put one in the wrong place), and those are
different failure modes worth telling apart -- same principle
`eval/score.py` applies to accuracy vs. faithfulness. So every item gets:

- **recall**: of the true positions, what fraction had a matching
  predicted position within 5 degrees. Misses a source -> recall drops.
- **precision**: of the predicted positions, what fraction were real.
  Hallucinates an extra one -> precision drops, recall doesn't.
- **exact_match**: both perfect for this item. The strict pass/fail,
  computed over every item including unanswered ones (unanswered counts as
  wrong, same rule as `within_tolerance_rate`).

Matching a predicted position to a true one is a plain greedy pairing, not
an assignment problem, and that's provably safe rather than assumed:
`benchmark/mixture_render.py`'s `MIN_ANGLE_SEPARATION_DEG` (10 degrees) is
more than twice the 5-degree tolerance, so no predicted value can ever
land within tolerance of two different true positions at once.

## Results, 2 held-out mixture items

Small sample and stated as such -- only 2 of the 28 accepted mixture items
landed in the test split (piece-level splits mean mixture items are scarce
per split; see caveat below).

| Condition | Exact match | Count correct | Recall | Precision | Coverage |
|---|---|---|---|---|---|
| no answer | 0.0% | 0.0% | 0.0% | n/a | 0% |
| always 2 sources | 0.0% | 50.0% | 41.7% | 50.0% | 100% |
| always 3 sources | 0.0% | 50.0% | 83.3% | 66.7% | 100% |
| front end (ceiling) | **100.0%** | 100.0% | 100.0% | 100.0% | 100% |

The two constant baselines guess a fixed number of sources (2 or 3) spaced
by the same `MIN_ANGLE_SEPARATION_DEG` real items respect -- not arbitrary
numbers, the actual constant the benchmark is built around. Neither reaches
count_correct on both items (one test item has 2 sources, the other 3), so
each baseline is right about count on exactly one of the two -- consistent
with `count_correct_rate` landing at 50% for both. Both score exact_match
0%: even the baseline that gets the count right doesn't land its positions
within 5 degrees of the real ones, since it isn't listening to anything.

**The ceiling is 100% and is not a model result**, same caveat every other
front-end baseline in this project carries. It answers with
`cues/mixture.py`'s own measurement, which `benchmark/mixture_validate.py`
already checked was within tolerance of every planted position before the
item was accepted -- so this number is a consistency check on the labels,
not a claim about difficulty.

## The honest caveat

**n=2 is not enough to trust any of these numbers as more than a sanity
check.** 28 accepted mixture items split 70/15/15 at the piece level
leaves very few in any one split; two of them land in test. The floor and
ceiling here confirm the scorer computes the right arithmetic on real
data (which is what this experiment actually needed to prove -- the
scorer, not the sample), not that the baselines are meaningfully
characterized. A larger mixture item pool needs more usable multi-instrument
pieces than URMP's 44 provide -- MedleyDB access (requested, pending) is
what that depends on.

## What is still missing

Same as the single-source items: no real model has been run over this
either. The scorer and the floor/ceiling exist; a generation pass over the
held-out items does not, for the same reason `docs/compute.md` already
gives -- it needs a rented GPU, not laptop CPU.
