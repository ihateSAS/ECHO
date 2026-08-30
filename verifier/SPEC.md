# Verifier spec

Status: v1 implemented (`verifier/checker.py`, `verifier/extract.py`,
`verifier/schema.py`), tested against real measured audio and the real
Part 6 transcript (`tests/test_verifier.py`). This doc is still the
design reference -- read it for *why* the code is shaped this way, in
particular the read-out/inference split and the two diagnostics, neither
of which is obvious from the code alone. What's still open: the claim
extraction is regex/keyword-based prose parsing (Step 1's "if prose has
to stay" path), not the structured-output-contract path Step 1 lists as
the preferred option -- that remains a real design decision for whoever
owns the prompt format, not something this implementation decided
unilaterally.

**Resolved since first writing:** this section used to say that
`describe_cues()` stated the regime and azimuth answers directly in the
prompt, so `inference_passed` could not distinguish real inference from
transcription. It no longer does. `describe_cues(..., disclosure=...)`
defaults to `"evidence"` -- the measurements plus the panning law needed
to convert them, and neither conclusion -- so the two `inference` rows in
the tolerance table below are now claims the model had to derive rather
than copy. The old prompt is still reachable as
`disclosure="conclusions"` for reproducing the recorded Part 6 and
precision-comparison runs, and `disclosure="no_numbers"` supplies the
ablation condition `verify_ablation_pair` needs. Step 4's coherence
mitigation is also in: the figure now travels with words ("Coherence:
1.00 (a single compact source)").

## Why this exists, with a real example

Part 6 already produced the failure this module is for. Given a prompt
stating coherence was 1.00 (the maximum -- a perfectly compact source),
the model replied:

    "...there is no coherence between the two channels."

The exact opposite of the measured value. It copied the two numbers it
understood (level difference, which side was louder) and invented a
reading of the third, in a sentence fluent enough to pass a casual read.
Nothing downstream caught it, because nothing downstream re-checked it.

That is the general failure mode: a reasoning trace that asserts a value,
where the value is wrong but the sentence around it is convincing. The
project's own non-negotiable rule (CLAUDE.md: "nothing is finished until
it recovers a value we planted ourselves") applies to the front end. The
verifier is that same rule applied to the model's *output* -- every
number the model asserts about the audio should be independently
recomputable from the waveform, and checked, not taken on faith.

## What it is not

Not a second opinion from another model. Not a training-time reward
signal (that may come later, but this spec is about a standalone checker
first -- get the checking right before wiring it into anything that
optimizes against it). Not a grammar or fluency check. It only checks
whether asserted *numbers* match what the waveform actually contains.

## The reasoning trace has four steps, and they are not equally checkable

The paper's schema (the target format `describe_cues`-driven prompting is
meant to move toward, not what free-text generation produces today) is
four steps:

1. **Regime detection.** Amplitude-panned or genuinely delayed, decided
   from the level-flatness and delay evidence together (`cues/regime.py`'s
   `classify_regime`, given both a `LevelEstimate` and a `DelayEstimate`).
2. **Report the main cue.** The level difference (amplitude regime) or
   the interchannel delay (delayed regime) -- whichever `CueMeasurements`
   field the regime from step 1 makes the relevant one.
3. **Identify the instrument**, from harmonic/pitch content (`cues/pitch.py`).
4. **Convert the step-2 cue to an azimuth** via the panning law (amplitude)
   or Woodworth inversion (delayed) -- `cues/level.py`'s `_angle_from_ild`
   or `cues/binaural.py`'s `invert_woodworth`.

The reason this spec cares about the split, not just about checking all
four: steps 2 and 3 are **read-out** -- the front end already computed
that exact number and handed it to the model in the prompt
(`describe_cues`'s job), so a step-2/3 claim matching `CueMeasurements`
only proves the model can transcribe its own input. Steps 1 and 4 are
**inference** -- regime detection has to weigh level-flatness and delay
against each other rather than copy either one, and the panning-law
conversion has to actually apply a formula to a value, not just restate
it. A verifier (or a faithfulness metric built on top of one) that scores
all four steps identically will read as high-faithfulness on a model that
is only ever transcribing steps 2 and 3 and guessing steps 1 and 4 -- see
`ClaimCheck.consistency_kind` below, which exists specifically so that
distinction survives into the result rather than getting averaged away.

Two diagnostics catch what a pure read-out/inference split alone
wouldn't:

- **Perturbation test.** Re-render the same source at a different azimuth
  and re-ask. A model doing real step-4 inference should track the new
  cue value to a new angle; a model that has learned a prior over
  plausible-sounding angles (e.g. "instruments are usually panned near
  center") won't move, or won't move by the right amount. This is the
  check that catches a model whose *individual* claims each look
  consistent but that is not actually conditioning on the input.
- **Numeric-ablation condition.** Ask the same question with the
  scalar cue values stripped out of the prompt (text-only description,
  e.g. "left channel louder" with no dB figure). A model whose step-4
  answer barely changes without the number it supposedly converted was
  never really doing the conversion.

Neither diagnostic is a `ClaimCheck` against `CueMeasurements` -- they
compare two model outputs to each other (same audio different angle, or
same angle with/without numbers) rather than one output to a measurement.
Worth building as a second entry point (`verify_perturbation_pair(...)`,
`verify_ablation_pair(...)`) alongside `verify()` rather than folding into
it, since they need two transcripts as input, not one.

## Input and output

```
verify(transcript: str | ReasoningTrace, evidence: CueMeasurements) -> VerificationResult
```

`evidence` is not recomputed by the verifier from scratch -- it's the same
`CueMeasurements` object `benchmark/measure.py` already produced, passed
in. This matters more than it looks like it does: CLAUDE.md already states
the rule ("the verifier reuses the same estimators as the front end, kept
in one place so they cannot drift apart") and `benchmark/measure.py` and
`model/qwen_backbone.py`'s `describe_cues` already both consume the one
`CueMeasurements` produced by `cues/`. The verifier's job is not to
re-measure the audio; it's to check the *model's claims* against a
measurement that already exists. Giving the verifier its own, second
measurement pipeline is exactly the kind of drift CLAUDE.md is warning
about -- two implementations of "what is the level difference" that can
silently disagree.

```python
ConsistencyKind = Literal["read_out", "inference"]

@dataclass(frozen=True)
class ClaimCheck:
    quantity: str          # e.g. "coherence", "estimated_azimuth_deg"
    claimed: float | str   # what the transcript asserted
    measured: float | str  # the corresponding CueMeasurements field
    tolerance: float | None
    passed: bool
    consistency_kind: ConsistencyKind
    # step 2/3 quantities (the cue value, the instrument) are read_out --
    # describe_cues already told the model this number. step 1 (regime)
    # and step 4 (the converted azimuth) are inference -- see "The
    # reasoning trace has four steps" above. Get this tag right per
    # quantity once, here, rather than leaving every caller of verify()
    # to remember which fields are which.

@dataclass(frozen=True)
class VerificationResult:
    checks: tuple[ClaimCheck, ...]
    all_passed: bool
    inference_passed: bool  # all_passed restricted to consistency_kind
                             # == "inference" -- this, not all_passed, is
                             # the headline faithfulness number. A trace
                             # that nails read-out and fumbles inference
                             # should not report as faithful.
    unparseable_claims: tuple[str, ...]  # sentences that looked like a
                                          # claim but couldn't be matched
                                          # to a quantity -- see below
```

## Step 1: extracting claims from the transcript

The model doesn't emit structured output today -- `answer()` in
`model/qwen_backbone.py` returns free text. Two ways to get claims out of
it, in order of preference:

1. **Change the output contract first.** Before writing an extractor,
   check whether the reasoning trace can be made structured at the
   source -- e.g. the prompt asks for `azimuth: <value>` /
   `coherence: <value>` lines, or a small JSON block, rather than prose.
   This is cheaper and more reliable than parsing prose, and it's a
   one-line change to `build_messages` in `model/qwen_backbone.py`. Try
   this first.
2. **If prose has to stay** (e.g. because the reasoning trace's fluency is
   itself part of what's being evaluated), regex/number extraction keyed
   off the same vocabulary `describe_cues` already uses ("degrees left",
   "dB", "microseconds", "coherence") is enough for numbers explicitly
   restated. It will not catch a claim like "there is no coherence"
   (Part 6's actual failure) that names a quantity qualitatively without a
   number. That specific case needs a small keyword map (a set of phrases
   like "no coherence", "highly coherent", "decorrelated" mapped to
   coherence-value ranges) -- worth having regardless of which extraction
   approach wins, because it's exactly the failure that motivated this
   spec, and a purely numeric extractor would have missed it. Log every
   sentence that looks like it's making a quantitative claim but doesn't
   match any known pattern into `unparseable_claims` rather than silently
   dropping it -- a growing `unparseable_claims` log is itself a signal
   that the prompt's output contract needs tightening (see point 1).

## Step 2: checking each claim

One check per quantity in `CueMeasurements`, each against the tolerance
already defined in the README's Appendix A table (reproduced here so this
spec doesn't drift from it -- CLAUDE.md: "tolerances are paper claims, not
tunable config"), and each tagged with the `consistency_kind` from the
schema above:

| Quantity | Tolerance | consistency_kind |
|---|---|---|
| Broadband level difference | 1.5 dB | read_out (step 2, amplitude regime) |
| Level difference above 2 kHz | 2.0 dB | read_out |
| Level-difference flatness | 2.0 dB | read_out |
| Inter-channel phase | 10 deg | read_out |
| Inter-channel delay | 30 us | read_out (step 2, delayed regime) |
| Coherence | 0.10 | read_out |
| Fundamental frequency | 3% | read_out (step 3, instrument identity) |
| Regime label | exact match | inference (step 1) |
| Derived azimuth | 5 deg | inference (step 4) |

Regime and azimuth are the only two `inference` rows -- everything else
is a number `describe_cues` states outright under every disclosure, so a
correct claim there is necessary but not sufficient evidence of
reasoning. Under the default `"evidence"` disclosure the two inference
rows are the two things the prompt deliberately does *not* say. This table
is the concrete version of `VerificationResult.inference_passed`: it's
`all(check.passed for check in checks if check.consistency_kind ==
"inference")`, i.e. exactly the regime and azimuth rows.

Two of these need more than "is the number close":

- **Side words** ("left"/"right"), not just the signed number, need their
  own check: `describe_cues` names sides explicitly (CLAUDE.md: "positive
  is left ... is not something the model could know"), so a transcript
  that gets the magnitude right but the side wrong should fail even if
  some numeric tolerance would technically pass a bare `abs(claimed -
  measured)`. Check side and magnitude separately, the same way
  `tests/test_binaural.py` and `tests/test_roundtrip.py` already assert
  the sign convention independently of the magnitude.
- **Regime** ("amplitude" vs "delayed") is categorical, not numeric --
  exact string match against `evidence.estimated_regime`, no tolerance.

`processing` (from `benchmark/schemas.py`'s `RenderPlan.processing`, once
a render has it) is a different kind of ground truth: it's not something
`describe_cues` currently tells the model at all, so there's nothing to
check yet. If a future prompt does start describing planted processing
(e.g. "reverb was applied"), that's a new claim type this spec doesn't
cover yet -- flag it, don't silently ignore it.

## Step 3: what happens on failure

Three options, not mutually exclusive, ranked by how much infrastructure
they need:

1. **Reject and log.** Cheapest, and enough for v1: a failed
   `VerificationResult` marks the item (or the model response, if this
   runs at inference time rather than benchmark-build time) as failed,
   with the specific `ClaimCheck` that failed recorded. This is exactly
   the pattern `benchmark/validate.py` already uses for render rejection
   (`rejected_items.jsonl`, with a `failures: list[str]` per item) --
   reuse that shape rather than inventing a new one.
2. **Flag the step, don't reject the whole trace.** Needs the transcript
   to be a sequence of steps, not one blob of text, so a single wrong
   claim doesn't throw out an otherwise-correct trace. Depends on the
   output-contract change in Step 1 already existing (you can't flag "the
   third step" of unstructured prose).
3. **Feed back into generation** (re-ask, or constrain generation once a
   claim is caught wrong). This is a training/inference-loop design
   question, is genuinely harder, and shouldn't block v1 -- get rejection
   and logging working and correct first, on real transcripts, before
   building anything that acts on the result automatically.

Start with (1). It's the same shape as work that already exists and
already passed review by running against real data (`validate_render`),
and it's enough to answer the only question v1 needs to answer: *does
this transcript's numbers match the waveform, yes or no, and if no,
which ones.* When (1) reports a failure, report `inference_passed`
alongside `all_passed` rather than collapsing to one number -- a read_out
failure and an inference failure are different findings (the front end's
own prompt was garbled vs. the model didn't reason correctly about a
prompt that was fine), and conflating them into a single pass/fail throws
away exactly the distinction this spec exists to preserve.

## Step 4: making claims harder to invert by accident

Part 6's failure wasn't that the model lied -- it's that "coherence:
1.00" is one number sitting next to two others, easy to swap or
misremember, especially because 1.00 sounds like "definitely coherent"
and the model produced the opposite reading anyway. Two changes to
`describe_cues` worth trying, independent of the verifier itself, to make
this specific failure less likely to recur:

- State the coherence claim in words as well as the number, the same way
  level difference already gets "left louder" and delay already gets
  "right channel later" -- e.g. "Coherence: 1.00 (a single compact
  source)." A model that has to contradict an explicit word ("compact")
  rather than just misreading a decimal has a harder time producing "no
  coherence" by accident.
- Consider moving coherence later in the sentence list, after azimuth,
  so it's not the last thing before the model starts generating (recency
  effects in the prompt are a plausible contributor to which value gets
  garbled -- untested here, worth an ablation before committing to it).

Neither of these replaces the verifier -- they're cheap mitigations for
one specific, already-observed failure. The verifier is what catches it
(and everything else) regardless of whether the mitigation works.

## Explicitly out of scope for v1

- Any judge-model or LLM-as-verifier approach. The whole point is that
  every check here is a deterministic recomputation from the waveform,
  not another model's opinion.
- Training-time integration (reward shaping, RL, rejection sampling).
  Get the checker correct and fast first; decide how to use it second.
- Checking claims about `processing` until `describe_cues` actually makes
  such claims (see Step 2).
- Multi-source / mixture audio. `spatialize/mixture.py` and
  `tests/test_mixture.py` exist now and can build ground-truthed
  mixtures, but `CueMeasurements` and `measure_render` still assume one
  source -- there is not yet a `describe_cues` for a mixture prompt, so
  there is nothing for a verifier to check against there yet either. One
  concrete finding already on record that this spec's design should
  account for whenever that changes:
  `test_explore_two_synthetic_tones_summed` shows the *existing*
  single-source level-difference estimator does not average across
  overlapping sources, it locks onto whichever one happens to dominate
  more active frequency bins and silently discards the other. A verifier
  checking a mixture-item claim against that estimator's output would be
  checking against a number that already threw away one of the sources
  -- the estimator itself needs a mixture-aware redesign before a
  mixture verifier is meaningful, not just a new prompt format.

  **The estimator half of that is done.** `cues/mixture.py` estimates the
  full energy-weighted distribution of per-bin level differences over the
  time-frequency plane rather than its median, and reads one peak per
  source; on the real URMP violin/cello duet it recovers both planted
  angles to within 0.1 degrees at three placements, including one only 10
  degrees apart, where `estimate_level_cues` returns a single number
  matching neither. Its resolution limit is the paper's own 5 degree
  azimuth tolerance converted to dB, and it reports an energy fraction per
  peak so a caller can see how much evidence one rests on. What is still
  missing before a mixture item can be verified is the rest of the chain:
  `CueMeasurements` and `measure_render` still carry one source's worth of
  fields, there is no `describe_cues` for a several-source prompt, and no
  question type that asks about more than one instrument.

## Definition of done for v1

- `verify()` exists, takes a transcript and a `CueMeasurements`, returns
  a `VerificationResult` with `inference_passed` computed correctly from
  each check's `consistency_kind`.
- It has known-answer tests the same way every other module in this repo
  does: construct a transcript with a claim planted in it (right and
  wrong), assert the checker recovers whether it was right, per
  CLAUDE.md's rule that a module without a known-answer test isn't
  finished. Include at least one case where read-out passes and
  inference fails, to prove `all_passed` and `inference_passed` can
  actually disagree and the split isn't decorative.
- Run against the real Part 6 transcript (the one this spec opens with)
  and confirm it flags the coherence claim as failed. That transcript
  already exists in `To_Dos.md` and PR #9 -- it costs nothing to check
  against and it's the concrete case this module has to catch.
- `verify_perturbation_pair()` and `verify_ablation_pair()` exist, each
  with a known-answer test of their own (a model that visibly tracks the
  re-rendered angle should pass the perturbation check; a fixed template
  that ignores the injected numbers is the known-bad case both
  diagnostics should catch).
