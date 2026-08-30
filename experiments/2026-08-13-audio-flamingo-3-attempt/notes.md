# Audio Flamingo 3: attempted, dropped, worst side accuracy of any model tested

## What this is

An attempt to add NVIDIA's Audio Flamingo 3 as a fourth/fifth model,
tried after Kimi-Audio was dropped. Unlike Kimi-Audio, AF3 is natively
supported in the standard `transformers` library
(`AudioFlamingo3ForConditionalGeneration`), so self-hosting risk looked
much lower going in. Went with the same Replicate-hosted path as Kimi
first anyway (`model/providers.py::AudioFlamingo3LLM`, provider name
`audio-flamingo-3`, deployment `zsxkib/audio-flamingo-3`) since a working
API already existed and was cheaper to validate before investing in
self-hosting.

```bash
python scripts/api_generate.py --provider audio-flamingo-3 --limit 15
```

## The real pilot result: worse than it looks at a glance, and worse than
## every other model tested

A naive read of the 15 real answers (just the stated left/right word)
looked reasonable: 11/15 = 73.3% matched the true side. But that's not
the project's actual scoring method, and rerunning it the same rigorous
way every other model was scored (extracting the model's own stated
azimuth via `verifier/extract.py`, then re-deriving side from that
signed value) gives a very different, much worse number:

| Metric | Value |
|---|---|
| Side accuracy (of all 15, rigorous) | **40.0%** |
| Within 5deg | 6.7% |
| Coverage | 93.3% |

**Why the gap between the naive and rigorous number is the actual
finding.** AF3's answers are frequently self-contradictory: it states
"left" as the side, then gives an azimuth of exactly **0deg** (which
under this project's own convention, and its own front end, means
*centred*, not left) -- six of its "left" answers do this. On "right"
answers it states azimuths of **120deg or 135deg**, which are physically
impossible under the amplitude-panning law explained in its own prompt
(max +-30deg per `docs/PROJECT_STATUS.md`'s physics section). It looks
like AF3 is drawing on some other spatial-audio convention it learned
elsewhere (a 360-degree bearing system, maybe) rather than the specific
physics this task actually describes.

This is a genuinely different failure mode from every other model tested
so far. Qwen2-Audio, Gemini, GPT-audio, and Kimi-Audio all snap to the
panning law's *correct, in-range* +-30deg boundary when they fail at the
arithmetic -- wrong magnitude, but internally consistent with the stated
side and physically valid. AF3's failure is a genuine self-contradiction:
the stated side word and the stated number disagree with each other, and
the numbers are frequently outside the range the task's own physics
allows.

## Same reliability problem as Kimi-Audio

13 of 15 items needed retries for transient `ReplicateError`s (all
eventually succeeded, no data lost, same retry-with-backoff logic as
every other run). Both AF3 and Kimi-Audio are packaged by the same
Replicate maintainer (`zsxkib`), and both showed this same
much-higher-than-normal transient-failure rate compared to Gemini or
GPT-audio's own APIs. Plausibly a property of this specific packager's
infrastructure rather than either model, but there's no way to confirm
that from outside it.

## Why this was dropped

40.0% side accuracy is the worst of any model tested in this project so
far -- below Qwen2-Audio's zero-shot 71.9%, and below Kimi-Audio's
by-hand 66.7% (`experiments/2026-08-13-kimi-audio-attempt/`). Combined
with the same reliability friction Kimi-Audio showed, this wasn't judged
worth scaling to a full run. Dropped for the same reason as Kimi: three
models (Qwen2-Audio, Gemini, GPT-audio) already form a real, defensible
comparison, and neither of the two attempted additions cleared the bar
to justify further investment.

## What's preserved, if this is revisited

- `model/providers.py::AudioFlamingo3LLM`, registered as
  `"audio-flamingo-3"` -- the Replicate API path, fully working plumbing,
  4 offline tests passing.
- `answers_cues_pilot.jsonl` in this folder -- the 15 real answers this
  was based on.
- Not attempted: self-hosting AF3 directly via `transformers`
  (`AudioFlamingo3ForConditionalGeneration`), which was the lower-risk
  path relative to Kimi-Audio's custom package, and would be the
  reasonable next step if AF3 is revisited -- the poor result here might
  be specific to this community Replicate deployment's configuration
  (temperature, system prompt, or another default this adapter doesn't
  control) rather than the model itself.
