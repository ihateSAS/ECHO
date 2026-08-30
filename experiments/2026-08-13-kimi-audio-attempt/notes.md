# Kimi-Audio: attempted, dropped after a rough real pilot

## What this is

An attempt to add Kimi-Audio-7B-Instruct as a fourth model in the
multi-model comparison. Documented here per the project's convention of
recording what was tried, not just what worked, so this isn't silently
lost or re-attempted from scratch later without the context of why it
stopped.

## Two paths tried

**Self-hosting on Modal** (`scripts/modal_generate_kimi.py`,
`model/kimi_backbone.py`), mirroring the Qwen2-Audio setup. Hit four
distinct, real dependency failures in sequence, each fixed before the
next surfaced:

1. `flash-attn` (a hard dependency of Kimi-Audio's own `pyproject.toml`)
   needs `nvcc`/CUDA_HOME at image-build time, which Modal's default
   build environment doesn't have. Fixed by switching to an
   `nvidia/cuda:12.4.0-devel-ubuntu22.04` base image with a GPU attached
   to that specific build step.
2. `--no-build-isolation` (needed for the fix above) disabled pip's
   automatic build environment, and `wheel`/`setuptools` weren't
   otherwise installed, so `bdist_wheel` failed. Fixed by installing them
   explicitly first.
3. Kimi-Audio's dependency list doesn't pin `transformers`, and the
   version pip resolved to had a real upstream bug (`NameError: name
   'torch' is not defined` inside `transformers/integrations/
   tensor_parallel.py`, an import-order bug in that release). Fixed by
   pinning `transformers==4.46.0` explicitly before Kimi-Audio's own
   install ran.
4. With that pin, the main 7B model loaded fine, but Kimi-Audio's bundled
   internal Whisper-large-v3 sub-component failed loading its checkpoint
   (`AttributeError: 'NoneType' object has no attribute 'get'` inside
   `transformers/modeling_utils.py`'s `from_pretrained`), a version
   mismatch between the pinned `transformers` and whatever the bundled
   checkpoint's format expects. Upstream's own `requirements.txt` doesn't
   pin `transformers` either, so there was no known-good version to copy;
   the next step would have been guessing at a different pin.

Abandoned at this point in favor of a hosted API instead of continuing
to guess at version combinations with no authoritative reference.

**Replicate-hosted API** (`model/providers.py::KimiAudioLLM`, provider
name `kimi-audio`), using a community Cog deployment
(`zsxkib/kimi-audio-7b-instruct`) the same way the OpenAI/Gemini adapters
work: one audio file, one text prompt, one text reply. This actually
worked end to end (confirmed via `scripts/api_generate.py --provider
kimi-audio`), but a real 15-item pilot on the cue-text condition surfaced
three separate concerns before any full run was considered:

1. **A synthetic sanity-check test got the side wrong** on an
   unambiguous case (20 dB louder on the left; the reply was "Right, 0, 0").
2. **Answer formatting was inconsistent run to run**: `"Left, 0"`,
   `"Right,30"` (no space), bare `"Left"` with no number at all,
   `"left, -30"`. The shared extractor (`verifier/extract.py`) couldn't
   parse any of these -- none include the word "degrees" or "°", which
   every other model's answers do. Not fixed; would need a new pattern
   for a bare `"side, number"` shape, the same category of fix as the
   earlier ° and first-vs-last-claim bugs, just not applied here since
   the model was dropped before it seemed worth it.
3. **13 of 15 items needed retries** for transient `ReplicateError`s
   (all eventually succeeded via the existing retry-with-backoff logic,
   no data lost) -- a much higher transient-failure rate than Gemini or
   GPT-audio ever showed. This is plausibly specific to this community
   deployment rather than Kimi-Audio itself, but there's no way to tell
   from outside it.

## What the 15-item pilot actually showed (read by hand, since the
## extractor couldn't parse it)

| Metric | Value |
|---|---|
| Side accuracy (by hand) | 10/15 = 66.7% |
| Boundary-snap rate among numbered answers | 9/13 = 69.2% (exactly ±30°) |

Both numbers are, on their face, consistent with the project's broader
story -- 66.7% side accuracy is in the same range as Qwen2-Audio's
zero-shot 71.9%, and the boundary-snapping rate would have added a
fourth independent confirmation of the same mechanism seen in
Qwen2-Audio, Gemini, and GPT-audio. **This is the frustrating part:** the
signal looks real and on-narrative, but it's sitting on top of a
deployment that showed real reliability and consistency problems at
n=15 already, which makes it hard to trust without a much larger,
cleaner run to confirm it wasn't noise or an artifact of the specific
community hosting.

## Why this was dropped, not just paused

Given the accumulated friction (four distinct dependency failures
self-hosting, then three distinct reliability/formatting concerns on the
API path), the cost of getting a trustworthy number out of this specific
deployment was judged not worth it relative to the payoff -- three
models (Qwen2-Audio, Gemini, GPT-audio) already form a real, defensible
cross-model comparison. A fourth data point that might just be noise from
an unreliable host isn't worth the further engineering time right now.

## What's preserved, if this is revisited

- `model/kimi_backbone.py`, `scripts/modal_generate_kimi.py` -- the
  self-hosted path, four real fixes documented above already applied. The
  next unresolved issue (the Whisper sub-component's `transformers`
  version mismatch) is exactly where to pick back up.
- `model/providers.py::KimiAudioLLM`, registered as `"kimi-audio"` in
  `scripts/api_generate.py` -- the API path, fully working plumbing.
  Revisiting this would mean either extending `verifier/extract.py` for
  the bare `"side, number"` format and running a larger pilot on the
  same Replicate deployment, or finding a more reliable hosted option.
- `answers_cues_pilot.jsonl` in this folder -- the 15 real answers this
  was based on.
