# Qwen2-Audio precision comparison: bf16 vs 4-bit

## What this is

Three generations, same source audio, same question, same measured cues.
Only the model's loading precision changes. The point was to check
whether a `--quantization 4bit` load (needed to fit Qwen2-Audio-7B-Instruct
on a 16 GB T4 -- the full bf16 weights are ~16.8 GB and don't fit) costs
anything in answer quality, before relying on it for anything real.

## Setup

Input: `tests/generated/AuSep_2_vc_01_Jupiter_angle_p25deg.wav` (real cello
stem from URMP, panned to +25 degrees with `create_wav_for_angle.py`).
Question: the script's default, "Where is this instrument positioned, and
why?" Measured cues (identical across all three runs, same file):

```
ild_db=19.5, itd_us=0.0, coherence=1.0,
estimated_azimuth_deg=25.0, estimated_regime=amplitude, f0_hz=147.7
```

Full evidence and each run's raw transcript + verifier output are in
`result.json`.

## Runs

| Run | Device | Precision | Tokens | Completed? | `inference_passed` |
|---|---|---|---|---|---|
| `4bit_120tok` | Colab T4 | 4-bit (NF4) | 120 | No -- hit token limit mid-sentence | False |
| `4bit_250tok` | Colab T4 | 4-bit (NF4) | 250 | Yes | False |
| `bf16_250tok` | local CPU | bf16 | 250 | Yes (3213s, ~53.6 min) | **True** |

Commands:

```bash
# 4-bit, on a T4
python scripts/run_qwen_backbone.py tests/generated/AuSep_2_vc_01_Jupiter_angle_p25deg.wav \
  --instrument cello --instrument-code vc --disclosure conclusions \
  --device cuda --quantization 4bit --max-new-tokens 120   # then 250

# bf16, local CPU (or any GPU with >=17GB free)
python scripts/run_qwen_backbone.py tests/generated/AuSep_2_vc_01_Jupiter_angle_p25deg.wav \
  --instrument cello --instrument-code vc --disclosure conclusions \
  --max-new-tokens 250
```

`--disclosure conclusions` was added to these commands after the fact. It
is the prompt these runs actually used -- the one that states the regime
and the azimuth outright -- and it stopped being the default when
`describe_cues` was changed so the verifier's inference checks measure
something. Without the flag these commands would now produce a different
prompt and the transcripts below would not be reproducible.

## What happened

All three runs correctly restated every read-out value it was handed:
level difference, delay, coherence, and (where relevant) pitch. That part
never varied.

The difference is the actual answer to the question asked. Both 4-bit
runs never state a position in degrees anywhere, even the 250-token one
that ran to a natural, complete conclusion ("...a controlled environment
where its sound can be precisely managed for optimal performance" -- not
an answer, just word salad that resembles one). The bf16 run states it
correctly and specifically: "...leaning slightly towards the left as
indicated by the 25.0-degree azimuth angle."

This isn't a case of the model needing to reason something out under
pressure -- `describe_cues()` already states the azimuth outright in the
prompt ("Azimuth implied by those numbers: 25.0 degrees left"), the same
way it states the other cues. So this isn't "4-bit can't do the trig,"
it's "4-bit selectively dropped two of the six facts it was handed
(regime and azimuth) while keeping the other four," across two separate
generations. That's a more typical shape for quantization-induced
degradation to take than an actual reasoning failure would be.

One methodology note worth recording: the first pass at checking the
bf16 transcript missed the azimuth claim entirely, because
`verifier/extract.py`'s regex expected "25 degrees to the left" and the
real text said "...left... 25.0-degree azimuth angle" -- side word and
number in a different clause, reversed order. Fixed by widening
`extract_azimuth` to fall back to a wider-window search when the tight
adjacent pattern doesn't match. Re-verified after the fix; the result
above is post-fix.

## What this means for the project

Generate StereoMusicQA training candidates in bf16, not 4-bit -- on a
rented GPU rather than CPU, since CPU bf16 is far too slow to generate a
real dataset with (53 minutes for one answer). `docs/compute.md` does that
arithmetic: the whole generation pass is a $10-25 job on an hourly A100,
and 60 days of continuous CPU. An earlier version of this paragraph said to
run it on free Modal/Lightning credits; those tiers are far too small for
it, and the compute policy supersedes that. Save 4-bit specifically for the actual
QLoRA fine-tuning step, where the model is being actively trained rather
than asked to already perform well zero-shot -- that's a different
situation and this result doesn't say anything about whether QLoRA
training itself will work.

## Caveats

n=1 comparison pair. Same audio, same angle, one question. This is a
real, clean signal -- not noise, since the only variable that changed
was precision and the outcome flipped exactly on the thing that
mattered -- but it should not be treated as a settled, citable result
until it's repeated across a handful of different angles and
instruments. Cheap to do once there's convenient bf16 GPU access; worth
doing before this goes in the paper as anything more than an
implementation note.
