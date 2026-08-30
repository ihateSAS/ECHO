# The cue front end on real stereo music

## What this is

Every azimuth this project has measured so far is one the project planted.
That is why the front end can be trusted, and it is also the obvious hole a
reviewer goes for: the benchmark currently measures how well `cues/` inverts
`spatialize/`, not how well it reads an actual mix. A real record is many
sources at once, through real room reverb, buss compression, and stereo
width that was already there.

So: run the existing front end over records nobody here rendered, and write
down what happens. There is no ground truth, so most of this is not
pass/fail. What it can answer is whether the numbers stay **stable** (a mix
does not swap sides between bar one and bar two, so successive windows
should agree) and **plausible** (a finished mix is roughly level-balanced,
so a whole-mix reading should sit near centre), plus one part that *is*
pass/fail — a planted angle put through real processing on real musical
content.

## Setup

Nine tracks from the Free Music Archive small subset, via the Hugging Face
mirror `benjamin-paine/free-music-archive-small`, picked by fixed row
offsets spread across the 7,916 rows, one track per artist. 30 seconds
each, 44.1 kHz stereo MP3. Titles, artists and FMA links are in
`tracks.json`; the audio is not in the repo and is not meant to be.

These are real independent releases — mixed, mastered, and published — not
major-label masters. Two of the nine are live concert recordings, which
turned out to matter. Genres span hip-hop, singer-songwriter, Balkan brass,
psych-rock, post-rock, and choral.

```bash
python experiments/2026-08-09-real-stereo-front-end-survey/fetch_tracks.py
python scripts/survey_real_stereo.py data/real_stereo \
    --json experiments/2026-08-09-real-stereo-front-end-survey/result.json
python experiments/2026-08-09-real-stereo-front-end-survey/reverb_wet_sweep.py \
    --json experiments/2026-08-09-real-stereo-front-end-survey/reverb_wet_sweep.json
```

Each track is measured in non-overlapping 5 second windows at 48 kHz (48
windows over the nine tracks), using `benchmark/measure.py` — the same
`measure_render` the benchmark builder calls, not a parallel copy. Full
per-window readings and all 225 control measurements are in `result.json`.

## What the mixes read as

| Track | Artist | side/total | coherence | azimuth | spread | ITD |
|---|---|---|---|---|---|---|
| `fma_00000` | AWOL (hip-hop) | −12.3 dB | 0.886 | −3.80° | 1.26° | −0.1 µs |
| `fma_00900` | Alice Rose (singer-songwriter) | −5.2 dB | 0.770 | −2.43° | 6.38° | +0.0 µs |
| `fma_01800` | Black Bear Combo (Balkan) | −2.8 dB | 0.506 | +1.62° | 2.21° | +0.1 µs |
| `fma_02700` | Strange Forces (psych-rock) | −3.0 dB | 0.460 | −0.62° | 1.43° | +0.2 µs |
| `fma_03600` | Zlatni Makedoncinja (live) | −11.4 dB | 0.877 | −0.14° | 0.26° | **+23.2 µs** |
| `fma_04500` | LOWdown (hip-hop) | −25.5 dB | 1.000 | +0.78° | 0.01° | −0.0 µs |
| `fma_05400` | Yale Women's Slavic Chorus (live) | −16.3 dB | 0.983 | +0.48° | 0.08° | −0.1 µs |
| `fma_06300` | Andy G. Cohen (post-rock) | −10.0 dB | 0.942 | −1.63° | 2.64° | +0.2 µs |
| `fma_07100` | Sinapsi (post-rock) | **+2.1 dB** | **0.099** | −3.60° | **14.09°** | +1.8 µs |

`spread` is peak-to-peak azimuth across that track's windows. Azimuth and
ITD are means.

**Nothing fell apart.** Every estimator ran on every window, no crashes, no
NaNs, no wild values. That was not guaranteed: these are dense polyphonic
mixes, and the level estimator's active-bin count ran from 613 to 1,797
across the 48 windows, so even its thinnest window had hundreds of bins with
real energy in them rather than a handful of survivors.

**Whole-mix readings are plausible.** Every track reads within ±3.8° of
centre, which is what a level-balanced mix should do. Seven of nine hold
within 1° standard deviation window to window.

**Coherence is the cue that earns its place here.** It spans nearly its
whole range across nine tracks — 0.099 to 1.000 — and it lines up with an
independent measurement of how wide each mix actually is: correlation with
side/total energy is **r = −0.846**. It is not a direction cue and does not
pretend to be, but it reads real stereo width on real records, not just on
the widening this project applies itself.

**Real recordings really do carry an ITD.** `fma_03600` is a live Balkan
brass band and shows a steady **+23.2 µs ± 1.2** across all six windows —
consistent to about a microsecond, so this is a genuine inter-channel delay
from a real microphone pair, not estimator noise. It sits just under the
paper's 30 µs delay tolerance, so the regime classifier still calls it
"amplitude". Worth knowing: that tolerance is not comfortably above what
real recordings show, it is barely above it. A slightly wider mic spacing
would put a real acoustic recording over the line.

**Where it does wobble, it wobbles honestly.** `fma_07100` is the widest
track in the set (more energy in L−R than in the average channel) and it is
the one that will not sit still: 14.09° of azimuth spread and coherence
0.099. The regime classifier returned "ambiguous" on **every** window rather
than committing to a number. That is the correct behaviour — there is no
single source position to report — and it means the instability is visible
in the output rather than hidden inside a confident-looking angle.

**Pitch on a full mix is not an instrument's f0**, and should not be read as
one. On the synthetic chord in `tests/test_real_stereo.py` pYIN returns
55 Hz for a stack whose lowest note is 110 Hz — the missing fundamental of a
2:3 interval, an octave below anything being played. The survey records the
mix f0 as an observation, not as a cue.

## The control: a planted angle through real processing

Same real audio, downmixed to mono to throw away whatever stereo the record
had, re-panned at a known angle, then put through processing chains built
from `ProcessingStep`s and dispatched by `benchmark/process.py` — the
benchmark's own reverb and compressor, not a second implementation. Worst
azimuth error over all nine tracks and five angles, against the 5° tolerance:

| Chain | Worst error | |
|---|---|---|
| `dry` | 0.00° | ok |
| `compression` (−18 dB, 4:1) | 0.00° | ok |
| `width` (mid/side 1.5) | 4.81° | ok, and exactly predictable |
| `reverb` (RT60 0.6 s, 30% wet) | **24.52°** | **over tolerance** |
| `mastered` (all three) | **24.56°** | **over tolerance** |

The dry result is honest but free: an amplitude pan multiplies one mono
signal by two constants, so every bin carries the same ratio and the median
returns it exactly, whatever the music is. It confirms the loading and
windowing path is sound and says nothing else. Stereo-linked compression is
free for the same reason — one shared gain envelope scales both channels
equally.

Mid/side width is off by up to 4.81°, and every bit of that is the
documented `d' = width · d` relationship rather than estimator error: a +10°
pan at width 1.5 lands at +14.815° and the closed form predicts +14.815°, to
four decimal places. That error is correctable, because it is a known
function of a known parameter.

**Reverb is the finding.** At an ordinary 30% wet send, a source planted at
+25° measures back at +1.6°. The tail is excited by the mono sum, so it is
centred, and adding centred energy drags the per-bin level ratio toward
unity at every bin at once — which the median across bins does nothing to
resist, because it is a systematic shift rather than a few bad bins. This is
`spatialize/reverb.py` working exactly as documented ("tends to pull the
estimated position toward the center"); what was not known before this run
is the size.

`reverb_wet_sweep.json` has the curve. At +25° with RT60 0.6 s:

| wet | recovered | error | coherence |
|---|---|---|---|
| 0.00 | +25.00° | 0.00° | 1.000 |
| 0.01 | +24.46° | 0.54° | 0.949 |
| 0.02 | +23.02° | 1.98° | 0.820 |
| 0.025 | +21.41° | 3.59° | 0.743 |
| **0.03** | **+19.58°** | **5.42°** | 0.666 |
| 0.05 | +15.17° | 9.83° | 0.415 |
| 0.10 | +7.83° | 17.17° | 0.152 |
| 0.30 | +1.63° | 23.37° | 0.022 |

Interpolating the measured points, the 5° tolerance is crossed at **wet ≈
0.029**. That is not a reverb setting anyone would use on a record; it is
barely audible. Smaller pans have more room — at +10° the crossing is at wet
≈ 0.080 — but the ceiling is set by the widest angle in the config, and
StereoMusicQA goes to ±25°.

## What this means for the project

1. **A reverb-processed benchmark item cannot carry an azimuth ground truth.**
   `benchmark/validate.py` currently returns early on any item with
   `plan.processing` set, skipping the azimuth, ITD, regime, and coherence
   checks — deliberately, and with a good reason written down, since reverb
   is *supposed* to lower coherence and mid/side is *supposed* to move the
   angle. But the effect of that early return is that a reverb item at 30%
   wet would be accepted while carrying `azimuth_deg = 25.0` as the answer to
   "where is this instrument?", when nothing in the audio supports 25°. The
   question would be unanswerable from the waveform, and the verifier would
   then be checking a model against a number the model has no way to reach.
   Before reverb is turned on in a config, `validate_render` needs a
   processed-item rule: either re-derive the effective ground truth per
   processing type (mid/side can, via `d' = width · d`) or drop the azimuth
   question for items where it cannot be re-derived and ask a coherence or
   regime question instead.

2. **Coherence is the cue to build the realistic-processing items around.**
   It is the one that behaved well on real music, it responds
   monotonically to wet level, and it does not need a planted angle to be
   meaningful.

3. **The 30 µs delay tolerance is tight against reality.** A real live
   recording in this set sits at 23 µs. Nothing to change today, but the
   assumption "amplitude-panned studio music has zero ITD" is an assumption
   about studio music specifically, and it does not survive contact with a
   microphone pair.

4. **A polarity inversion reads as a plausible angle.** Found while checking
   mid/side: width past `d' = 1` flips the right channel's sign, and the
   front end reports +25.48° for a +25° pan at width 1.5 — a number that
   looks right and means nothing, arrived at by folding back through the
   range limit. The tell is `itd_peak_strength` collapsing to 0.000 while
   coherence stays at 1.000: a decorrelating process lowers coherence, and
   only a sign flip kills the GCC-PHAT peak while leaving coherence alone.
   The regime label ("delayed") and the ITD (±103.65 µs) are both wrong in
   this state. Pinned in
   `tests/test_real_stereo.py::test_a_polarity_inversion_reads_as_a_dead_gcc_phat_peak`,
   and a good candidate for a verifier check.

## Caveats

Nine tracks, 30 seconds each, one 5-second window granularity. FMA
independent releases, not major-label masters, and two of the nine are live
recordings — a set of nine cannot be representative of "commercial stereo".
Everything is 44.1 kHz MP3, so there is lossy-codec pre-echo and joint-stereo
coding in the input, which is realistic for distributed music but is not what
a mastering-grade WAV would give.

The r = −0.846 between coherence and side energy is nine points. It is a
clean, expected relationship rather than a surprising one, so it is
believable, but it is not a measurement anyone should cite.

None of this is a claim about accuracy on real music, because there is no
ground truth to be accurate against. The only pass/fail numbers here are the
controls, and those are planted.
