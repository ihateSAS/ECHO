#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cues.level import estimate_level_cues  # noqa: E402
from cues.mixture import estimate_mixture_level_cues  # noqa: E402
from spatialize.mixture import StemPlacement, mix_stems  # noqa: E402

SURVEY = REPO_ROOT / "experiments/2026-08-09-real-stereo-front-end-survey"
FINETUNE = REPO_ROOT / "experiments/2026-08-10-cues-finetune-fixes-arithmetic"
ENCODER_PROBE = REPO_ROOT / "experiments/2026-08-13-encoder-level-probe"
FIGURES = REPO_ROOT / "figures"
SAMPLE_RATE = 48_000


SERIES_1 = "#2a78d6"
SERIES_2 = "#008300"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"


def _style_axes(ax) -> None:
    ax.grid(True, color=GRIDLINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_color(INK_SECONDARY)


def _save(fig: Figure, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(REPO_ROOT)}")
    return path


def figure_reverb_destroys_azimuth() -> Path | None:
    source = SURVEY / "reverb_wet_sweep.json"
    if not source.is_file():
        print(f"skipping reverb figure: {source} not found")
        return None
    rows = json.loads(source.read_text())["rows"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for angle, colour in ((25.0, "#c1440e"), (10.0, "#1f6feb")):
        points = sorted(
            (row for row in rows if row["planted_azimuth_deg"] == angle),
            key=lambda row: row["wet"],
        )
        wet = [row["wet"] for row in points]
        axes[0].plot(
            wet,
            [row["recovered_azimuth_deg"] for row in points],
            "o-",
            color=colour,
            label=f"planted {angle:+.0f}$\\degree$",
        )
        axes[0].axhline(angle, color=colour, linestyle=":", linewidth=1)
        axes[1].plot(
            wet,
            [row["azimuth_error_deg"] for row in points],
            "o-",
            color=colour,
            label=f"planted {angle:+.0f}$\\degree$",
        )

    axes[0].set_xlabel("reverb wet fraction (RT60 0.6 s)")
    axes[0].set_ylabel("recovered azimuth (degrees)")
    axes[0].set_title("The angle collapses toward centre")
    axes[0].legend()

    axes[1].axhline(5.0, color="black", linestyle="--", linewidth=1)
    axes[1].annotate(
        "5$\\degree$ tolerance",
        xy=(0.30, 5.0),
        xytext=(0.22, 7.5),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none"),
    )
    axes[1].axvline(0.029, color="#c1440e", linestyle=":", linewidth=1)
    axes[1].annotate(
        "wet $\\approx$ 0.029",
        xy=(0.029, 20.0),
        xytext=(0.075, 20.0),
        fontsize=9,
        color="#c1440e",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none"),
    )
    axes[1].set_xlabel("reverb wet fraction (RT60 0.6 s)")
    axes[1].set_ylabel("azimuth error (degrees)")
    axes[1].set_title("Tolerance crossed at an inaudible send")
    axes[1].legend()

    fig.suptitle(
        "Reverb destroys a planted azimuth "
        "(experiments/2026-08-09-real-stereo-front-end-survey)",
        fontsize=10,
    )
    return _save(fig, "reverb_destroys_azimuth.png")


def figure_coherence_tracks_real_width() -> Path | None:
    source = SURVEY / "result.json"
    if not source.is_file():
        print(f"skipping coherence figure: {source} not found")
        return None
    tracks = json.loads(source.read_text())["tracks"]

    side = np.array([track["side_to_total_db"] for track in tracks])
    coherence = np.array([track["summary"]["coherence"]["mean"] for track in tracks])
    spread = np.array(
        [track["summary"]["azimuth_deg"]["peak_to_peak"] for track in tracks]
    )
    correlation = float(np.corrcoef(side, coherence)[0, 1])

    fig, ax = plt.subplots(figsize=(6, 4.5))
    points = ax.scatter(
        side, coherence, c=spread, cmap="viridis", s=90, edgecolor="black", linewidth=0.5
    )
    ax.set_xlabel("side energy relative to total (dB)")
    ax.set_ylabel("mean coherence over 5 s windows")
    ax.set_title(
        f"Coherence reads real stereo width\n"
        f"nine released tracks, r = {correlation:.3f}",
        fontsize=11,
    )
    bar = fig.colorbar(points, ax=ax)
    bar.set_label("within-track azimuth spread (degrees)")
    return _save(fig, "coherence_tracks_real_width.png")


def figure_mixture_finds_both_sources() -> Path | None:
    urmp = REPO_ROOT / "data/URMP/01_Jupiter_vn_vc"
    violin_path = urmp / "AuSep_1_vn_01_Jupiter.wav"
    cello_path = urmp / "AuSep_2_vc_01_Jupiter.wav"
    if not (violin_path.is_file() and cello_path.is_file()):
        print(f"skipping mixture figure: URMP stems not found under {urmp}")
        return None

    import soundfile as sf

    def excerpt(path: Path, seconds: float = 8.0, skip: float = 3.0) -> np.ndarray:
        audio, rate = sf.read(path, dtype="float64")
        start = round(skip * rate)
        return audio[start : start + round(seconds * rate)]

    planted = {"violin": 20.0, "cello": -20.0}
    mix, _ = mix_stems(
        [
            StemPlacement(excerpt(violin_path), planted["violin"], "violin"),
            StemPlacement(excerpt(cello_path), planted["cello"], "cello"),
        ]
    )
    estimate = estimate_mixture_level_cues(mix, SAMPLE_RATE)
    single = estimate_level_cues(mix, SAMPLE_RATE)

    inside = np.abs(estimate.grid_ild_db) <= 30.0
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(
        estimate.grid_ild_db[inside],
        estimate.density[inside] / estimate.density.max(),
        color="#1f6feb",
        alpha=0.25,
    )
    ax.plot(
        estimate.grid_ild_db[inside],
        estimate.density[inside] / estimate.density.max(),
        color="#1f6feb",
    )
    normalised = estimate.density / estimate.density.max()
    for source in estimate.sources:
        ax.axvline(source.ild_db, color="#0b7a3b", linewidth=1.5)
        height = float(
            normalised[int(np.argmin(np.abs(estimate.grid_ild_db - source.ild_db)))]
        )
        ax.annotate(
            f"{source.azimuth_deg:+.1f}$\\degree$\n{source.energy_fraction:.0%}",

            xy=(source.ild_db + (1.6 if source.ild_db > 0 else -1.6),
                min(height + 0.04, 0.92)),
            ha="left" if source.ild_db > 0 else "right",
            va="bottom",
            fontsize=9,
            color="#0b7a3b",
        )
    ax.axvline(
        single.broadband_ild_db,
        color="#c1440e",
        linestyle="--",
        linewidth=1.5,
        label=f"median: {single.azimuth_deg:+.1f}$\\degree$ (neither source)",
    )
    ax.set_xlabel("per-bin inter-channel level difference (dB)")
    ax.set_ylabel("energy-weighted density (normalised)")
    ax.set_ylim(-0.05, 1.15)
    ax.set_title(
        "URMP violin at +20$\\degree$ and cello at -20$\\degree$, mixed\n"
        "cues/mixture.py finds both; the single-source median finds neither",
        fontsize=11,
    )
    ax.legend(loc="lower center", fontsize=9)
    return _save(fig, "mixture_finds_both_sources.png")


def _finetune_answer_pairs() -> list[tuple[float, float]]:
    import re

    source = FINETUNE / "answers.jsonl"
    if not source.is_file():
        return []
    level = re.compile(r"level difference is\s*(-?\d+(?:\.\d+)?)\s*dB")
    azimuth = re.compile(r"azimuth\s*=\s*arctan\([^)]*\)\s*=\s*(-?\d+(?:\.\d+)?)")
    pairs = []
    for line in source.read_text().splitlines():
        if not line.strip():
            continue
        answer = json.loads(line)["answer"]
        found_level = level.search(answer)
        found_azimuth = azimuth.search(answer)
        if found_level and found_azimuth:
            pairs.append((float(found_level.group(1)), float(found_azimuth.group(1))))
    return pairs


def figure_finetune_answers_are_a_lookup_table() -> Path | None:
    pairs = _finetune_answer_pairs()
    if not pairs:
        print("skipping lookup-table figure: no fine-tune answers recorded")
        return None

    seen = sorted(set(pairs))
    counts = {pair: 0 for pair in seen}
    for pair in pairs:
        counts[pair] += 1

    from spatialize.amplitude import SPEAKER_ANGLE_DEG

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    _style_axes(ax)


    level_db = np.linspace(-40.0, 40.0, 800)
    ratio = 10.0 ** (level_db / 20.0)
    coordinate = (ratio - 1.0) / (ratio + 1.0)
    law = np.degrees(np.arctan(coordinate * np.tan(np.radians(SPEAKER_ANGLE_DEG))))
    ax.plot(
        level_db,
        law,
        color=MUTED,
        linewidth=2.0,
        zorder=2,
        label="the panning law, continuous",
    )

    xs = [pair[0] for pair in seen]
    ys = [pair[1] for pair in seen]
    ax.scatter(
        xs,
        ys,
        s=64,
        color=SERIES_1,
        edgecolor="#fcfcfb",
        linewidth=1.4,
        zorder=4,
        label=f"every answer given ({len(seen)} distinct, {len(pairs):,} items)",
    )

    ax.set_xlabel("inter-channel level difference in the prompt (dB)", color=INK_SECONDARY)
    ax.set_ylabel("azimuth the model asserted (degrees)", color=INK_SECONDARY)
    ax.set_title(
        "The fine-tune is exact on 21 points, and is never asked anything else",
        color=INK,
        fontsize=12,
        pad=12,
    )
    ax.set_xlim(-40, 40)
    ax.set_ylim(-32, 32)
    ax.axhline(0, color="#c3c2b7", linewidth=0.8, zorder=1)
    ax.axvline(0, color="#c3c2b7", linewidth=0.8, zorder=1)


    ax.annotate(
        "nothing in the benchmark\nfalls between two points",
        xy=(1.35, 2.5),
        xytext=(9.0, -14.0),
        color=INK_SECONDARY,
        fontsize=9,
        ha="left",
        arrowprops={"arrowstyle": "->", "color": MUTED, "linewidth": 1.0},
    )
    ax.legend(loc="upper left", fontsize=9, frameon=False, labelcolor=INK_SECONDARY)
    return _save(fig, "finetune_answers_are_a_lookup_table.png")


def figure_offgrid_eval_design() -> Path | None:
    from eval.generalization import (
        choose_eval_angles,
        discriminability,
        trained_answer_angles,
    )

    grid = (-25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0)
    trained = trained_answer_angles(grid)
    naive = tuple(angle + 2.5 for angle in grid[:-1])
    chosen = choose_eval_angles(trained, 10)

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(8.6, 6.9), gridspec_kw={"height_ratios": [1.5, 2.0], "hspace": 1.15}
    )

    _style_axes(top)
    top.grid(False)
    top.set_yticks([])
    top.spines["left"].set_visible(False)

    for angle in trained:
        top.axvline(angle, color=MUTED, linewidth=1.6, ymin=0.62, ymax=1.0, zorder=3)
    top.scatter(
        naive, [0.36] * len(naive), s=58, color=SERIES_2, marker="v", zorder=4,
        label="halfway between the grid angles",
    )
    top.scatter(
        chosen, [0.12] * len(chosen), s=58, color=SERIES_1, marker="^", zorder=4,
        label="chosen: widest gaps in the trained set",
    )
    top.text(
        0, 1.14, f"the {len(trained)} trained answers",
        color=INK_SECONDARY, fontsize=9, ha="center", va="bottom",
        transform=top.get_xaxis_transform(),
    )
    top.set_xlim(-31, 31)
    top.set_ylim(-0.02, 1.02)
    top.set_xlabel("azimuth (degrees)", color=INK_SECONDARY)
    top.set_title(
        "An off-grid eval has to avoid the mid/side angles, not just the grid",
        color=INK, fontsize=12, pad=26,
    )
    top.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.42), fontsize=9,
        frameon=False, labelcolor=INK_SECONDARY, ncol=2, handletextpad=0.5,
    )

    _style_axes(bottom)
    labels = ["within 5 deg\n(the headline metric)", "answers on a\ntrained angle"]
    naive_result = discriminability(naive, trained)
    chosen_result = discriminability(chosen, trained)
    memorised = [
        naive_result.memorises.within_tolerance,
        chosen_result.memorises.on_grid_rate,
    ]
    computed = [
        naive_result.computes.within_tolerance,
        chosen_result.computes.on_grid_rate,
    ]
    positions = np.arange(len(labels), dtype=float)
    width = 0.30
    offset = width / 2 + 0.025
    bars_m = bottom.bar(
        positions - offset, memorised, width, color=SERIES_2,
        label="if it memorised the table", zorder=3,
    )
    bars_c = bottom.bar(
        positions + offset, computed, width, color=SERIES_1,
        label="if it learned the law", zorder=3,
    )
    for bars in (bars_m, bars_c):
        for bar in bars:
            bottom.annotate(
                f"{bar.get_height():.0%}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", fontsize=10, color=INK_SECONDARY,
            )
    bottom.set_xticks(positions)
    bottom.set_xticklabels(labels, fontsize=10)
    bottom.set_xlim(-0.6, len(labels) - 0.4)
    bottom.set_ylim(0, 1.3)
    bottom.set_yticks([0, 0.5, 1.0])
    bottom.set_yticklabels(["0%", "50%", "100%"])
    bottom.set_title(
        "Identical bars mean the metric cannot tell the two apart",
        color=INK, fontsize=11, pad=30,
    )
    bottom.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.20), fontsize=9,
        frameon=False, labelcolor=INK_SECONDARY, ncol=2, handletextpad=0.5,
    )
    return _save(fig, "offgrid_eval_design.png")


def figure_encoder_keeps_the_level_difference() -> Path | None:
    source = ENCODER_PROBE / "result.json"
    if not source.is_file():
        print("skipping encoder-probe figure: probe has not been run")
        return None
    data = json.loads(source.read_text())

    stages = [
        ("log-mel features\n(encoder input)", "log_mel_control", SERIES_1),
        ("encoder output\n(what the LLM sees)", "encoder_output", SERIES_2),
    ]
    if "encoder_shuffled_labels" in data:
        stages.append(
            ("same features,\nshuffled labels", "encoder_shuffled_labels", MUTED)
        )
    errors = [data[key]["mae_deg"] for _, key, _ in stages]
    r_squared = [data[key]["r2"] for _, key, _ in stages]

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    _style_axes(ax)
    positions = np.arange(len(stages), dtype=float)
    bars = ax.bar(
        positions, errors, 0.46, color=[color for _, _, color in stages], zorder=3
    )
    for bar, r2 in zip(bars, r_squared):
        ax.annotate(
            f"{bar.get_height():.1f} deg\nR2 = {r2:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            color=INK_SECONDARY,
        )
    ax.axhline(5.0, color="#c3c2b7", linewidth=1.4, linestyle="--", zorder=2)
    ax.text(
        -0.42, 5.5, "5 deg tolerance", color=INK_SECONDARY, fontsize=9, ha="left"
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([label for label, _, _ in stages], fontsize=10)
    ax.set_ylabel("error decoding the planted azimuth (degrees)", color=INK_SECONDARY)
    ax.set_title(
        "The pan survives both stages: the encoder does not discard it\n"
        f"linear probe, leave-one-excerpt-out, {data['n_items']} items",
        color=INK,
        fontsize=12,
        pad=14,
    )
    ax.set_xlim(-0.6, len(stages) - 0.4)
    ax.set_ylim(0, max(errors) * 1.32)
    fig.tight_layout()
    return _save(fig, "encoder_keeps_the_level_difference.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", default=None, help="substring of a figure name to draw just one"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figures = {
        "reverb": figure_reverb_destroys_azimuth,
        "coherence": figure_coherence_tracks_real_width,
        "mixture": figure_mixture_finds_both_sources,
        "lookup": figure_finetune_answers_are_a_lookup_table,
        "offgrid": figure_offgrid_eval_design,
        "encoder": figure_encoder_keeps_the_level_difference,
    }
    drawn = 0
    for name, draw in figures.items():
        if args.only is not None and args.only not in name:
            continue
        if draw() is not None:
            drawn += 1
    print(f"\n{drawn} figure(s) written to {FIGURES.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
