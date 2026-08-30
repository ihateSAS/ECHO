from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

FIGURES_DIR = Path(__file__).resolve().parents[1] / "paper" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


RED = "#ef1b16"
BLUE = "#4169e1"
GREEN = "#07850a"
NAVY = "#00008b"
GREY = "#8a8a8a"
QWEN = BLUE
GEMINI = GREEN
GPTAUDIO = RED
DARK = NAVY
LIGHT = "#6f91ee"
LAVENDER = "#b79aeb"
PROCESS_BLUE = "#4773c3"
PROMPT_BLUE = "#a9bfe0"
OUTPUT_BLUE = "#8da8d8"
VERIFY_GREY = "#8f8f8f"

plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": "serif",
        "font.serif": ["cmr10", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": "#000000",
        "axes.linewidth": 0.9,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": False,
        "legend.edgecolor": "#000000",
        "legend.fancybox": False,
    }
)


def style_axes(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(True, linestyle="--", linewidth=0.7, color="#bdbdbd", alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", direction="out", length=3)


def panel_label(ax, text: str) -> None:
    ax.set_title(text, fontsize=11, loc="left", pad=8)


def value_labels(ax, bars, values, fmt: str = "{:.1f}", offset: float = 1.5) -> None:
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset, fmt.format(value),
                 ha="center", fontsize=8.5, color="#000000")


def save(fig, name: str) -> None:
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def _overview_card(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str = "#fafafd",
    edgecolor: str = "#d9dce5",
    radius: float = 0.08,
    linewidth: float = 0.9,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle=f"round,pad=0.02,rounding_size={radius}",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=1,
        )
    )


def _overview_label_box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    label: str,
    color: str,
    *,
    fontsize: float = 9.3,
    text_color: str = "#111111",
) -> None:
    ax.add_patch(
        Rectangle(
            xy,
            width,
            height,
            facecolor=color,
            edgecolor="#111111",
            linewidth=0.9,
            zorder=3,
        )
    )
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        label,
        ha="center",
        va="center",
        color=text_color,
        fontsize=fontsize,
        fontfamily="serif",
        linespacing=1.18,
        zorder=4,
    )


def _overview_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.35,
            color="#111111",
            connectionstyle=connectionstyle,
            zorder=2,
        )
    )


def _overview_waveform(ax, x0: float, x1: float, y: float, color: str) -> None:
    x = np.linspace(x0, x1, 220)
    envelope = np.sin(np.linspace(0, np.pi, x.size)) ** 0.7
    signal = np.sin(np.linspace(0, 13 * np.pi, x.size)) * envelope
    ax.plot(x, y + 0.28 * signal, color=color, linewidth=1.7, zorder=3)
    ax.plot([x0, x1], [y, y], color="#c6cad4", linewidth=0.6, zorder=2)


def fig0_framework_and_benchmark() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.5))
    ax.set_xlim(0, 16.8)
    ax.set_ylim(0, 8.2)
    ax.axis("off")

    cue_color = "#f5d994"
    audio_color = "#c9d9e8"
    model_color = "#e7e7e7"
    grounded_color = "#bdd7a8"
    render_color = "#f2d0b5"
    output_color = "#d8c4ae"
    verifier_color = "#bebebe"

    for x in (3.50, 8.18, 11.55):
        ax.plot([x, x], [0.62, 6.78], color="#777777", linewidth=0.6, zorder=0)

    ax.text(
        8.4,
        7.80,
        "ECHO probes where spatial localization fails",
        ha="center",
        va="center",
        fontsize=16,
        fontfamily="cmb10",
    )
    ax.text(
        8.4,
        7.36,
        "StereoMusicQA fixes the physical answer; ECHO compares explicit cue + rule with the full audio route.",
        ha="center",
        va="center",
        fontsize=10.2,
        color="#000000",
        fontfamily="serif",
    )


    ax.text(1.83, 6.48, "1   StereoMusicQA", ha="center", fontsize=11.0, fontfamily="cmb10")
    _overview_waveform(ax, 0.72, 2.95, 5.70, "#6f6f6f")
    ax.text(1.83, 5.18, "Real isolated instrument track", ha="center", fontsize=9.2, fontfamily="serif")
    _overview_arrow(ax, (1.83, 4.95), (1.83, 4.55))
    _overview_label_box(
        ax,
        (0.67, 3.58),
        2.32,
        0.82,
        "Controlled stereo placement\nknown angle  $\\theta$",
        render_color,
        fontsize=8.5,
    )
    _overview_arrow(ax, (1.83, 3.58), (1.83, 3.14))
    _overview_label_box(
        ax,
        (0.60, 2.27),
        2.46,
        0.72,
        "Stereo audio  |  cue  |  angle",
        grounded_color,
        fontsize=8.3,
    )
    ax.text(
        1.83,
        1.33,
        "Reversible placement rule\n+ audio checks",
        ha="center",
        va="center",
        fontsize=9.0,
        color="#000000",
        fontfamily="serif",
    )


    ax.text(5.84, 6.48, "2   Item-matched views", ha="center", fontsize=11.0, fontfamily="cmb10")
    _overview_label_box(
        ax,
        (4.50, 5.48),
        2.68,
        0.62,
        "Same item and target",
        model_color,
        fontsize=9.4,
    )
    _overview_label_box(
        ax,
        (4.10, 3.58),
        3.48,
        1.16,
        "Cue-text\nmono audio\nwritten ILD + equation",
        cue_color,
        fontsize=8.6,
    )
    _overview_label_box(
        ax,
        (4.10, 1.61),
        3.48,
        1.16,
        "Audio-only\ntwo labeled clips\nno written cue or equation",
        audio_color,
        fontsize=8.45,
    )
    _overview_arrow(
        ax,
        (5.58, 5.48),
        (5.18, 4.74),
        connectionstyle="arc3,rad=0.12",
    )
    _overview_arrow(
        ax,
        (6.10, 5.48),
        (6.48, 2.77),
        connectionstyle="arc3,rad=-0.12",
    )
    ax.text(
        5.84,
        1.02,
        "Same target angle  |  same question",
        ha="center",
        fontsize=9.2,
        color="#000000",
        fontfamily="cmb10",
    )


    ax.text(9.87, 6.48, "3   Evaluation", ha="center", fontsize=11.0, fontfamily="cmb10")
    _overview_label_box(
        ax,
        (8.73, 4.72),
        2.28,
        0.86,
        "Same model",
        model_color,
        fontsize=9.2,
    )
    _overview_arrow(ax, (7.52, 4.16), (8.73, 5.10))
    _overview_arrow(
        ax,
        (7.52, 2.19),
        (8.73, 4.94),
        connectionstyle="arc3,rad=-0.12",
    )
    _overview_arrow(ax, (9.87, 4.72), (9.87, 4.28))
    _overview_label_box(
        ax,
        (8.73, 3.42),
        2.28,
        0.68,
        "Side  |  angle  |  steps",
        output_color,
        fontsize=8.2,
    )
    _overview_arrow(ax, (9.87, 3.42), (9.87, 2.98))
    _overview_label_box(
        ax,
        (8.65, 1.82),
        2.44,
        0.98,
        "Premise-grounded\nverifier",
        verifier_color,
        fontsize=8.7,
    )
    ax.text(
        9.87,
        1.10,
        "Scores side, angle,\nand written steps",
        ha="center",
        va="center",
        fontsize=8.7,
        color="#000000",
        fontfamily="serif",
    )


    ax.text(14.10, 6.48, "4   Main finding", ha="center", fontsize=11.0, fontfamily="cmb10")
    ax.add_patch(Rectangle((12.08, 5.92), 0.22, 0.18, facecolor=cue_color, edgecolor="#111111", linewidth=0.7))
    ax.text(12.38, 6.01, "Cue + rule", va="center", fontsize=8.4, fontfamily="serif")
    ax.add_patch(Rectangle((14.10, 5.92), 0.22, 0.18, facecolor=audio_color, edgecolor="#111111", linewidth=0.7))
    ax.text(14.40, 6.01, "Audio-only", va="center", fontsize=8.4, fontfamily="serif")

    bar_x0 = 13.04
    bar_width = 2.70
    baseline_x = bar_x0 + bar_width * 0.455
    ax.plot(
        [baseline_x, baseline_x],
        [2.62, 5.68],
        color="#555861",
        linestyle="--",
        linewidth=0.9,
        zorder=2,
    )
    ax.text(baseline_x, 5.73, "45.5 baseline", ha="center", va="bottom", fontsize=7.4, fontfamily="serif")

    models = ["Qwen2-Audio", "Gemini", "GPT-audio"]
    explicit = [71.9, 93.6, 90.4]
    audio_only = [28.5, 36.6, 42.0]
    row_y = [5.03, 4.02, 3.01]
    for model, cue_value, audio_value, y in zip(models, explicit, audio_only, row_y):
        ax.text(11.93, y, model, va="center", fontsize=8.0, fontfamily="serif")
        cue_w = bar_width * cue_value / 100
        audio_w = bar_width * audio_value / 100
        ax.add_patch(Rectangle((bar_x0, y + 0.06), cue_w, 0.22, facecolor=cue_color, edgecolor="#111111", linewidth=0.6))
        ax.add_patch(Rectangle((bar_x0, y - 0.28), audio_w, 0.22, facecolor=audio_color, edgecolor="#111111", linewidth=0.6))
        ax.text(bar_x0 + cue_w + 0.05, y + 0.17, f"{cue_value:.1f}", va="center", fontsize=7.5, fontfamily="serif")
        ax.text(bar_x0 + audio_w + 0.05, y - 0.17, f"{audio_value:.1f}", va="center", fontsize=7.5, fontfamily="serif")

    ax.text(14.42, 2.56, "Left/center/right accuracy (%)", ha="center", va="top", fontsize=7.8, color="#000000", fontfamily="serif")

    _overview_card(
        ax,
        (12.05, 1.10),
        4.08,
        1.08,
        facecolor="#e5efe1",
        edgecolor="#111111",
        radius=0.0,
        linewidth=0.8,
    )
    ax.text(
        14.09,
        1.74,
        "+43.3 to +57.0 percentage points",
        ha="center",
        va="center",
        fontsize=10.1,
        color="#176b22",
        fontfamily="cmb10",
    )
    ax.text(
        14.09,
        1.38,
        "Audio-only never beats the\nmost-common-label baseline",
        ha="center",
        va="center",
        fontsize=7.8,
        color="#000000",
        fontfamily="serif",
    )

    fig.tight_layout(pad=0.5)
    save(fig, "fig0_framework_and_benchmark.png")


def fig1_reasoning_vs_perception() -> None:
    models = ["Qwen2-Audio", "Gemini\nflash-lite", "GPT-audio"]


    reasoning = [71.9, 93.6, 90.4]
    reasoning_ci = [(65.6, 77.6), (92.5, 94.6), (89.4, 91.5)]
    perception = [28.5, 36.6, 42.0]
    perception_ci = [(21.0, 36.1), (33.7, 39.6), (39.9, 44.0)]
    paired_gap = [43.3, 57.0, 48.4]
    paired_gap_ci = [(35.9, 50.9), (54.0, 60.0), (46.2, 50.8)]
    colors = [QWEN, GEMINI, GPTAUDIO]

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.8))

    ax = axes[0]
    style_axes(ax)
    reasoning_yerr = np.array(
        [[value - low for value, (low, _) in zip(reasoning, reasoning_ci)],
         [high - value for value, (_, high) in zip(reasoning, reasoning_ci)]]
    )
    bars = ax.bar(
        models,
        reasoning,
        yerr=reasoning_yerr,
        capsize=3,
        color=colors,
        width=0.55,
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )
    for bar, value, (_, high) in zip(bars, reasoning, reasoning_ci):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            high + 1.2,
            f"{value:.1f}%",
            ha="center",
            fontsize=8.5,
            color="#000000",
        )
    ax.axhline(45.5, color=GREY, linestyle="--", linewidth=1.1, zorder=2)
    ax.set_ylim(0, 104)
    ax.set_ylabel("Left/center/right accuracy")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    panel_label(ax, "(a) Cue and rule written in the prompt")

    ax = axes[1]
    style_axes(ax)
    perception_yerr = np.array(
        [[value - low for value, (low, _) in zip(perception, perception_ci)],
         [high - value for value, (_, high) in zip(perception, perception_ci)]]
    )
    bars = ax.bar(
        models,
        perception,
        yerr=perception_yerr,
        capsize=3,
        color=colors,
        width=0.55,
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )
    for bar, value, (_, high) in zip(bars, perception, perception_ci):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            high + 1.2,
            f"{value:.1f}%",
            ha="center",
            fontsize=8.5,
            color="#000000",
        )
    ax.axhline(45.5, color=GREY, linestyle="--", linewidth=1.1, zorder=2, label="Most-common-label baseline")
    ax.set_ylim(0, 104)
    ax.set_ylabel("Left/center/right accuracy")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    panel_label(ax, "(b) Full audio route; no written cue or rule")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=1.0)

    ax = axes[2]
    style_axes(ax)
    gap_yerr = np.array(
        [[value - low for value, (low, _) in zip(paired_gap, paired_gap_ci)],
         [high - value for value, (_, high) in zip(paired_gap, paired_gap_ci)]]
    )
    bars = ax.bar(
        models,
        paired_gap,
        yerr=gap_yerr,
        capsize=3,
        color=colors,
        width=0.55,
        edgecolor="black",
        linewidth=0.6,
        zorder=3,
    )
    for bar, value, (_, high) in zip(bars, paired_gap, paired_gap_ci):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            high + 1.0,
            f"+{value:.1f}",
            ha="center",
            fontsize=8.5,
            color="#000000",
        )
    ax.axhline(0.0, color="#333333", linewidth=0.9, zorder=2)
    ax.set_ylim(-5, 66)
    ax.set_ylabel("Cue-text gain (percentage points)")
    panel_label(ax, "(c) Gain on the same items")

    fig.tight_layout()
    save(fig, "fig1_reasoning_vs_perception.png")


def fig2_lookup_table_before_after() -> None:
    metrics = ["Answer on original\n21-angle grid", "Premise-inconsistent\nderivation"]
    original = [100.0, 99.5]
    fixed = [0.0, 0.0]

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})

    ax = axes[0]
    style_axes(ax)
    x = np.arange(len(metrics))
    width = 0.34
    bars1 = ax.bar(x - width / 2, original, width, label="Original adapter", color=RED, edgecolor="black", linewidth=0.6, zorder=3)
    bars2 = ax.bar(x + width / 2, fixed, width, label="Dense-augmented adapter", color=BLUE, edgecolor="black", linewidth=0.6, zorder=3)
    value_labels(ax, bars1, original, fmt="{:.1f}%")
    value_labels(ax, bars2, fixed, fmt="{:.1f}%")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 112)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.set_ylabel("Failure rate (lower is better)")
    panel_label(ax, "(a) Off-grid audit of the reasoning fine-tune")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2, fontsize=8.5, framealpha=1.0)

    ax = axes[1]
    style_axes(ax)
    mae = [4.81, 0.24]
    bars = ax.bar(["Original", "Dense-\naugmented"], mae, color=[RED, BLUE], width=0.5, edgecolor="black", linewidth=0.6, zorder=3)
    value_labels(ax, bars, mae, fmt="{:.2f}°", offset=0.11)
    ax.set_ylim(0, 5.6)
    ax.set_ylabel("Mean absolute error (degrees)")
    panel_label(ax, "(b) Error after dense cue augmentation")

    fig.tight_layout()
    save(fig, "fig2_lookup_table_before_after.png")


def fig3_dithering_replication() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

    ax = axes[0]
    style_axes(ax)
    runs = ["Original\nrun", "Seed-1\nreproduction"]
    accuracy = [75.6, 77.1]
    bars = ax.bar(runs, accuracy, color=[DARK, LIGHT], width=0.45, edgecolor="black", linewidth=0.6, zorder=3)
    value_labels(ax, bars, accuracy, fmt="{:.1f}%")
    ax.axhline(45.5, color=GREY, linestyle="--", linewidth=1.1, zorder=2, label="Most-common-label baseline")
    ax.set_ylim(0, 108)
    ax.set_ylabel("Left/center/right accuracy")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    panel_label(ax, "(a) Original run and seeded reproduction")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=1.0)

    ax = axes[1]
    style_axes(ax)
    x = np.arange(2)
    width = 0.34
    true_left = [99.8, 98.7]
    true_right = [66.5, 70.9]
    bars1 = ax.bar(x - width / 2, true_left, width, label="True label: left", color=BLUE, edgecolor="black", linewidth=0.6, zorder=3)
    bars2 = ax.bar(x + width / 2, true_right, width, label="True label: right", color=RED, edgecolor="black", linewidth=0.6, zorder=3)
    value_labels(ax, bars1, true_left, fmt="{:.1f}%")
    value_labels(ax, bars2, true_right, fmt="{:.1f}%")
    ax.set_xticks(x)
    ax.set_xticklabels(runs)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Accuracy within each true label")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    panel_label(ax, "(b) Left and right behave differently in both runs")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.40), ncol=2, fontsize=8.5, framealpha=1.0)

    fig.tight_layout()
    save(fig, "fig3_dithering_replication.png")


def fig4_appendix_boundary_snap() -> None:
    models = ["Qwen2-Audio", "Gemini Flash-Lite", "GPT-audio"]
    snap_rate = [100.0, 19.7, 1.2]
    accuracy = [0.4, 44.9, 80.7]

    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    style_axes(ax)
    x = np.arange(len(models))
    width = 0.34
    ax.bar(x - width / 2, snap_rate, width, color=RED, edgecolor="black",
           linewidth=0.6, label="Boundary-substitution rate")
    ax.bar(x + width / 2, accuracy, width, color=BLUE, edgecolor="black",
           linewidth=0.6, label="Within-5° accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(-4, 108)
    ax.set_xlim(-0.3, 2.4)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8.5, framealpha=1.0)
    panel_label(ax, "Boundary substitution and exact-angle accuracy")
    save(fig, "fig_appendix_boundary_snap.png")


def fig5_appendix_encoder_probe() -> None:
    conditions = ["Log-mel\n(control)", "Encoder\noutput", "Encoder,\nshuffled labels"]
    r2 = [0.983, 0.953, -0.375]
    colors = [LIGHT, BLUE, GREY]

    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    style_axes(ax)
    bars = ax.bar(conditions, r2, color=colors, width=0.5, edgecolor="black", linewidth=0.6, zorder=3)
    for bar, value in zip(bars, r2):
        y = bar.get_height() + (0.03 if value >= 0 else -0.09)
        ax.text(bar.get_x() + bar.get_width() / 2, y, rf"$R^2$={value:.2f}", ha="center", fontsize=8.5, color="#000000")
    ax.axhline(0, color="#000000", linewidth=0.7, zorder=2)
    ax.set_ylim(-0.55, 1.1)
    ax.set_ylabel(r"Probe prediction score ($R^2$)")
    panel_label(ax, "Angle remains predictable after comparing the channels")
    save(fig, "fig_appendix_encoder_probe.png")


def _heatmap(ax, data: np.ndarray, row_labels: list[str], col_labels: list[str],
             fmt: str = "{:.1f}", vmin: float | None = None, vmax: float | None = None) -> None:
    vmin = data.min() if vmin is None else vmin
    vmax = data.max() if vmax is None else vmax
    im = ax.imshow(data, cmap="Reds", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9, rotation=20, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9.5)
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    threshold = vmin + 0.6 * (vmax - vmin)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            color = "white" if value >= threshold else "black"
            ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=9, color=color)
    return im


def fig6_model_metric_heatmap() -> None:
    rows = ["Qwen2-Audio", "Gemini Flash-Lite", "GPT-audio"]
    cols = ["Cue-text\nwithin $5^{\\circ}$", "Cue-text\nleft/center/right",
            "Written calculation\nfaithfulness", "Audio-only\nleft/center/right"]
    data = np.array([
        [0.4, 71.9, 0.4, 28.5],
        [44.9, 93.6, 42.4, 36.6],
        [80.7, 90.4, 77.8, 42.0],
    ])

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    _heatmap(ax, data, rows, cols, fmt="{:.1f}", vmin=0, vmax=100)
    fig.tight_layout()
    save(fig, "fig6_model_metric_heatmap.png")


def fig7_dithering_confusion_heatmap() -> None:
    row_labels = ["True: left", "True: right", "True: centre"]
    col_labels = ["Stated: left", "Stated: right"]


    original = np.array([
        [99.8, 0.2],
        [33.5, 66.5],
        [90.5, 9.5],
    ])
    replication = np.array([
        [98.7, 1.3],
        [28.8, 70.9],
        [88.4, 11.6],
    ])

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    _heatmap(axes[0], original, row_labels, col_labels, fmt="{:.1f}", vmin=0, vmax=100)
    panel_label(axes[0], "(a) Original run")
    _heatmap(axes[1], replication, [""] * 3, col_labels, fmt="{:.1f}", vmin=0, vmax=100)
    panel_label(axes[1], "(b) Seed-1 reproduction")
    fig.tight_layout()
    save(fig, "fig7_dithering_confusion_heatmap.png")


if __name__ == "__main__":
    fig0_framework_and_benchmark()
    fig1_reasoning_vs_perception()
    fig2_lookup_table_before_after()
    fig3_dithering_replication()
    fig4_appendix_boundary_snap()
    fig5_appendix_encoder_probe()
    fig6_model_metric_heatmap()
    fig7_dithering_confusion_heatmap()
    print("done")
