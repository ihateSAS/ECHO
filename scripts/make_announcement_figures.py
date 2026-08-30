from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

QWEN = "#7c88d8"
GEMINI = "#5fb0a0"
GPTAUDIO = "#e28b6d"
BASELINE = "#a8a8a8"
LIGHT = "#c3cae8"
DARK = "#3d4f9e"
PURPLE = "#9b6bc4"
GRID = "#e4e4e4"
TEXT_LABEL = "#555555"

plt.rcParams.update(
    {
        "font.size": 12,
        "font.family": ["Helvetica Neue", "Arial", "sans-serif"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.edgecolor": "#888888",
        "axes.linewidth": 1.0,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11,
    }
)


def style_axes(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=1.0, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)


def title(ax_or_fig, text: str, size: float = 17) -> None:
    ax_or_fig.set_title(text, fontsize=size, fontweight="bold", color="#1a1a1a", loc="left", pad=16)


def clean_legend(ax, **kwargs) -> None:
    kwargs.setdefault("fontsize", 11)
    leg = ax.legend(
        frameon=True,
        fancybox=True,
        edgecolor="#dddddd",
        facecolor="white",
        borderpad=0.9,
        **kwargs,
    )
    leg.get_frame().set_linewidth(1.0)


def value_labels(ax, bars, values, fmt: str = "{:.1f}%", offset: float = 1.6) -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            fmt.format(value),
            ha="center",
            fontsize=10.5,
            color="#444444",
        )


def save(fig, name: str) -> None:
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def chart_reasoning_within5deg() -> None:
    models = ["Qwen2-Audio", "Gemini flash-lite", "GPT-audio"]
    values = [0.4, 44.9, 80.7]
    colors = [QWEN, GEMINI, GPTAUDIO]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(models, values, color=colors, width=0.5, zorder=3)
    value_labels(ax, bars, values)
    style_axes(ax)
    ax.set_ylim(0, 92)
    ax.set_ylabel("Within 5° of true angle")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    title(ax, "Zero-shot reasoning accuracy, by model")
    save(fig, "reasoning_within5deg_by_model.png")


def chart_boundary_snap_gradient() -> None:
    models = ["Qwen2-Audio", "Gemini flash-lite", "GPT-audio"]
    snap_rate = [100.0, 19.7, 1.2]
    accuracy = [0.4, 44.9, 80.7]

    fig, ax = plt.subplots(figsize=(8, 5.2))
    style_axes(ax)
    x = range(len(models))

    ax.plot(x, snap_rate, "o-", color="#c0605a", linewidth=2.5, markersize=9, zorder=3, label="Boundary-snap rate")
    ax.plot(x, accuracy, "o-", color=DARK, linewidth=2.5, markersize=9, zorder=3, label="Within-5° accuracy")

    ax.set_xticks(list(x))
    ax.set_xticklabels(models)
    ax.set_ylim(-4, 108)
    ax.set_xlim(-0.3, 2.55)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    clean_legend(ax, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    title(ax, "A failure mode that fades as accuracy rises")
    save(fig, "boundary_snap_gradient.png")


def chart_perception_all_fail() -> None:
    models = ["Qwen2-Audio", "Gemini flash-lite", "GPT-audio"]
    values = [28.5, 36.6, 42.0]
    colors = [QWEN, GEMINI, GPTAUDIO]
    baseline = 45.5

    fig, ax = plt.subplots(figsize=(7.5, 5))
    style_axes(ax)
    bars = ax.bar(models, values, color=colors, width=0.5, zorder=3)
    value_labels(ax, bars, values)
    ax.axhline(baseline, color=BASELINE, linestyle="--", linewidth=1.6, zorder=2, label=f"Always-guess-majority ({baseline:.1f}%)")
    ax.set_ylim(0, 55)
    ax.set_ylabel("Side accuracy, from raw audio only")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    clean_legend(ax, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    title(ax, "Perception accuracy, by model")
    save(fig, "perception_all_models_fail.png")


def chart_finetune_split() -> None:
    labels = ["Reasoning\n(cue-text)", "Perception\n(audio-only)"]
    zero_shot = [0.4, 28.5]
    fine_tuned = [100.0, 44.4]

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    style_axes(ax)
    x = [0, 1.15]
    width = 0.34
    bars1 = ax.bar([xi - width / 2 for xi in x], zero_shot, width, label="Zero-shot", color=LIGHT, zorder=3)
    bars2 = ax.bar([xi + width / 2 for xi in x], fine_tuned, width, label="Fine-tuned", color=DARK, zorder=3)
    value_labels(ax, bars1, zero_shot)
    value_labels(ax, bars2, fine_tuned)

    ax.axhline(45.5, color=BASELINE, linestyle="--", linewidth=1.6, zorder=2, label="Always-majority baseline (45.5%)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 112)
    ax.set_xlim(-0.5, 1.9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    clean_legend(ax, loc="upper right")
    title(ax, "Fine-tuning fixes reasoning, not perception")
    save(fig, "finetune_reasoning_vs_perception.png")


def chart_fewshot_pilot() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5), sharey=True)

    gemini_zs, gemini_fs, gemini_base = 35.0, 27.5, 42.5
    gpt_zs, gpt_fs, gpt_base = 50.0, 55.0, 50.0

    for ax, (subtitle, zs, fs, base) in zip(
        axes,
        [
            ("Gemini flash-lite (n=40)", gemini_zs, gemini_fs, gemini_base),
            ("GPT-audio (n=20)", gpt_zs, gpt_fs, gpt_base),
        ],
    ):
        style_axes(ax)
        bars = ax.bar(
            ["Zero-shot", "Few-shot"],
            [zs, fs],
            color=[LIGHT, PURPLE],
            width=0.48,
            zorder=3,
        )
        value_labels(ax, bars, [zs, fs])
        ax.axhline(
            base,
            color=BASELINE,
            linestyle="--",
            linewidth=1.6,
            zorder=2,
            label=f"Subset baseline ({base:.1f}%)",
        )
        ax.set_title(subtitle, fontsize=12.5, color="#333333", pad=10)
        ax.set_ylim(0, 85)
        ax.set_xlim(-0.55, 1.55)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        clean_legend(ax, loc="upper center", fontsize=9.5)

    axes[0].set_ylabel("Side accuracy")
    fig.suptitle("Few-shot examples don't rescue perception", fontsize=17, fontweight="bold", color="#1a1a1a", x=0.02, ha="left", y=1.03)
    fig.tight_layout()
    save(fig, "fewshot_pilot.png")


if __name__ == "__main__":
    chart_reasoning_within5deg()
    chart_boundary_snap_gradient()
    chart_perception_all_fail()
    chart_finetune_split()
    chart_fewshot_pilot()
    print("done")
