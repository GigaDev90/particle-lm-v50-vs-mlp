"""Recreate publication figures from the included, sanitized result tables.

This script does not download data, load a model, or train anything.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"MLP": "#b2784d", "V50": "#167b83"}


def read_results(directory: Path):
    summary = json.loads((directory / "summary.json").read_text())
    curves = {"MLP": {}, "V50": {}}
    with (directory / "validation.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            model, step = row["model"], int(row["update"])
            if model not in curves:
                raise ValueError(f"Unknown model: {model}")
            if step in curves[model]:
                raise ValueError(f"Duplicate validation: {model}, update {step}")
            loss = float(row["validation_cross_entropy"])
            ppl = float(row["validation_perplexity"])
            if not (math.isfinite(loss) and math.isfinite(ppl)
                    and math.isclose(math.exp(loss), ppl, rel_tol=1e-12)):
                raise ValueError(f"Invalid validation score: {model}, update {step}")
            curves[model][step] = {
                "tokens": int(row["training_tokens"]), "loss": loss, "perplexity": ppl,
                "training_seconds": float(row["cumulative_training_seconds"]),
            }
    common = sorted((curves["MLP"].keys() & curves["V50"].keys()) - {0})
    stop = int(summary["matched_update"])
    if not common or max(common) != stop:
        raise ValueError("Summary does not describe the longest common evaluated budget")
    if len(common) != summary["matched_trained_validation_points"]:
        raise ValueError("Matched validation count differs from summary")
    for step in common:
        if curves["MLP"][step]["tokens"] != curves["V50"][step]["tokens"]:
            raise ValueError("Matched updates have different token budgets")
    for model in curves:
        if curves[model][stop]["perplexity"] != summary["primary_result"][model]["perplexity"]:
            raise ValueError("CSV endpoint and summary disagree")
    return summary, curves, common


def save(fig, directory: Path, stem: str):
    for suffix in ("svg", "png"):
        metadata = {"Date": None} if suffix == "svg" else {}
        fig.savefig(directory / f"{stem}.{suffix}", dpi=170, bbox_inches="tight", metadata=metadata)
    plt.close(fig)


def make_plots(results: Path, output: Path):
    summary, curves, common = read_results(results)
    output.mkdir(parents=True, exist_ok=True)
    end = summary["matched_update"]
    end_tokens = summary["matched_training_tokens_per_model"]
    x = [curves["MLP"][step]["tokens"] / 1e6 for step in common]
    plt.rcParams.update({
        "font.size": 11, "font.family": "DejaVu Sans", "figure.facecolor": "white",
        "axes.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False,
        "svg.hashsalt": "v50-mlp-publication-v1",
    })
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for model in ("MLP", "V50"):
        y = [curves[model][step]["perplexity"] for step in common]
        label = f"{model} ({summary['primary_result'][model]['parameters']/1e6:.2f}M parameters)"
        axes[0].plot(x, y, color=COLORS[model], lw=2, label=label)
        late = [(a, b) for a, b in zip(x, y) if a >= 100]
        axes[1].plot([a for a, _ in late], [b for _, b in late], color=COLORS[model], lw=2, label=model)
        axes[1].plot(x[-1], y[-1], "o", color=COLORS[model], ms=5)
    axes[0].set(yscale="log", title="All matched checkpoints", xlabel="Sampled training tokens (millions)",
                ylabel="Validation perplexity · log scale", xlim=(0, x[-1]*1.02))
    axes[0].set_yticks([30, 50, 100, 200, 500])
    axes[0].yaxis.set_major_formatter(ScalarFormatter())
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    axes[0].legend(frameon=False, fontsize=10)
    axes[1].set(title="Detail after 100M tokens", xlabel="Sampled training tokens (millions)",
                ylabel="Validation perplexity", xlim=(100, x[-1]*1.02))
    axes[1].legend(frameon=False, fontsize=10)
    endpoint = summary["primary_result"]
    axes[1].text(.98, .94, f"At {end_tokens/1e6:.2f}M tokens\nMLP: {endpoint['MLP']['perplexity']:.3f}\nV50: {endpoint['V50']['perplexity']:.3f}",
                 ha="right", va="top", transform=axes[1].transAxes, fontsize=10)
    for ax in axes:
        ax.grid(alpha=.18)
    fig.suptitle("FineWeb-Edu · V50 versus the standard MLP Transformer", fontsize=14)
    fig.tight_layout(rect=(0, .08, 1, .94))
    fig.text(.5, .015, "Single seed · reused validation sample · lower is better · planned 1B-token training is incomplete",
             ha="center", fontsize=9, color="#555555")
    save(fig, output, "perplexity")

    gap = [100*(curves["V50"][step]["perplexity"] / curves["MLP"][step]["perplexity"]-1) for step in common]
    fig, ax = plt.subplots(figsize=(9, 4.7))
    ax.axhline(0, color="#555555", lw=1)
    ax.plot(x, gap, color=COLORS["V50"], lw=2)
    ax.fill_between(x, gap, 0, color=COLORS["V50"], alpha=.10)
    ax.plot(x[-1], gap[-1], "o", color=COLORS["V50"])
    ax.annotate(f"{gap[-1]:.2f}% at {end_tokens/1e6:.2f}M tokens", xy=(x[-1], gap[-1]),
                xytext=(-155, -35), textcoords="offset points", color=COLORS["V50"],
                arrowprops={"arrowstyle": "-", "color": COLORS["V50"]})
    ax.set(title="V50 perplexity difference at the same training budget", xlabel="Sampled training tokens (millions)",
           ylabel="100 × (V50 perplexity / MLP perplexity − 1)", xlim=(0, x[-1]*1.03), ylim=(min(gap)*1.12, .8))
    ax.grid(alpha=.18)
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.text(.5, .015, f"Negative values favor V50 · {len(common)} matched trained checkpoints · no initial score or MLP-only extension",
             ha="center", fontsize=9, color="#555555")
    save(fig, output, "relative_gap")

    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    names = ["MLP", "V50"]
    minutes = [curves[v][end]["training_seconds"]/60 for v in names]
    ax.bar(names, minutes, width=.55, color=[COLORS[v] for v in names])
    for i, minutes_value in enumerate(minutes):
        ax.text(i, minutes_value+2, f"{minutes_value:.1f} min", ha="center", va="bottom", fontsize=12)
    ax.set(title=f"Recorded training time at {end_tokens/1e6:.2f}M sampled tokens", ylabel="Cumulative logged training minutes",
           ylim=(0, max(minutes)*1.2))
    ax.text(.5, .94, f"V50 / MLP: {minutes[1]/minutes[0]:.2f}×", ha="center", va="top", transform=ax.transAxes)
    ax.yaxis.grid(alpha=.18); ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, .12, 1, 1))
    fig.text(.5, .035, "Historical runs on the same RTX 3080 Laptop GPU; not a controlled benchmark.\n"
             "Excludes validation/reporting/checkpoint time; clocks, temperature and load may differ.",
             ha="center", fontsize=9, color="#555555")
    save(fig, output, "training_time")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    make_plots(args.results, args.output)


if __name__ == "__main__":
    main()
