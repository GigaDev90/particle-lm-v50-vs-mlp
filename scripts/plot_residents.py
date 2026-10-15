"""Regenerate resident figures from the included aggregate measurements.

No checkpoint, source text, GPU, or training is needed. The document-balanced
snapshot and the one-window training history have different sampling scopes.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEAL, ORANGE, PURPLE, GREY = "#167b83", "#b2784d", "#77639c", "#a2a8af"
BLOCK_COLORS = {1: TEAL, 3: ORANGE, 8: PURPLE}
SNAPSHOT = "V50 · 500.04M training tokens · CPU FP32 · 256 evaluation documents / 8,192 positions"


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def save(fig, output, name, note, bottom=.13):
    fig.tight_layout(rect=(0, bottom, 1, .93))
    fig.text(.5, .018, note, ha="center", va="bottom", fontsize=9, color="#555555")
    for extension in ("png", "svg"):
        metadata = {"Date": None} if extension == "svg" else {}
        fig.savefig(output / f"{name}.{extension}", dpi=170, bbox_inches="tight", metadata=metadata)
    plt.close(fig)


def make_plots(results, output):
    summary = json.loads((results / "activity-summary.json").read_text())
    residents = rows(results / "residents.csv")
    history = rows(results / "history.csv")
    clouds = json.loads((results / "cloud-views.json").read_text())["views"]
    blocks = summary["blocks"]
    if [b["block"] for b in blocks] != list(range(1, 9)) or len(residents) != 3488:
        raise ValueError("Expected eight blocks and 3,488 resident records")
    if len({(r["block"], r["space"], r["resident"]) for r in residents}) != 3488:
        raise ValueError("Resident records are not unique")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.hashsalt": "v50-residents-500m-v1"})
    x = np.arange(1, 9)

    # Positive gates and effective contributors answer different questions.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    means = np.array([100*b["active_fraction"]["mean"] for b in blocks])
    ci = np.array([b["active_fraction"]["ci95"] for b in blocks])*100
    axes[0].bar(x, means, color=[ORANGE if b == 3 else TEAL for b in x], width=.68)
    axes[0].errorbar(x, means, yerr=np.maximum(0, np.stack((means-ci[:, 0], ci[:, 1]-means))),
                     fmt="none", color="#333333", capsize=3, lw=1)
    for i, value in enumerate(means):
        axes[0].text(x[i], value+2.2, f"{value:.1f}%", ha="center", fontsize=9)
    axes[0].set(title="How often is a distance gate positive?", xlabel="Transformer block",
                ylabel="Token–resident pairs with a positive gate (%)", xticks=x, ylim=(0, 108))
    for key, label, color in (("active_residents_per_space", "Positive gates", TEAL),
                              ("effective_gate_residents", "Effective gate count", ORANGE),
                              ("effective_response_magnitude_residents", "Effective response magnitude", PURPLE)):
        values = [b[key]["mean"] if isinstance(b[key], dict) else b[key] for b in blocks]
        axes[1].plot(x, values, "o-", color=color, label=label, lw=2, ms=4)
    axes[1].set(title="Many nonzero responses carry little weight", xlabel="Transformer block",
                ylabel="Residents per query per space (of 109)", xticks=x, ylim=(0, 112))
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.set_axisbelow(True); ax.yaxis.grid(alpha=.18)
    fig.suptitle("Resident activity is strongly dependent on depth", fontsize=15)
    save(fig, output, "resident_activity", SNAPSHOT+"\nError bars: document-bootstrap 95% intervals; effective counts are concentration measures.")

    usage = np.empty((32, 109))
    for r in residents:
        usage[(int(r["block"])-1)*4+int(r["space"])-1, int(r["resident"])-1] = float(r["eval_active_fraction"])
    if not np.isfinite(usage).all() or not ((usage >= 0) & (usage <= 1)).all():
        raise ValueError("Invalid resident activity frequencies")
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(100*usage, aspect="auto", cmap="viridis", vmin=0, vmax=100,
                   extent=(.5, 109.5, 32.5, .5), interpolation="nearest")
    ax.set(yticks=np.arange(1, 33), yticklabels=[f"B{b} · S{s}" for b in x for s in range(1, 5)],
           xlabel="Resident identity within its own space (1–109)", ylabel="Block and routing space")
    ax.tick_params(axis="y", labelsize=8)
    for boundary in np.arange(4.5, 32, 4): ax.axhline(boundary, color="white", lw=1.3)
    fig.colorbar(im, ax=ax, pad=.02, label="Evaluation positions with a positive gate (%)")
    fig.suptitle("Some residents respond broadly; others rarely activate", fontsize=14)
    save(fig, output, "resident_usage", SNAPSHOT+"\nColumns in different spaces are different residents. Zero here means unused in this sample, not globally inactive.", bottom=.10)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, title, ylabel, color in (
        (axes[0], "movement_degrees", "Resident direction changes from initialization", "Net angular displacement (degrees)", TEAL),
        (axes[1], "radius", "Learned influence radii", "Chord-distance radius on the unit sphere", ORANGE)):
        data = [np.array([float(r[key]) for r in residents if int(r["block"]) == b]) for b in x]
        boxes = ax.boxplot(data, positions=x, widths=.55, whis=(5, 95), showfliers=False,
            patch_artist=True, medianprops={"color": "#222222"}, showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": color, "markersize": 4})
        for patch in boxes["boxes"]: patch.set(facecolor=color, alpha=.35)
        ax.set(title=title, xlabel="Transformer block", ylabel=ylabel, xticks=x)
        ax.set_axisbelow(True); ax.yaxis.grid(alpha=.18)
    initial_radius = np.array([float(r["initial_radius"]) for r in residents])
    if np.ptp(initial_radius) < 1e-5:
        axes[1].axhline(initial_radius.mean(), ls="--", color="#666666", label=f"Initial radius {initial_radius.mean():.3f}")
        axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle("How all 3,488 residents moved and changed their influence range", fontsize=15)
    save(fig, output, "resident_movement", "V50 · 500.04M tokens · 436 residents per block\nBox: middle 50%; line: median; dot: mean; whiskers: 5th–95th percentiles. Net motion is not travelled path.")

    # Separate views of the same 128D vectors; do not draw projected radius circles.
    selected = [v for v in clouds if v["block"] in (1, 3, 8) and v["title"] in ("Query PCA", "Resident PCA")]
    fig, axes = plt.subplots(3, 2, figsize=(10, 11))
    for row, b in enumerate((1, 3, 8)):
        for column, title in enumerate(("Query PCA", "Resident PCA")):
            view = next(v for v in selected if v["block"] == b and v["title"] == title)
            ax = axes[row, column]
            q, p, p0 = [np.asarray(view[key]) for key in ("q", "p", "p0")]
            ax.scatter(q[:, 0], q[:, 1], s=7, color=TEAL, alpha=.16, linewidths=0)
            ax.scatter(p0[:, 0], p0[:, 1], s=19, color=GREY, marker="x", alpha=.8, linewidths=.8)
            ax.scatter(p[:, 0], p[:, 1], s=17, color=ORANGE, alpha=.9, linewidths=0)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set(title=f"Block {b} · space 1 · {title}\nVariance shown: queries {view['query_variance_fraction']:.1%}, residents {view['resident_variance_fraction']:.1%}",
                   xlabel="Projection coordinate 1", ylabel="Projection coordinate 2")
            ax.grid(alpha=.12)
    handles = [Line2D([], [], marker="o", ls="", color=TEAL, label="Evaluation queries"),
               Line2D([], [], marker="x", ls="", color=GREY, label="Initial residents"),
               Line2D([], [], marker="o", ls="", color=ORANGE, label="Trained residents")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(.5, .956), frameon=False)
    fig.suptitle("Query and resident clouds: two views of each 128D space", fontsize=15, y=.995)
    save(fig, output, "resident_geometry", "V50 · 500.04M tokens · fixed basis and shared origin within each panel\nQuery PCA uses reference documents; dots show separate evaluation documents.\n2D projections hide dimensions and do not establish neighbor distances or sphere coverage.", bottom=.105)

    steps = sorted({int(r["update"]) for r in history})
    lookup = {(int(r["update"]), int(r["block"])): r for r in history}
    if max(steps) != 15260 or len(lookup) != len(steps)*8:
        raise ValueError("Incomplete 500M resident history")
    tokens = np.array([int(lookup[s, 1]["tokens"]) for s in steps])/1e6
    activity = np.array([[100*float(lookup[s, b]["active_fraction"]) for s in steps] for b in x])
    edges = np.r_[0, (tokens[:-1]+tokens[1:])/2, tokens[-1]]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    im = ax.pcolormesh(edges, np.arange(.5, 9.5), activity, vmin=0, vmax=100,
                      shading="flat", cmap="viridis", rasterized=True)
    ax.set(yticks=x, ylim=(8.5, .5), xlabel="Sampled training tokens (millions)", ylabel="Transformer block")
    fig.colorbar(im, ax=ax, pad=.02, label="Positive gates (%)")
    fig.suptitle("The blocks develop different activity patterns during training", fontsize=15)
    save(fig, output, "resident_history", "One fixed 512-token development window; only its first 256 positions are diagnosed.\nGenuine validation events only. This is a trajectory probe, not a population-wide estimate.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for key, label, color in (("query_mean_norm", "Query centroid", TEAL), ("resident_mean_norm", "Resident centroid", ORANGE)):
        axes[0].plot(x, [b[key] for b in blocks], "o-", color=color, lw=2, label=label, ms=4)
    axes[0].set(xticks=x, xlabel="Transformer block", ylabel="Centroid norm of unit vectors", ylim=(0, 1.04),
                title="Concentration measured in all 128 dimensions")
    axes[0].legend(frameon=False, fontsize=9)
    budgets = [8, 16, 32, 64]
    for b in (1, 3, 8):
        row = blocks[b-1]
        for selection, style in (("dynamic", "-"), ("fixed", "--")):
            values = [100*row["budgets"][str(k)][selection+"_response_magnitude"] for k in budgets]
            axes[1].plot(budgets, values, style, marker="o" if selection == "dynamic" else None,
                         color=BLOCK_COLORS[b], label=f"B{b} {selection}", lw=1.8, ms=4)
    axes[1].set(xticks=budgets, xlabel="Residents retained per space", ylabel="Unsigned response magnitude retained (%)",
                ylim=(0, 104), title="Ranking by gate: per query versus a fixed set")
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    for ax in axes: ax.grid(alpha=.18)
    fig.suptitle("Concentrated queries and concentrated contributions", fontsize=15)
    save(fig, output, "resident_concentration", SNAPSHOT+"\nFixed rankings use reference documents; empty query–space events receive zero retention.\nUnsigned magnitude ignores cross-resident cancellation; this is not a pruning-accuracy result.", bottom=.16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/residents")
    parser.add_argument("--output", type=Path, default=ROOT / "figures/residents")
    args = parser.parse_args()
    make_plots(args.results, args.output)


if __name__ == "__main__":
    main()
