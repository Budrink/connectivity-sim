#!/usr/bin/env python3
"""Single hero image for the heating-width sweep.

Designed for LinkedIn: square canvas, 1 big chart + 2 compact panels,
large fonts, clean layout, one message.

Usage:
    python3 scripts/plot_heating_hero.py heating_sweep.csv [out=heating_hero.png]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch


CORD_RADIUS = 0.05


def pick_colors(widths):
    cmap = plt.cm.plasma
    n = len(widths)
    return {w: cmap(0.05 + 0.85 * i / max(n - 1, 1)) for i, w in enumerate(widths)}


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("heating_sweep.csv")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("heating_hero.png")

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path).sort_values(["heat_r", "t"]).reset_index(drop=True)
    widths = sorted(df["heat_r"].unique())
    colors = pick_colors(widths)

    t_cut = df["t"].quantile(0.3)
    steady = df[df["t"] >= t_cut]
    agg = (
        steady.groupby("heat_r")
              .agg(core=("center_E", "mean"), edge=("edge_E", "mean"))
              .reset_index()
              .sort_values("heat_r")
    )
    final = (
        df.sort_values("t")
          .groupby("heat_r")
          .agg(cum=("wall_flux_cum", "last"))
          .reset_index()
          .sort_values("heat_r")
    )

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#2b2b2b",
        "axes.linewidth": 1.1,
        "xtick.color": "#2b2b2b",
        "ytick.color": "#2b2b2b",
        "axes.facecolor": "#fbfbfb",
        "figure.facecolor": "white",
    })

    fig = plt.figure(figsize=(12, 12), constrained_layout=False)
    gs = gridspec.GridSpec(
        nrows=3, ncols=2,
        height_ratios=[0.10, 0.58, 0.32],
        hspace=0.50, wspace=0.22,
        left=0.08, right=0.94, top=0.97, bottom=0.08,
    )

    # -------------------- TITLE BAND ------------------------------------
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(
        0.0, 0.75,
        "Narrow plasma heating leaks 6× more than wide",
        fontsize=29, fontweight="bold", color="#0e0e0e",
        ha="left", va="center",
    )
    ax_title.text(
        0.0, 0.18,
        "narrow heating → cold edge → weak ionization → leak     "
        "│     wide heating → warm edge → plasma holds",
        fontsize=12.5, color="#444", fontweight="bold",
        ha="left", va="center",
    )
    ax_title.text(
        0.0, -0.18,
        "toy Monte-Carlo plasma  ·  6 Gaussian heating profiles  ·  "
        "identical total injected power  ·  cord radius = 0.05",
        fontsize=11, color="#888",
        ha="left", va="center",
    )

    # -------------------- HERO ------------------------------------------
    ax = fig.add_subplot(gs[1, :])

    for w in widths:
        g = df[df["heat_r"] == w]
        ax.plot(g["t"], g["wall_flux_cum"],
                color=colors[w], lw=3.0, solid_capstyle="round",
                label=f"σ = {w:.02f}    ({w/CORD_RADIUS:.1f}× cord)")

    t_end = df["t"].max()
    ax.set_xlim(df["t"].min(), t_end * 1.15)
    y_top = df["wall_flux_cum"].max()
    ax.set_ylim(0, y_top * 1.05)

    # Labels right at curve ends
    for w in widths:
        g = df[df["heat_r"] == w]
        y = g["wall_flux_cum"].iloc[-1]
        ax.annotate(
            f"  {y:.0f}",
            xy=(t_end, y), xytext=(t_end + 0.4, y),
            fontsize=12, color=colors[w], fontweight="bold",
            va="center", ha="left", clip_on=False,
        )

    ax.set_xlabel("time  (simulation units)", labelpad=6)
    ax.set_ylabel("cumulative plasma energy → wall", labelpad=6)
    ax.set_title("Integrated leakage over the run", loc="left", pad=10)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(
        loc="upper left", fontsize=11, framealpha=0.95,
        title="heating Gaussian σ", title_fontsize=11,
    )

    # ──────── "6× spread" callout on the right margin ────────
    y_worst = final.loc[final["cum"].idxmax(), "cum"]
    y_best  = final.loc[final["cum"].idxmin(), "cum"]
    arrow_x = t_end * 1.12
    arr = FancyArrowPatch(
        (arrow_x, y_worst), (arrow_x, y_best),
        arrowstyle="<->", mutation_scale=20,
        lw=2.2, color="#222",
    )
    ax.add_patch(arr)
    ax.text(
        arrow_x * 1.005, (y_worst + y_best) / 2,
        "6×", fontsize=18, fontweight="bold", color="#222",
        ha="left", va="center",
    )


    # -------------------- BOTTOM-LEFT: core vs edge E ---------------------
    ax1 = fig.add_subplot(gs[2, 0])
    x = agg["heat_r"].values
    ax1.fill_between(x, 0, agg["core"], color="#c0392b", alpha=0.12)
    ax1.plot(x, agg["core"], "-o", color="#c0392b", lw=2.8, ms=10,
             label="core E")
    ax1.plot(x, agg["edge"], "-o", color="#1f77b4", lw=2.8, ms=10,
             label="edge E")
    ax1.axvline(CORD_RADIUS, color="#555", ls=":", lw=1.2, alpha=0.7)

    core_max = float(agg["core"].max())
    ax1.set_ylim(0, core_max * 1.22)
    ax1.text(
        CORD_RADIUS, core_max * 1.18,
        "cord radius", fontsize=10, color="#555",
        rotation=0, va="center", ha="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                  ec="#ccc", lw=0.8),
    )

    peak_idx = int(np.argmax(agg["core"].values))
    peak_w = float(agg["heat_r"].iloc[peak_idx])
    peak_c = float(agg["core"].iloc[peak_idx])
    ax1.annotate(
        f"core peak  σ≈{peak_w:.02f}\n({peak_w/CORD_RADIUS:.1f}× cord)",
        xy=(peak_w, peak_c),
        xytext=(peak_w + 0.07, peak_c * 0.62),
        fontsize=11, color="#c0392b", fontweight="bold",
        ha="left",
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.3),
    )
    ax1.set_xlabel("heating Gaussian σ")
    ax1.set_ylabel("steady-state mean energy")
    ax1.set_title("Where the energy sits", loc="left", pad=8)
    ax1.grid(alpha=0.25, linestyle="--")
    ax1.legend(loc="lower right", fontsize=11, framealpha=0.95)

    # -------------------- BOTTOM-RIGHT: bars ------------------------------
    ax2 = fig.add_subplot(gs[2, 1])
    xs = np.arange(len(final))
    bars = ax2.bar(
        xs, final["cum"],
        color=[colors[w] for w in final["heat_r"]],
        edgecolor="#222", linewidth=0.8,
    )
    ax2.set_xticks(xs)
    ax2.set_xticklabels(
        [f"{w:.02f}\n({w/CORD_RADIUS:.1f}×)" for w in final["heat_r"]],
        fontsize=10,
    )
    ax2.set_xlabel("heating σ   (× cord radius)")
    ax2.set_ylabel("total energy lost to wall")
    ax2.set_title("Integrated leakage  —  6× spread", loc="left", pad=8)

    ymax = float(final["cum"].max())
    ax2.set_ylim(0, ymax * 1.28)
    for bar, val in zip(bars, final["cum"]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.02,
            f"{val:.0f}", ha="center", fontsize=11, fontweight="bold",
            color="#222",
        )
    ax2.grid(axis="y", alpha=0.25, linestyle="--")

    worst_idx = int(np.argmax(final["cum"].values))
    best_idx  = int(np.argmin(final["cum"].values))
    for idx, label, color in [
        (worst_idx, "worst", "#7a1032"),
        (best_idx,  "best",  "#1a4f0a"),
    ]:
        bar = bars[idx]
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.17,
            label,
            fontsize=11, fontweight="bold", color=color,
            ha="center", va="bottom",
        )

    # -------------------- FOOTER ------------------------------------------
    fig.text(
        0.08, 0.018,
        "custom CUDA Monte-Carlo transport  ·  radiation + ohmic heating + "
        "external B-field  ·  6000 steps × 6 configs",
        fontsize=10, color="#999",
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
