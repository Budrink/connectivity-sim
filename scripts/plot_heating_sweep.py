#!/usr/bin/env python3
"""Plot heating-width sweep — narrow vs sweet-spot vs wide.

Reads CSV produced by ``aniso_sweep_heating`` (columns:
``tag,heat_r,heater_power,step,t,center_E,edge_E,confinement,
wall_flux_step,wall_flux_cum,max_Jz,total_E``) and renders:
  * ``<prefix>_leakage.png``  — headline: cumulative wall flux vs time
  * ``<prefix>_panel.png``    — 2x2 panel (leakage / E-profile / core-E / summary)
  * ``<prefix>_summary.png``  — bar chart: leakage & mean core-E vs width

Usage:
    python3 scripts/plot_heating_sweep.py heating_sweep.csv [out_prefix]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("heating_sweep.csv")
    prefix   = sys.argv[2] if len(sys.argv) > 2 else "heating_sweep"

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df = df.sort_values(["heat_r", "t"]).reset_index(drop=True)

    cord_radius = 0.05
    widths = sorted(df["heat_r"].unique())
    power_by_w = (
        df.groupby("heat_r")["heater_power"].first().to_dict()
        if "heater_power" in df.columns else {w: float("nan") for w in widths}
    )

    cmap = plt.cm.plasma
    colors = {r: cmap(i / max(len(widths) - 1, 1)) for i, r in enumerate(widths)}

    def label_full(r: float) -> str:
        ratio = r / cord_radius
        P = power_by_w.get(r, float("nan"))
        if P == P:
            return f"σ={r:.02f}  (σ/rcord={ratio:.1f},  P_peak={P:.0f})"
        return f"σ={r:.02f}  (σ/rcord={ratio:.1f})"

    def label_short(r: float) -> str:
        return f"σ={r:.02f}  (σ/rcord={r/cord_radius:.1f})"

    def smooth(y, n=5):
        y = np.asarray(y, dtype=float)
        if len(y) < 2:
            return y
        k = np.ones(n) / n
        return np.convolve(y, k, mode="same")

    # --------------------------------------------------------- HEADLINE PLOT
    fig_h, ax_h = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for r in widths:
        g = df[df["heat_r"] == r]
        ax_h.plot(g["t"], g["wall_flux_cum"],
                  color=colors[r], lw=2.4, label=label_full(r))
    ax_h.set_xlabel("t  (sim time)")
    ax_h.set_ylabel("Σ wall flux   (cumulative plasma energy → wall)")
    ax_h.set_title(
        "Cold-edge disruption loop — narrow heating leaks ~6× more than wide\n"
        "toy MC plasma  (cord r=0.05,  B_ext=6,  V_loop=10,  rad α=0.05,  ∫Q·dV = const)",
        fontsize=12,
    )
    ax_h.grid(alpha=0.25)
    ax_h.legend(title="heating Gaussian σ", loc="upper left", fontsize=9)
    fig_h.savefig(f"{prefix}_leakage.png", dpi=160, bbox_inches="tight")
    print(f"saved {prefix}_leakage.png")
    plt.close(fig_h)

    # --------------------------------------------------------- 2x2 PANEL
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    ax = axes[0, 0]
    for r in widths:
        g = df[df["heat_r"] == r]
        ax.plot(g["t"], g["wall_flux_cum"],
                color=colors[r], lw=2.0, label=label_short(r))
    ax.set_xlabel("t")
    ax.set_ylabel("Σ wall flux (cumulative)")
    ax.set_title("Leakage vs heating width")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    for r in widths:
        g = df[df["heat_r"] == r]
        ax.plot(g["t"], g["center_E"], color=colors[r], lw=2.0,
                label=f"core σ={r:.02f}")
        ax.plot(g["t"], g["edge_E"], color=colors[r], lw=1.0,
                ls="--", alpha=0.65)
    ax.set_xlabel("t")
    ax.set_ylabel("E (solid = core, dashed = edge)")
    ax.set_title("Core vs edge energy")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    t_cut = df["t"].quantile(0.3)
    steady = df[df["t"] >= t_cut]
    agg = (
        steady.groupby("heat_r")
              .agg(mean_core=("center_E", "mean"),
                   mean_edge=("edge_E", "mean"))
              .reset_index()
              .sort_values("heat_r")
    )
    x = agg["heat_r"].values
    ax.plot(x, agg["mean_core"], "-o", color="tab:red",
            lw=2.2, ms=8, label="mean core E")
    ax.plot(x, agg["mean_edge"], "-o", color="tab:blue",
            lw=2.2, ms=8, label="mean edge E")
    ax.axvline(cord_radius, color="k", ls=":", alpha=0.5, label="cord radius")
    ax.set_xlabel("heating σ")
    ax.set_ylabel(f"E averaged over t ≥ {t_cut:.1f}")
    ax.set_title("Steady-state core/edge energy vs width")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    final = (
        df.sort_values("t")
          .groupby("heat_r")
          .agg(final_cum=("wall_flux_cum", "last"))
          .reset_index()
          .sort_values("heat_r")
    )
    bars = ax.bar(np.arange(len(final)), final["final_cum"],
                  color=[colors[r] for r in final["heat_r"]],
                  edgecolor="k")
    ax.set_xticks(np.arange(len(final)))
    ax.set_xticklabels([f"{r:.02f}\n({r/cord_radius:.1f}×)"
                        for r in final["heat_r"]], fontsize=9)
    ax.set_xlabel("heating σ   (×cord radius)")
    ax.set_ylabel("total leakage  Σ wall flux  at t=end")
    ax.set_title("Integrated leakage: narrower heating → more leakage")
    for bar, val in zip(bars, final["final_cum"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + final["final_cum"].max() * 0.01,
                f"{val:.0f}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Heating profile width vs confinement — toy MC plasma",
        fontsize=14,
    )
    fig.savefig(f"{prefix}_panel.png", dpi=150, bbox_inches="tight")
    print(f"saved {prefix}_panel.png")
    plt.close(fig)

    # --------------------------------------------------------- SUMMARY
    fig_s, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    # left y — cumulative leakage
    bars = ax.bar(np.arange(len(final)), final["final_cum"],
                  color="#b0b0b0", edgecolor="k", label="total leakage")
    ax.set_xticks(np.arange(len(final)))
    ax.set_xticklabels([f"σ={r:.02f}\n(σ/rcord={r/cord_radius:.1f})"
                        for r in final["heat_r"]], fontsize=10)
    ax.set_ylabel("total Σ wall flux  (cumulative)", color="#404040")
    ax.tick_params(axis="y", colors="#404040")
    for bar, val in zip(bars, final["final_cum"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + final["final_cum"].max() * 0.015,
                f"{val:.0f}", ha="center", fontsize=10, color="#404040")

    # right y — steady-state core E
    ax2 = ax.twinx()
    ax2.plot(np.arange(len(agg)), agg["mean_core"],
             "-o", color="tab:red", lw=2.5, ms=10,
             label="mean core E (steady state)")
    ax2.set_ylabel("mean core E  (t ≥ 30% of run)", color="tab:red")
    ax2.tick_params(axis="y", colors="tab:red")

    # Combine legends
    ln1, lb1 = ax.get_legend_handles_labels()
    ln2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(ln1 + ln2, lb1 + lb2, loc="upper left", fontsize=10)

    ax.set_title(
        "Cold-edge disruption loop: narrow heating leaks, wide heating holds\n"
        "edge E controls ionization → magnetic confinement; "
        "core E peaks where σ ≈ 2-3 × r_cord",
        fontsize=12,
    )
    fig_s.savefig(f"{prefix}_summary.png", dpi=160, bbox_inches="tight")
    print(f"saved {prefix}_summary.png")
    plt.close(fig_s)

    # --------------------------------------------------------- TEXT SUMMARY
    summary = (
        df.groupby("heat_r")
          .agg(final_cum_wall=("wall_flux_cum", "last"),
               mean_core_E=("center_E", "mean"),
               mean_edge_E=("edge_E", "mean"),
               mean_conf=("confinement", "mean"),
               mean_max_Jz=("max_Jz", "mean"))
          .reset_index()
    )
    summary["P_peak"] = summary["heat_r"].map(power_by_w)
    summary["sigma_over_rcord"] = summary["heat_r"] / cord_radius
    cols = ["heat_r", "sigma_over_rcord", "P_peak",
            "final_cum_wall", "mean_core_E", "mean_edge_E", "mean_conf"]
    print("\nSummary:")
    print(summary[cols].to_string(index=False,
                                  float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
