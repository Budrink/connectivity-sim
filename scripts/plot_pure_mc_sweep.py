#!/usr/bin/env python3
"""Analyze pure Monte Carlo transport sweep results."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os

csv = sys.argv[1] if len(sys.argv) > 1 else "data/sweep_mc_pure_128.csv"
df = pd.read_csv(csv)
N = df["grid_N"].iloc[0]

df["stable"] = (~df["collapsed"].astype(bool)) & (df["confinement"] > 1.5) & (df["barrier_aniso"] > 1.0)
df["score"] = df["confinement"] * df["barrier_aniso"] / (1.0 + df["effort"])

os.makedirs("figures", exist_ok=True)

# 1. Controller stability comparison (with vs without coupling)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, coup_label, coup_cond in zip(
    axes,
    ["No coupling (α=0)", "With coupling (α>0)"],
    [df["coupling_alpha"] == 0, df["coupling_alpha"] > 0]
):
    sub = df[coup_cond]
    ctrl_stab = sub.groupby("controller")["stable"].mean().sort_values(ascending=False)
    ctrl_stab.plot.bar(ax=ax, color=["#2ecc71" if v > 0.5 else "#e74c3c" for v in ctrl_stab.values])
    ax.set_title(f"{coup_label}")
    ax.set_ylabel("Fraction stable")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)
fig.suptitle(f"Controller stability — Pure MC transport ({N}×{N})", fontsize=14)
fig.tight_layout()
fig.savefig(f"figures/ctrl_stability_mc_{N}.png", dpi=150)
print(f"Saved ctrl_stability_mc_{N}.png")

# 2. Controller necessity: stable WITH controller coupling vs WITHOUT
fig, ax = plt.subplots(figsize=(10, 5))
ctrls = df["controller"].unique()
x = np.arange(len(ctrls))
w = 0.35

for i, (label, cond) in enumerate([
    ("α = 0 (no control)", df["coupling_alpha"] == 0),
    ("α > 0 (active control)", df["coupling_alpha"] > 0)
]):
    rates = []
    for c in ctrls:
        sub = df[cond & (df["controller"] == c)]
        rates.append(sub["stable"].mean() if len(sub) > 0 else 0)
    ax.bar(x + i * w, rates, w, label=label)

ax.set_xticks(x + w/2)
ax.set_xticklabels(ctrls, rotation=45)
ax.set_ylabel("Fraction stable")
ax.set_title(f"Controller necessity — Pure MC ({N}×{N})")
ax.legend()
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(f"figures/ctrl_necessity_mc_{N}.png", dpi=150)
print(f"Saved ctrl_necessity_mc_{N}.png")

# 3. Heater × Controller heatmap
pivot = df.groupby(["heater", "controller"])["stable"].mean().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, rotation=45)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        ax.text(j, i, f"{pivot.values[i,j]:.0%}", ha="center", va="center", fontsize=9)
plt.colorbar(im, ax=ax, label="Stability fraction")
ax.set_title(f"Heater × Controller stability — Pure MC ({N}×{N})")
fig.tight_layout()
fig.savefig(f"figures/heater_ctrl_heatmap_mc_{N}.png", dpi=150)
print(f"Saved heater_ctrl_heatmap_mc_{N}.png")

# 4. D_E effect on stability
fig, ax = plt.subplots(figsize=(8, 5))
for ctrl in ctrls:
    sub = df[df["controller"] == ctrl]
    de_stab = sub.groupby("D_E")["stable"].mean()
    ax.plot(de_stab.index, de_stab.values, "o-", label=ctrl)
ax.set_xlabel("D_E (diffusion coefficient)")
ax.set_ylabel("Fraction stable")
ax.set_title(f"Stability vs D_E — Pure MC ({N}×{N})")
ax.legend()
ax.set_xscale("log")
fig.tight_layout()
fig.savefig(f"figures/de_stability_mc_{N}.png", dpi=150)
print(f"Saved de_stability_mc_{N}.png")

# 5. Best configurations (top 20 by score, stable only)
stable = df[df["stable"]].sort_values("score", ascending=False).head(20)
print(f"\n=== Top 20 stable configs ({N}×{N}) ===")
print(stable[["controller", "heater", "heat_rx", "ctrl_gain", "coupling_alpha",
              "D_E", "confinement", "barrier_aniso", "effort", "score"]].to_string(index=False))

# 6. Coupling strength effect
fig, ax = plt.subplots(figsize=(8, 5))
for ctrl in ctrls:
    sub = df[df["controller"] == ctrl]
    coup_stab = sub.groupby("coupling_alpha")["stable"].mean()
    ax.plot(coup_stab.index, coup_stab.values, "o-", label=ctrl)
ax.set_xlabel("Coupling α")
ax.set_ylabel("Fraction stable")
ax.set_title(f"Stability vs coupling — Pure MC ({N}×{N})")
ax.legend()
fig.tight_layout()
fig.savefig(f"figures/coupling_stability_mc_{N}.png", dpi=150)
print(f"\nSaved coupling_stability_mc_{N}.png")

# 7. Summary stats
total_stable = df["stable"].mean()
print(f"\nOverall stability: {total_stable:.1%}")
print(f"Stable with α=0: {df[df['coupling_alpha']==0]['stable'].mean():.1%}")
print(f"Stable with α>0: {df[df['coupling_alpha']>0]['stable'].mean():.1%}")
print(f"Stable without MC controller: {df[df['coupling_alpha']==0]['stable'].mean():.1%}")

plt.close("all")
