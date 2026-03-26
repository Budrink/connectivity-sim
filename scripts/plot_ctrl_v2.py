#!/usr/bin/env python3
"""Analyze controller v2 (fluctuation damping) sweep results."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os

csv = sys.argv[1] if len(sys.argv) > 1 else "data/sweep_ctrl_v2_128.csv"
df = pd.read_csv(csv)
N = df["grid_N"].iloc[0]

df["stable"] = (~df["collapsed"].astype(bool)) & (df["confinement"] > 1.5) & (df["barrier_aniso"] > 1.0)
df["score"] = df["confinement"] * df["barrier_aniso"] / (1.0 + df["effort"])

os.makedirs("figures", exist_ok=True)

# Key question: does coupling_alpha > 0 help?
no_ctrl = df[df["coupling_alpha"] == 0]
with_ctrl = df[df["coupling_alpha"] > 0]

print(f"=== Controller Necessity Analysis ({N}×{N}) ===")
print(f"Total runs: {len(df)}")
print(f"α=0 (no coupling):  {no_ctrl['stable'].mean():.1%} stable ({no_ctrl['stable'].sum()}/{len(no_ctrl)})")
print(f"α>0 (with coupling): {with_ctrl['stable'].mean():.1%} stable ({with_ctrl['stable'].sum()}/{len(with_ctrl)})")
print()

# Per coupling value
for alpha in sorted(df["coupling_alpha"].unique()):
    sub = df[df["coupling_alpha"] == alpha]
    print(f"  α={alpha:.1f}: {sub['stable'].mean():.1%} stable  (mean conf={sub['confinement'].mean():.1f}, ban={sub['barrier_aniso'].mean():.1f})")

print()

# Per controller × coupling
print("=== Stability by controller × coupling ===")
pivot = df.groupby(["controller", "coupling_alpha"])["stable"].mean().unstack(fill_value=0)
print(pivot.applymap(lambda x: f"{x:.1%}").to_string())
print()

# Per controller × gain × coupling (condensed)
print("=== Best configs (stable, top 15 by score) ===")
stable = df[df["stable"]].sort_values("score", ascending=False).head(15)
print(stable[["controller", "heater", "heat_rx", "ctrl_gain", "coupling_alpha",
              "D_E", "confinement", "barrier_aniso", "effort", "score"]].to_string(index=False))

# --- Plots ---

# 1. Controller necessity: α=0 vs α>0 per controller type
fig, ax = plt.subplots(figsize=(10, 5))
ctrls = sorted(df["controller"].unique())
x = np.arange(len(ctrls))
alphas = sorted(df["coupling_alpha"].unique())
w = 0.8 / len(alphas)

for i, alpha in enumerate(alphas):
    rates = [df[(df["controller"]==c) & (df["coupling_alpha"]==alpha)]["stable"].mean() for c in ctrls]
    label = f"α={alpha:.1f}" + (" (no ctrl)" if alpha == 0 else "")
    ax.bar(x + i*w, rates, w, label=label)

ax.set_xticks(x + w*(len(alphas)-1)/2)
ax.set_xticklabels(ctrls, rotation=45)
ax.set_ylabel("Fraction stable")
ax.set_title(f"Controller necessity — v2 fluctuation damping ({N}×{N})")
ax.legend()
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(f"figures/ctrl_necessity_v2_{N}.png", dpi=150)
print(f"\nSaved ctrl_necessity_v2_{N}.png")

# 2. Heatmap: heater × controller at best coupling
best_alpha = df.groupby("coupling_alpha")["stable"].mean().idxmax()
sub = df[df["coupling_alpha"] == best_alpha]
pivot_hm = sub.groupby(["heater", "controller"])["stable"].mean().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(pivot_hm.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(pivot_hm.columns)))
ax.set_xticklabels(pivot_hm.columns, rotation=45)
ax.set_yticks(range(len(pivot_hm.index)))
ax.set_yticklabels(pivot_hm.index)
for i in range(len(pivot_hm.index)):
    for j in range(len(pivot_hm.columns)):
        ax.text(j, i, f"{pivot_hm.values[i,j]:.0%}", ha="center", va="center", fontsize=9)
plt.colorbar(im, ax=ax, label="Stability fraction")
ax.set_title(f"Heater × Controller @ α={best_alpha:.1f} ({N}×{N})")
fig.tight_layout()
fig.savefig(f"figures/heater_ctrl_v2_{N}.png", dpi=150)
print(f"Saved heater_ctrl_v2_{N}.png")

# 3. Gain effect per controller
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, alpha_cond, title in zip(axes,
    [df["coupling_alpha"] == 0, df["coupling_alpha"] > 0],
    ["α=0 (baseline)", "α>0 (with damping)"]):
    sub = df[alpha_cond]
    for ctrl in ctrls:
        cs = sub[sub["controller"] == ctrl]
        gs = cs.groupby("ctrl_gain")["stable"].mean()
        ax.plot(gs.index, gs.values, "o-", label=ctrl)
    ax.set_xlabel("Controller gain")
    ax.set_ylabel("Fraction stable")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
fig.suptitle(f"Gain effect on stability ({N}×{N})", fontsize=14)
fig.tight_layout()
fig.savefig(f"figures/gain_effect_v2_{N}.png", dpi=150)
print(f"Saved gain_effect_v2_{N}.png")

# 4. D_E effect with and without controller
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, alpha_cond, title in zip(axes,
    [df["coupling_alpha"] == 0, df["coupling_alpha"] > 0],
    ["No coupling (α=0)", "With coupling (α>0)"]):
    sub = df[alpha_cond]
    for ctrl in ctrls:
        cs = sub[sub["controller"] == ctrl]
        ds = cs.groupby("D_E")["stable"].mean()
        ax.plot(ds.index, ds.values, "o-", label=ctrl)
    ax.set_xlabel("D_E")
    ax.set_ylabel("Fraction stable")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
fig.suptitle(f"D_E effect on stability ({N}×{N})", fontsize=14)
fig.tight_layout()
fig.savefig(f"figures/de_effect_v2_{N}.png", dpi=150)
print(f"Saved de_effect_v2_{N}.png")

plt.close("all")
