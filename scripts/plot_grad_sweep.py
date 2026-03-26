#!/usr/bin/env python3
"""Analyze gradient-model 256×256 sweep: controller necessity & stability maps."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys, os

mpl.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13,
    'figure.facecolor': 'white', 'figure.dpi': 120,
})

csv_path = sys.argv[1] if len(sys.argv) > 1 else 'data/sweep_grad_256_clean.csv'
out_dir  = sys.argv[2] if len(sys.argv) > 2 else 'figures'
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(csv_path)
N  = int(df['grid_N'].iloc[0])
print(f"Loaded {len(df)} runs at {N}×{N}")

df['stable'] = (df['collapsed'] == 0) & (df['confinement'] > 1.02) & (df['disruptions'] < 3)
df['has_ctrl'] = df['coupling_alpha'] > 0
df['score'] = np.where(df['stable'],
                       df['confinement'] * df['barrier_aniso'] / (1 + df['effort']), 0)

# ====================================================================
# 1. Controller necessity: α=0 vs α>0  broken by noise_S
# ====================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for ctrl in sorted(df['controller'].unique()):
    sub = df[df['controller'] == ctrl]
    piv = sub.groupby('noise_S')['stable'].mean()
    ax.plot(piv.index, piv.values, 'o-', label=ctrl, lw=2, ms=5)
ax.set_xlabel('noise_S')
ax.set_ylabel('Stability fraction')
ax.set_ylim(-0.05, 1.05)
ax.set_title('Stability vs S-noise by controller')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[1]
no_ctrl  = df[df['coupling_alpha'] == 0].groupby('noise_S')['stable'].mean()
yes_ctrl = df[df['coupling_alpha'] > 0].groupby('noise_S')['stable'].mean()
x = np.arange(len(no_ctrl))
w = 0.35
ax.bar(x - w/2, no_ctrl.values, w, label='No control (α=0)', color='#cc4444')
ax.bar(x + w/2, [yes_ctrl.get(sv, 0) for sv in no_ctrl.index], w,
       label='With controller (α>0)', color='#44aa44')
ax.set_xticks(x)
ax.set_xticklabels([f'{v:.2f}' for v in no_ctrl.index])
ax.set_xlabel('noise_S')
ax.set_ylabel('Stability fraction')
ax.set_ylim(0, 1.1)
ax.legend(); ax.set_title('Controller necessity vs S-noise')
ax.grid(True, alpha=0.3, axis='y')

fig.suptitle(f'Controller necessity — gradient model ({N}×{N})', y=1.02)
fig.tight_layout()
fig.savefig(f'{out_dir}/grad_ctrl_necessity_{N}.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved grad_ctrl_necessity_{N}.png')


# ====================================================================
# 2. Heatmap: controller × heater  — stability fraction
# ====================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, sub, title in [
    (axes[0], df, 'All configs'),
    (axes[1], df[df['coupling_alpha'] > 0], 'Active controller (α>0)'),
]:
    piv = sub.groupby(['controller', 'heater'])['stable'].mean().unstack(fill_value=0)
    im = ax.imshow(piv.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=30, ha='right')
    ax.set_yticks(range(len(piv.index)));   ax.set_yticklabels(piv.index)
    ax.set_title(title)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f'{v:.0%}', ha='center', va='center',
                    color='white' if v < 0.5 else 'black', fontsize=11, fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle(f'Controller × Heater stability ({N}×{N})', y=1.02)
fig.tight_layout()
fig.savefig(f'{out_dir}/grad_ctrl_heat_matrix_{N}.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved grad_ctrl_heat_matrix_{N}.png')


# ====================================================================
# 3. Heatmap: coupling_alpha × noise_S  — stability (per controller)
# ====================================================================
ctrls = sorted(df['controller'].unique())
fig, axes = plt.subplots(1, len(ctrls), figsize=(4.2*len(ctrls), 4.5), sharey=True)
if len(ctrls) == 1:
    axes = [axes]
for ax, ctrl in zip(axes, ctrls):
    sub = df[df['controller'] == ctrl]
    piv = sub.groupby(['coupling_alpha', 'noise_S'])['stable'].mean().unstack(fill_value=0)
    im = ax.imshow(piv.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1, origin='lower')
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f'{v:.2f}' for v in piv.columns])
    ax.set_yticks(range(len(piv.index)));   ax.set_yticklabels([f'{v:.0f}' for v in piv.index])
    ax.set_xlabel('noise_S'); ax.set_title(ctrl, fontsize=10)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f'{v:.0%}', ha='center', va='center',
                    color='white' if v < 0.5 else 'black', fontsize=9)
axes[0].set_ylabel('coupling_α')
fig.suptitle(f'coupling_α × noise_S stability per controller ({N}×{N})', y=1.02)
fig.tight_layout()
fig.savefig(f'{out_dir}/grad_coupling_noise_{N}.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved grad_coupling_noise_{N}.png')


# ====================================================================
# 4. Heatmap: grad_kappa × noise_S  — stability (all / with ctrl)
# ====================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, sub, title in [
    (axes[0], df, 'All'),
    (axes[1], df[df['coupling_alpha'] > 0], 'With controller'),
]:
    piv = sub.groupby(['grad_kappa', 'noise_S'])['stable'].mean().unstack(fill_value=0)
    im = ax.imshow(piv.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1, origin='lower')
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f'{v:.2f}' for v in piv.columns])
    ax.set_yticks(range(len(piv.index)));   ax.set_yticklabels([f'{v:.0f}' for v in piv.index])
    ax.set_xlabel('noise_S'); ax.set_ylabel('grad_κ')
    ax.set_title(title)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f'{v:.0%}', ha='center', va='center',
                    color='white' if v < 0.5 else 'black', fontsize=11, fontweight='bold')
    fig.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle(f'grad_κ × noise_S stability ({N}×{N})', y=1.02)
fig.tight_layout()
fig.savefig(f'{out_dir}/grad_kappa_noise_{N}.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved grad_kappa_noise_{N}.png')


# ====================================================================
# 5. Confinement & barrier anisotropy by coupling_alpha (box plots)
# ====================================================================
stable = df[df['stable']]
if len(stable) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, label in [
        (axes[0], 'confinement', 'Confinement ratio'),
        (axes[1], 'barrier_aniso', 'Barrier anisotropy'),
    ]:
        data_per_alpha = [stable[stable['coupling_alpha'] == a][col].values
                          for a in sorted(stable['coupling_alpha'].unique())]
        bp = ax.boxplot(data_per_alpha, labels=[f'{a:.0f}' for a in sorted(stable['coupling_alpha'].unique())],
                        patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#77aadd')
        ax.set_xlabel('coupling_α'); ax.set_ylabel(label)
        ax.set_title(f'{label} vs coupling_α')
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'Performance metrics — stable configs ({N}×{N})', y=1.02)
    fig.tight_layout()
    fig.savefig(f'{out_dir}/grad_performance_{N}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved grad_performance_{N}.png')


# ====================================================================
# 6. Top configurations
# ====================================================================
print(f'\n=== Summary: {len(df)} total, {df["stable"].sum()} stable ({df["stable"].mean():.1%}) ===')
print(f'  α=0: {df[df["coupling_alpha"]==0]["stable"].mean():.1%} stable')
print(f'  α>0: {df[df["coupling_alpha"]>0]["stable"].mean():.1%} stable')

best = df[df['stable']].nlargest(20, 'score')
cols = ['controller', 'heater', 'heat_rx', 'ctrl_gain', 'coupling_alpha',
        'grad_kappa', 'noise_S', 'confinement', 'barrier_aniso', 'wall_flux',
        'effort', 'disruptions', 'score']
if len(best) > 0:
    print(f'\n=== Top 20 stable configs ({N}×{N}) ===')
    print(best[cols].to_string(index=False))
else:
    print('\nNo stable configs found!')

print('\nDone!')
