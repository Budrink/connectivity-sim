#!/usr/bin/env python3
"""Analyze GL sweep v3 results: controller × heater heatmaps."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os

mpl.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'figure.facecolor': 'white',
})

csv_path = sys.argv[1] if len(sys.argv) > 1 else 'data/sweep_gl_v3.csv'
out_dir = sys.argv[2] if len(sys.argv) > 2 else 'figures'
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(csv_path)
N = df['grid_N'].iloc[0]

df['stable'] = (df['collapsed'] == 0) & (df['confinement'] > 1.5) & (df['disruptions'] < 5)
df['score'] = np.where(df['stable'], df['confinement'] * df['barrier_aniso'] / (1 + df['effort']), 0)

# ---- 1. Stability heatmap: controller × heater (aggregated over all other params) ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

stab_pct = df.groupby(['controller', 'heater'])['stable'].mean().unstack(fill_value=0)
score_avg = df[df['stable']].groupby(['controller', 'heater'])['score'].mean().unstack(fill_value=0)

for ax, data, title, cmap, fmt in [
    (axes[0], stab_pct, 'Stability fraction', 'RdYlGn', '.0%'),
    (axes[1], score_avg, 'Avg quality score (stable runs)', 'viridis', '.2f'),
]:
    im = ax.imshow(data.values, cmap=cmap, aspect='auto',
                   vmin=0, vmax=max(data.values.max(), 0.01))
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=30, ha='right')
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index)
    ax.set_xlabel('Heater'); ax.set_ylabel('Controller')
    ax.set_title(title)
    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            v = data.values[i, j]
            ax.text(j, i, f'{v:{fmt}}', ha='center', va='center',
                    color='white' if v < data.values.max()*0.5 else 'black', fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle(f'Controller × Heater stability ({N}×{N})', fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(f'{out_dir}/ctrl_heat_matrix_{N}.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved ctrl_heat_matrix_{N}.png')


# ---- 2. Stability vs heat_rx for each heater type ----
fig, axes = plt.subplots(1, len(df['heater'].unique()), figsize=(4*len(df['heater'].unique()), 4),
                         sharey=True, squeeze=False)
axes = axes[0]

for ax_i, ht in enumerate(sorted(df['heater'].unique())):
    sub = df[df['heater'] == ht]
    piv = sub.groupby(['controller', 'heat_rx'])['stable'].mean().unstack(fill_value=0)
    for ctrl in piv.index:
        ax = axes[ax_i]
        ax.plot(piv.columns, piv.loc[ctrl], 'o-', label=ctrl, markersize=4)
    axes[ax_i].set_title(ht)
    axes[ax_i].set_xlabel('heat_rx')
    axes[ax_i].set_ylim(-0.05, 1.05)
    axes[ax_i].grid(True, alpha=0.3)

axes[0].set_ylabel('Stability fraction')
axes[-1].legend(fontsize=8, loc='upper right')
fig.suptitle(f'Stability vs heater radius ({N}×{N})', y=1.02)
fig.tight_layout()
fig.savefig(f'{out_dir}/stab_vs_radius_{N}.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved stab_vs_radius_{N}.png')


# ---- 3. Effect of coupling_alpha and eta_ctrl ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, param, label in [
    (axes[0], 'coupling_alpha', 'coupling α'),
    (axes[1], 'eta_ctrl', 'η_ctrl (parasitic heat)'),
]:
    piv = df.groupby(['controller', param])['stable'].mean().unstack(fill_value=0)
    for ctrl in piv.index:
        ax.plot(piv.columns, piv.loc[ctrl], 'o-', label=ctrl, markersize=5)
    ax.set_xlabel(label)
    ax.set_ylabel('Stability fraction')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

fig.suptitle(f'Effect of coupling and parasitic heat ({N}×{N})', y=1.02)
fig.tight_layout()
fig.savefig(f'{out_dir}/coupling_effect_{N}.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved coupling_effect_{N}.png')


# ---- 4. Stability vs gain for each controller ----
fig, axes = plt.subplots(1, len(df['controller'].unique()),
                         figsize=(4*len(df['controller'].unique()), 4),
                         sharey=True, squeeze=False)
axes = axes[0]

for ax_i, ctrl in enumerate(sorted(df['controller'].unique())):
    sub = df[df['controller'] == ctrl]
    piv = sub.groupby(['heater', 'ctrl_gain'])['stable'].mean().unstack(fill_value=0)
    for ht in piv.index:
        axes[ax_i].plot(piv.columns, piv.loc[ht], 'o-', label=ht, markersize=4)
    axes[ax_i].set_title(ctrl)
    axes[ax_i].set_xlabel('ctrl_gain')
    axes[ax_i].set_ylim(-0.05, 1.05)
    axes[ax_i].grid(True, alpha=0.3)

axes[0].set_ylabel('Stability fraction')
axes[-1].legend(fontsize=8, loc='upper right')
fig.suptitle(f'Stability vs controller gain ({N}×{N})', y=1.02)
fig.tight_layout()
fig.savefig(f'{out_dir}/stab_vs_gain_{N}.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved stab_vs_gain_{N}.png')


# ---- 5. Best configs table ----
best = df[df['stable']].nlargest(20, 'score')
print(f'\n=== Top 20 stable configs ({N}×{N}) ===')
cols = ['controller','heater','heat_rx','ctrl_gain','coupling_alpha','eta_ctrl',
        'confinement','barrier_aniso','wall_flux','effort','disruptions','score']
print(best[cols].to_string(index=False))


# ---- 6. Confinement heatmap: coupling_alpha × ctrl_gain for best controller/heater ----
if len(best) > 0:
    top = best.iloc[0]
    sub = df[(df['controller'] == top['controller']) & (df['heater'] == top['heater'])]
    piv = sub.groupby(['ctrl_gain', 'coupling_alpha'])['confinement'].mean().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(piv.values, cmap='plasma', aspect='auto', origin='lower')
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f'{v:.1f}' for v in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f'{v:.1f}' for v in piv.index])
    ax.set_xlabel('coupling_alpha'); ax.set_ylabel('ctrl_gain')
    ax.set_title(f'Confinement: {top["controller"]} + {top["heater"]} ({N}×{N})')
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            ax.text(j, i, f'{piv.values[i,j]:.1f}', ha='center', va='center',
                    color='white' if piv.values[i,j] < piv.values.max()*0.6 else 'black',
                    fontsize=8)
    fig.colorbar(im, ax=ax, label='confinement')
    fig.tight_layout()
    fig.savefig(f'{out_dir}/confinement_best_{N}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved confinement_best_{N}.png')


# ---- 7. Controller necessity plot ----
# Compare no-control (coupling=0, gain=min) vs best control for each heater
fig, ax = plt.subplots(figsize=(10, 5))

min_gain = df['ctrl_gain'].min()
no_ctrl = df[(df['coupling_alpha'] == 0.0) & (df['ctrl_gain'] == min_gain)]
no_ctrl_stab = no_ctrl.groupby('heater')['stable'].mean()

best_ctrl_stab = {}
for ht in df['heater'].unique():
    sub = df[(df['heater'] == ht) & (df['coupling_alpha'] > 0)]
    if len(sub) > 0:
        ctrl_stab = sub.groupby('controller')['stable'].mean()
        best_ctrl_stab[ht] = ctrl_stab.max()
    else:
        best_ctrl_stab[ht] = 0

x = np.arange(len(no_ctrl_stab))
w = 0.35
ax.bar(x - w/2, no_ctrl_stab.values, w, label='No control (α=0, gain=min)', color='#cc4444')
ax.bar(x + w/2, [best_ctrl_stab.get(h, 0) for h in no_ctrl_stab.index], w,
       label='Best controller (α>0)', color='#44aa44')
ax.set_xticks(x)
ax.set_xticklabels(no_ctrl_stab.index, rotation=20, ha='right')
ax.set_ylabel('Stability fraction')
ax.set_ylim(0, 1.1)
ax.legend()
ax.set_title(f'Controller necessity by heater type ({N}×{N})')
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig(f'{out_dir}/ctrl_necessity_{N}.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved ctrl_necessity_{N}.png')

print('\nDone!')
