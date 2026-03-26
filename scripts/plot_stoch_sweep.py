#!/usr/bin/env python3
"""Analyze stochastic transport sweep: show controller necessity."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys, os

mpl.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'figure.facecolor': 'white'})

csv_path = sys.argv[1] if len(sys.argv) > 1 else 'data/sweep_stoch_v1.csv'
out_dir = sys.argv[2] if len(sys.argv) > 2 else 'figures'
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(csv_path)
N = df['grid_N'].iloc[0]

df['stable'] = (df['collapsed'] == 0) & (df['confinement'] > 1.5) & (df['disruptions'] < 5)
df['score'] = np.where(df['stable'], df['confinement'] * df['barrier_aniso'] / (1 + df['effort']), 0)

# ---- 1. KEY PLOT: Stability vs stoch_E with/without controller ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Stability vs stoch_E per controller
ax = axes[0]
for ctrl in sorted(df['controller'].unique()):
    sub = df[df['controller'] == ctrl]
    piv = sub.groupby('stoch_E')['stable'].mean()
    ax.plot(piv.index, piv.values, 'o-', label=ctrl, markersize=5, linewidth=2)
ax.set_xlabel('stoch_E (noise amplitude)')
ax.set_ylabel('Stability fraction')
ax.set_ylim(-0.05, 1.05)
ax.set_title('Stability vs noise by controller')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right: Controller necessity — no-ctrl vs best-ctrl per stoch_E
ax = axes[1]
min_gain = df['ctrl_gain'].min()
no_ctrl = df[(df['coupling_alpha'] == 0.0) & (df['ctrl_gain'] == min_gain)]
no_ctrl_s = no_ctrl.groupby('stoch_E')['stable'].mean()

yes_ctrl = df[(df['coupling_alpha'] > 0)]
yes_ctrl_s = yes_ctrl.groupby('stoch_E')['stable'].mean()

x = np.arange(len(no_ctrl_s))
w = 0.35
ax.bar(x - w/2, no_ctrl_s.values, w, label='No control (α=0, gain=min)', color='#cc4444')
ax.bar(x + w/2, [yes_ctrl_s.get(sv, 0) for sv in no_ctrl_s.index], w,
       label='With controller', color='#44aa44')
ax.set_xticks(x)
ax.set_xticklabels([f'{v:.1f}' for v in no_ctrl_s.index])
ax.set_xlabel('stoch_E')
ax.set_ylabel('Stability fraction')
ax.set_ylim(0, 1.1)
ax.legend()
ax.set_title('Controller necessity vs noise level')
ax.grid(True, alpha=0.3, axis='y')

fig.suptitle(f'Stochastic transport: controller necessity ({N}×{N})', y=1.02)
fig.tight_layout()
fig.savefig(f'{out_dir}/stoch_ctrl_necessity_{N}.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved stoch_ctrl_necessity_{N}.png')


# ---- 2. Heatmap: stoch_E × heater, stability fraction ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# All data
piv_all = df.groupby(['heater', 'stoch_E'])['stable'].mean().unstack(fill_value=0)
# Only with controller (coupling > 0)
ctrl_df = df[df['coupling_alpha'] > 0]
piv_ctrl = ctrl_df.groupby(['heater', 'stoch_E'])['stable'].mean().unstack(fill_value=0)

for ax, data, title in [
    (axes[0], piv_all, 'All configs'),
    (axes[1], piv_ctrl, 'With active controller (α>0)'),
]:
    im = ax.imshow(data.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels([f'{v:.1f}' for v in data.columns])
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index)
    ax.set_xlabel('stoch_E'); ax.set_ylabel('Heater')
    ax.set_title(title)
    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            v = data.values[i, j]
            ax.text(j, i, f'{v:.0%}', ha='center', va='center',
                    color='white' if v < 0.5 else 'black', fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle(f'Heater × Noise stability ({N}×{N})', y=1.02)
fig.tight_layout()
fig.savefig(f'{out_dir}/stoch_heat_matrix_{N}.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved stoch_heat_matrix_{N}.png')


# ---- 3. Controller × stoch_E heatmap ----
fig, ax = plt.subplots(figsize=(8, 5))
piv = df.groupby(['controller', 'stoch_E'])['stable'].mean().unstack(fill_value=0)
im = ax.imshow(piv.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(len(piv.columns)))
ax.set_xticklabels([f'{v:.1f}' for v in piv.columns])
ax.set_yticks(range(len(piv.index)))
ax.set_yticklabels(piv.index)
ax.set_xlabel('stoch_E'); ax.set_ylabel('Controller')
ax.set_title(f'Controller × Noise stability ({N}×{N})')
for i in range(len(piv.index)):
    for j in range(len(piv.columns)):
        v = piv.values[i, j]
        ax.text(j, i, f'{v:.0%}', ha='center', va='center',
                color='white' if v < 0.5 else 'black', fontsize=10)
fig.colorbar(im, ax=ax, label='stability fraction')
fig.tight_layout()
fig.savefig(f'{out_dir}/stoch_ctrl_matrix_{N}.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved stoch_ctrl_matrix_{N}.png')


# ---- 4. Gain effect at high noise ----
high_noise = df[df['stoch_E'] >= df['stoch_E'].quantile(0.6)]
fig, axes = plt.subplots(1, len(high_noise['controller'].unique()),
                         figsize=(4*len(high_noise['controller'].unique()), 4),
                         sharey=True, squeeze=False)
axes = axes[0]
for ax_i, ctrl in enumerate(sorted(high_noise['controller'].unique())):
    sub = high_noise[high_noise['controller'] == ctrl]
    piv = sub.groupby(['heater', 'ctrl_gain'])['stable'].mean().unstack(fill_value=0)
    for ht in piv.index:
        axes[ax_i].plot(piv.columns, piv.loc[ht], 'o-', label=ht, markersize=4)
    axes[ax_i].set_title(f'{ctrl} (high noise)')
    axes[ax_i].set_xlabel('ctrl_gain')
    axes[ax_i].set_ylim(-0.05, 1.05)
    axes[ax_i].grid(True, alpha=0.3)
axes[0].set_ylabel('Stability fraction')
axes[-1].legend(fontsize=8, loc='upper right')
fig.suptitle(f'Gain effect at high stoch_E ({N}×{N})', y=1.02)
fig.tight_layout()
fig.savefig(f'{out_dir}/stoch_gain_effect_{N}.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved stoch_gain_effect_{N}.png')


# ---- 5. Top configs ----
best = df[df['stable']].nlargest(20, 'score')
print(f'\n=== Top 20 stable configs ({N}×{N}) ===')
cols = ['controller','heater','heat_rx','ctrl_gain','coupling_alpha','stoch_E',
        'confinement','barrier_aniso','wall_flux','effort','disruptions','score']
if len(best) > 0:
    print(best[cols].to_string(index=False))
else:
    print('No stable configs found!')


# ---- 6. Coupling effect at different noise levels ----
fig, axes = plt.subplots(1, len(df['stoch_E'].unique()),
                         figsize=(3.5*len(df['stoch_E'].unique()), 4),
                         sharey=True, squeeze=False)
axes = axes[0]
for ax_i, sv in enumerate(sorted(df['stoch_E'].unique())):
    sub = df[df['stoch_E'] == sv]
    piv = sub.groupby(['controller', 'coupling_alpha'])['stable'].mean().unstack(fill_value=0)
    for ctrl in piv.index:
        axes[ax_i].plot(piv.columns, piv.loc[ctrl], 'o-', label=ctrl, markersize=4)
    axes[ax_i].set_title(f'stoch={sv:.1f}')
    axes[ax_i].set_xlabel('coupling_alpha')
    axes[ax_i].set_ylim(-0.05, 1.05)
    axes[ax_i].grid(True, alpha=0.3)
axes[0].set_ylabel('Stability fraction')
axes[-1].legend(fontsize=7, loc='best')
fig.suptitle(f'Coupling effect vs noise level ({N}×{N})', y=1.02)
fig.tight_layout()
fig.savefig(f'{out_dir}/stoch_coupling_{N}.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved stoch_coupling_{N}.png')

print('\nDone!')
