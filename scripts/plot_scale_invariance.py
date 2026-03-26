#!/usr/bin/env python3
"""Scale-invariance and GL-bifurcation analysis plots."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / 'data' / 'scale_test_clean.csv'
OUT  = Path(__file__).resolve().parent.parent / 'figures'
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
df = df[df['collapsed'] == 0].copy()  # only non-collapsed runs
df['wall_flux_norm'] = df['wall_flux'] / df['grid_N']  # normalize by perimeter

grids = sorted(df['grid_N'].unique())
controllers = ['proportional', 'aniso_aware', 'event_triggered', 'pid']
ctrl_labels = {'proportional': 'P', 'aniso_aware': 'AnisoAware',
               'event_triggered': 'EventTrig', 'pid': 'PID'}
heaters = ['constant', 'target']
models = ['relax_aniso', 'shear_bifurcation']
model_labels = {'relax_aniso': 'Relax-Aniso', 'shear_bifurcation': 'GL-Shear'}

colors_ctrl = {'proportional': '#e74c3c', 'aniso_aware': '#2ecc71',
               'event_triggered': '#3498db', 'pid': '#9b59b6'}
ls_model = {'relax_aniso': '-', 'shear_bifurcation': '--'}
mk_model = {'relax_aniso': 'o', 'shear_bifurcation': 's'}

# ============================================================================
# Figure 1: Scale invariance — center_E, confinement, wall_flux vs grid_N
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Scale Invariance: Physics vs Grid Resolution', fontsize=16, fontweight='bold')

metrics = [('center_E', 'Center Energy'), ('confinement', 'Confinement Ratio'),
           ('wall_flux_norm', 'Wall Flux / N'), ('barrier_aniso', 'Barrier Anisotropy'),
           ('effort', 'Control Effort'), ('fisher_ctrl', 'Fisher Information')]

for heater_idx, heater in enumerate(heaters):
    for ax_idx, (metric, label) in enumerate(metrics):
        ax = axes[ax_idx // 3, ax_idx % 3]
        for ctrl in controllers:
            for model in models:
                sub = df[(df['controller'] == ctrl) & (df['heater'] == heater)
                         & (df['g_response'] == model)]
                if sub.empty:
                    continue
                style = '-' if heater == 'constant' else '--'
                alpha = 1.0 if heater == 'constant' else 0.5
                marker = mk_model[model]
                ax.plot(sub['grid_N'], sub[metric],
                        color=colors_ctrl[ctrl], linestyle=style,
                        marker=marker, markersize=5, alpha=alpha,
                        label=f'{ctrl_labels[ctrl]} {model_labels[model]} ({heater})'
                        if ax_idx == 0 else '')

for ax_idx, (metric, label) in enumerate(metrics):
    ax = axes[ax_idx // 3, ax_idx % 3]
    ax.set_xlabel('Grid N')
    ax.set_ylabel(label)
    ax.set_xscale('log', base=2)
    ax.set_xticks(grids)
    ax.set_xticklabels([str(g) for g in grids])
    ax.grid(True, alpha=0.3)

axes[0, 0].legend(fontsize=6, ncol=2, loc='best')
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / 'gl_scale_invariance.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved {OUT / "gl_scale_invariance.png"}')

# ============================================================================
# Figure 2: GL vs Relax comparison (direct) — grouped bar charts
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('GL-Shear Bifurcation vs Relax-Aniso: Direct Comparison', fontsize=16, fontweight='bold')

for gi, N in enumerate([64, 128, 256, 512]):
    ax = axes[gi // 2, gi % 2]
    ax.set_title(f'N={N}', fontsize=13, fontweight='bold')

    sub = df[(df['grid_N'] == N) & (df['heater'] == 'constant')]
    if sub.empty:
        continue

    bar_w = 0.15
    x = np.arange(len(controllers))
    for mi, model in enumerate(models):
        for metric_name, color, offset in [
            ('confinement', '#2ecc71', -bar_w),
            ('barrier_aniso', '#3498db', 0),
            ('fisher_ctrl', '#e74c3c', bar_w),
        ]:
            vals = []
            for ctrl in controllers:
                row = sub[(sub['controller'] == ctrl) & (sub['g_response'] == model)]
                vals.append(row[metric_name].values[0] if len(row) else 0)
            label = f'{metric_name} ({model_labels[model]})' if mi == 0 or gi == 0 else ''
            pattern = '' if model == 'relax_aniso' else '///'
            bars = ax.bar(x + offset + mi * bar_w * 3, vals, bar_w * 0.9,
                         color=color, alpha=0.7 if model == 'relax_aniso' else 0.5,
                         hatch=pattern, label=label if offset == -bar_w else '')

    ax.set_xticks(x + bar_w)
    ax.set_xticklabels([ctrl_labels[c] for c in controllers], fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylabel('Value')

axes[0, 0].legend(fontsize=7, loc='best')
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / 'gl_vs_relax_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved {OUT / "gl_vs_relax_comparison.png"}')

# ============================================================================
# Figure 3: Convergence test — metric vs 1/N (should approach constant)
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Convergence: Metrics vs 1/N (constant heater, non-PID controllers)',
             fontsize=14, fontweight='bold')

conv_metrics = [('center_E', 'Center Energy'), ('confinement', 'Confinement'),
                ('fisher_ctrl', 'Fisher Info')]

for ax_idx, (metric, label) in enumerate(conv_metrics):
    ax = axes[ax_idx]
    for ctrl in ['proportional', 'aniso_aware', 'event_triggered']:
        for model in models:
            sub = df[(df['controller'] == ctrl) & (df['heater'] == 'constant')
                     & (df['g_response'] == model)]
            if sub.empty:
                continue
            inv_n = 1.0 / sub['grid_N'].values
            ax.plot(inv_n, sub[metric].values,
                    color=colors_ctrl[ctrl], linestyle=ls_model[model],
                    marker=mk_model[model], markersize=7,
                    label=f'{ctrl_labels[ctrl]} {model_labels[model]}')
    ax.set_xlabel('1/N')
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
    if ax_idx == 0:
        ax.legend(fontsize=7)

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / 'gl_convergence.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved {OUT / "gl_convergence.png"}')

# ============================================================================
# Figure 4: Controller stability heatmap — grid_N × controller
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Controller Stability: Confinement Heatmap (constant heater)',
             fontsize=14, fontweight='bold')

for mi, model in enumerate(models):
    ax = axes[mi]
    ax.set_title(model_labels[model], fontsize=12)

    data_matrix = np.zeros((len(grids), len(controllers)))
    for gi, N in enumerate(grids):
        for ci, ctrl in enumerate(controllers):
            row = df[(df['grid_N'] == N) & (df['controller'] == ctrl)
                     & (df['heater'] == 'constant') & (df['g_response'] == model)]
            if len(row):
                data_matrix[gi, ci] = row['confinement'].values[0]
            else:
                data_matrix[gi, ci] = 0

    im = ax.imshow(data_matrix, aspect='auto', cmap='RdYlGn',
                   vmin=0, vmax=data_matrix.max() * 1.1)
    ax.set_xticks(range(len(controllers)))
    ax.set_xticklabels([ctrl_labels[c] for c in controllers], fontsize=9)
    ax.set_yticks(range(len(grids)))
    ax.set_yticklabels([str(g) for g in grids])
    ax.set_ylabel('Grid N')
    ax.set_xlabel('Controller')

    for gi in range(len(grids)):
        for ci in range(len(controllers)):
            val = data_matrix[gi, ci]
            color = 'white' if val < data_matrix.max() * 0.4 else 'black'
            ax.text(ci, gi, f'{val:.1f}', ha='center', va='center',
                    fontsize=10, color=color, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Confinement')

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / 'gl_stability_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved {OUT / "gl_stability_heatmap.png"}')

# ============================================================================
# Figure 5: Key finding — Fisher information enhancement by GL model
# ============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
fig.suptitle('Fisher Information: GL-Shear vs Relax-Aniso', fontsize=14, fontweight='bold')

for ctrl in ['proportional', 'aniso_aware', 'event_triggered']:
    for model in models:
        sub = df[(df['controller'] == ctrl) & (df['heater'] == 'constant')
                 & (df['g_response'] == model)]
        if sub.empty:
            continue
        ax.plot(sub['grid_N'], sub['fisher_ctrl'],
                color=colors_ctrl[ctrl], linestyle=ls_model[model],
                marker=mk_model[model], markersize=8, linewidth=2,
                label=f'{ctrl_labels[ctrl]} {model_labels[model]}')

    # Target heater too
    for model in models:
        sub = df[(df['controller'] == ctrl) & (df['heater'] == 'target')
                 & (df['g_response'] == model)]
        if sub.empty:
            continue
        ax.plot(sub['grid_N'], sub['fisher_ctrl'],
                color=colors_ctrl[ctrl], linestyle=ls_model[model],
                marker=mk_model[model], markersize=5, linewidth=1, alpha=0.4)

ax.set_xlabel('Grid N', fontsize=12)
ax.set_ylabel('Mean Fisher Information', fontsize=12)
ax.set_xscale('log', base=2)
ax.set_xticks(grids)
ax.set_xticklabels([str(g) for g in grids])
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUT / 'gl_fisher_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved {OUT / "gl_fisher_comparison.png"}')

# ============================================================================
# Summary statistics
# ============================================================================
print('\n=== Scale Invariance Summary ===')
for model in models:
    print(f'\n--- {model_labels[model]} ---')
    for heater in heaters:
        print(f'  Heater: {heater}')
        for ctrl in controllers:
            sub = df[(df['controller'] == ctrl) & (df['heater'] == heater)
                     & (df['g_response'] == model)]
            if sub.empty:
                print(f'    {ctrl_labels[ctrl]:15s}: COLLAPSED at all scales')
                continue
            cE = sub['center_E'].values
            conf = sub['confinement'].values
            ns = sub['grid_N'].values
            cE_cv = np.std(cE) / np.mean(cE) * 100 if np.mean(cE) > 0 else 999
            conf_cv = np.std(conf) / np.mean(conf) * 100 if np.mean(conf) > 0 else 999
            print(f'    {ctrl_labels[ctrl]:15s}: cE={cE.mean():.2f}±{cE.std():.2f} '
                  f'(CV={cE_cv:.1f}%)  conf={conf.mean():.2f}±{conf.std():.2f} '
                  f'(CV={conf_cv:.1f}%)  grids={list(ns)}')
