#!/usr/bin/env python3
"""Analyze GL shear-bifurcation sweep results across resolutions."""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import os

OUT = 'figures'
os.makedirs(OUT, exist_ok=True)

def load(fn):
    with open(fn) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ('center_E','edge_E','confinement','wall_flux',
                   'barrier_aniso','effort','fisher_ctrl'):
            r[k] = float(r[k])
        for k in ('disruptions','collapsed','grid_N'):
            r[k] = int(r[k])
        r['grad_crit'] = float(r['grad_crit'])
        r['gl_rate'] = float(r['gl_rate'])
        r['coupling'] = float(r['coupling'])
    return rows

data128 = load('data/gl_sweep_128.csv')
data256 = load('data/gl_sweep_256.csv')
all_data = data128 + data256

# ─── 1. Confinement heatmap: controller × heater, for best grad_crit ─────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (label, data) in zip(axes, [('128×128', data128), ('256×256', data256)]):
    ctrls = sorted(set(d['controller'] for d in data))
    heats = sorted(set(d['heater'] for d in data))
    gc_vals = sorted(set(d['grad_crit'] for d in data))

    matrix = np.zeros((len(ctrls), len(heats)))
    for ic, c in enumerate(ctrls):
        for ih, h in enumerate(heats):
            sub = [d for d in data if d['controller']==c and d['heater']==h and d['collapsed']==0]
            if sub:
                matrix[ic, ih] = max(d['confinement'] for d in sub)

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(heats))); ax.set_xticklabels(heats, rotation=45)
    ax.set_yticks(range(len(ctrls))); ax.set_yticklabels(ctrls)
    for i in range(len(ctrls)):
        for j in range(len(heats)):
            ax.text(j, i, f'{matrix[i,j]:.1f}', ha='center', va='center', fontsize=10,
                    color='white' if matrix[i,j] > matrix.max()*0.6 else 'black')
    ax.set_title(f'{label}: Best Confinement')
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle('GL Barrier Model — Best Confinement by Controller×Heater', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUT}/gl_confinement_heatmap.png', dpi=150)
plt.close()
print('Saved gl_confinement_heatmap.png')

# ─── 2. Barrier anisotropy vs grad_crit, by controller ─────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (label, data) in zip(axes, [('128×128', data128), ('256×256', data256)]):
    ctrls = sorted(set(d['controller'] for d in data))
    gc_vals = sorted(set(d['grad_crit'] for d in data))
    for ctrl in ctrls:
        means = []
        for gc in gc_vals:
            sub = [d['barrier_aniso'] for d in data
                   if d['controller']==ctrl and d['grad_crit']==gc and d['collapsed']==0]
            means.append(np.mean(sub) if sub else 0)
        ax.plot(gc_vals, means, 'o-', label=ctrl)
    ax.set_xlabel('grad_crit (∇E² threshold)')
    ax.set_ylabel('Barrier Anisotropy')
    ax.set_title(f'{label}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Barrier Anisotropy vs GL Threshold', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUT}/gl_barrier_vs_crit.png', dpi=150)
plt.close()
print('Saved gl_barrier_vs_crit.png')

# ─── 3. Wall flux comparison: constant vs target heater ─────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (label, data) in zip(axes, [('128×128', data128), ('256×256', data256)]):
    for h in ['constant', 'target']:
        ctrls = sorted(set(d['controller'] for d in data))
        vals = []
        for c in ctrls:
            sub = [d['wall_flux'] for d in data
                   if d['controller']==c and d['heater']==h and d['collapsed']==0]
            vals.append(np.mean(sub) if sub else 0)
        ax.bar([i + (0.2 if h=='target' else -0.2) for i in range(len(ctrls))],
               vals, width=0.35, label=h, alpha=0.8)
    ax.set_xticks(range(len(ctrls))); ax.set_xticklabels(ctrls, rotation=45)
    ax.set_ylabel('Mean Wall Flux')
    ax.set_title(f'{label}')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Wall Flux by Controller & Heater', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUT}/gl_wall_flux.png', dpi=150)
plt.close()
print('Saved gl_wall_flux.png')

# ─── 4. Scale invariance: 128 vs 256 for key metrics ─────

def match_key(d):
    return (d['controller'], d['heater'], d['grad_crit'], d['gl_rate'], d['coupling'])

d128 = {match_key(d): d for d in data128 if d['collapsed']==0}
d256 = {match_key(d): d for d in data256 if d['collapsed']==0}
common = set(d128.keys()) & set(d256.keys())

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
metrics = [('confinement', 'Confinement'), ('barrier_aniso', 'Barrier Anisotropy'),
           ('center_E', 'Center Energy'), ('wall_flux', 'Wall Flux')]
for ax, (m, title) in zip(axes.flat, metrics):
    x = [d128[k][m] for k in common]
    y = [d256[k][m] for k in common]
    ax.scatter(x, y, s=8, alpha=0.4)
    mn, mx = min(min(x), min(y)), max(max(x), max(y))
    ax.plot([mn, mx], [mn, mx], 'r--', alpha=0.5, label='y=x')
    ax.set_xlabel('128×128')
    ax.set_ylabel('256×256')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Scale Invariance: 128 vs 256', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUT}/gl_scale_invariance.png', dpi=150)
plt.close()
print('Saved gl_scale_invariance.png')

# ─── 5. Coupling alpha effect on confinement ─────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (label, data) in zip(axes, [('128×128', data128), ('256×256', data256)]):
    ctrls = sorted(set(d['controller'] for d in data))
    coup_vals = sorted(set(d['coupling'] for d in data))
    for ctrl in ctrls:
        means = []
        for a in coup_vals:
            sub = [d['confinement'] for d in data
                   if d['controller']==ctrl and d['coupling']==a and d['collapsed']==0
                   and d['heater']=='target']
            means.append(np.mean(sub) if sub else 0)
        ax.plot(coup_vals, means, 'o-', label=ctrl)
    ax.set_xlabel('coupling_alpha')
    ax.set_ylabel('Confinement (target heater)')
    ax.set_title(f'{label}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Control Coupling Effect on Confinement', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUT}/gl_coupling_effect.png', dpi=150)
plt.close()
print('Saved gl_coupling_effect.png')

# ─── 6. Composite score radar: best configs ─────

fig, ax = plt.subplots(figsize=(10, 6))
for label, data in [('128×128', data128), ('256×256', data256)]:
    for d in data:
        if d['collapsed'] == 0:
            conf = d['confinement']
            ban = d['barrier_aniso']
            wf = d['wall_flux']
            dis = d['disruptions']
            d['score'] = conf * ban / max(wf, 0.01) / max(dis+1, 1)
        else:
            d['score'] = 0
    data_ok = [d for d in data if d['collapsed']==0]
    data_ok.sort(key=lambda d: -d['score'])
    top = data_ok[:20]
    labels_str = [f"{d['controller'][:6]}/{d['heater'][:4]}/gc{d['grad_crit']:.0f}/a{d['coupling']:.1f}" for d in top]
    scores = [d['score'] for d in top]
    ax.barh(range(len(top)), scores, alpha=0.6, label=label)

ax.set_yticks(range(20))
top20_128 = sorted([d for d in data128 if d.get('score',0)>0], key=lambda d:-d['score'])[:20]
ax.set_yticklabels([f"{d['controller'][:6]}/{d['heater'][:4]}/gc{d['grad_crit']:.0f}" for d in top20_128], fontsize=8)
ax.set_xlabel('Composite Score (conf × barrier_aniso / wall_flux / (disrupts+1))')
ax.set_title('Top 20 Configurations')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f'{OUT}/gl_top_configs.png', dpi=150)
plt.close()
print('Saved gl_top_configs.png')

# ─── 7. Summary table ─────
print('\n' + '='*80)
print('BEST CONFIGURATIONS (consistent across both resolutions)')
print('='*80)
print(f'{"Config":<50s} {"Conf128":>8s} {"Conf256":>8s} {"Ban128":>8s} {"Ban256":>8s} {"WF128":>10s} {"WF256":>10s}')
print('-'*100)

for d in sorted(data128, key=lambda d: -d.get('score',0))[:15]:
    k = match_key(d)
    if k in d256:
        d2 = d256[k]
        tag = f"{d['controller']}/{d['heater']}/gc={d['grad_crit']:.0f}/glr={d['gl_rate']:.0f}/a={d['coupling']:.1f}"
        print(f'{tag:<50s} {d["confinement"]:8.2f} {d2["confinement"]:8.2f} '
              f'{d["barrier_aniso"]:8.2f} {d2["barrier_aniso"]:8.2f} '
              f'{d["wall_flux"]:10.4f} {d2["wall_flux"]:10.4f}')
