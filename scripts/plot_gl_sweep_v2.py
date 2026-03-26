#!/usr/bin/env python3
"""GL sweep v2 analysis: heater profile, controller, coupling importance."""

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
        r['heat_rx'] = float(r['heat_rx'])
        r['ctrl_gain'] = float(r['ctrl_gain'])
        r['coupling'] = float(r['coupling'])
        conf = r['confinement']
        ban = r['barrier_aniso']
        wf = r['wall_flux']
        dis = r['disruptions']
        r['score'] = conf * ban / max(wf, 0.01) / max(dis+1, 1) if r['collapsed']==0 else 0
    return rows

data = load('data/gl_sweep_v2_256.csv')
ok = [d for d in data if d['collapsed']==0]
coll = [d for d in data if d['collapsed']==1]
print(f'Total: {len(data)}, OK: {len(ok)} ({100*len(ok)/len(data):.0f}%), Collapsed: {len(coll)}')

# ─── 1. Stability rate by heater × heat_rx ─────
fig, ax = plt.subplots(figsize=(10, 6))
heaters = sorted(set(d['heater'] for d in data))
radii = sorted(set(d['heat_rx'] for d in data))
for h in heaters:
    rates = []
    for r in radii:
        sub = [d for d in data if d['heater']==h and d['heat_rx']==r]
        ok_sub = [d for d in sub if d['collapsed']==0]
        rates.append(100*len(ok_sub)/max(len(sub),1))
    ax.plot(radii, rates, 'o-', label=h, linewidth=2)
ax.set_xlabel('Heater radius (rx=ry)')
ax.set_ylabel('Stability rate (%)')
ax.set_title('Stability vs Heater Profile Width')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/gl2_stability_vs_radius.png', dpi=150)
plt.close()

# ─── 2. Confinement vs heat_rx by heater (averaged over controllers) ─────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, metric, title in zip(axes, ['confinement', 'barrier_aniso'],
                              ['Confinement', 'Barrier Anisotropy']):
    for h in heaters:
        means, stds = [], []
        for r in radii:
            vals = [d[metric] for d in ok if d['heater']==h and d['heat_rx']==r]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if vals else 0)
        ax.errorbar(radii, means, yerr=stds, fmt='o-', label=h, capsize=3)
    ax.set_xlabel('Heater radius'); ax.set_ylabel(title)
    ax.set_title(f'{title} vs Heater Width'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/gl2_confinement_vs_radius.png', dpi=150)
plt.close()

# ─── 3. Controller importance: does controller type matter? ─────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ctrls = sorted(set(d['controller'] for d in data))

# Stability by controller
ctrl_rates = {}
for c in ctrls:
    sub = [d for d in data if d['controller']==c]
    ok_sub = [d for d in sub if d['collapsed']==0]
    ctrl_rates[c] = 100*len(ok_sub)/max(len(sub),1)
axes[0].bar(range(len(ctrls)), [ctrl_rates[c] for c in ctrls])
axes[0].set_xticks(range(len(ctrls))); axes[0].set_xticklabels(ctrls, rotation=45)
axes[0].set_ylabel('Stability rate (%)'); axes[0].set_title('Stability by Controller')
axes[0].grid(True, alpha=0.3, axis='y')

# Mean confinement by controller (only non-collapsed, narrow heater)
for h in ['constant', 'event_driven']:
    conf_by_ctrl = {}
    for c in ctrls:
        vals = [d['confinement'] for d in ok
                if d['controller']==c and d['heater']==h and d['heat_rx'] <= 0.20]
        conf_by_ctrl[c] = np.mean(vals) if vals else 0
    x = np.arange(len(ctrls))
    w = 0.35
    offset = -0.175 if h == 'constant' else 0.175
    axes[1].bar(x+offset, [conf_by_ctrl[c] for c in ctrls], w, label=h, alpha=0.8)
axes[1].set_xticks(range(len(ctrls))); axes[1].set_xticklabels(ctrls, rotation=45)
axes[1].set_ylabel('Mean Confinement (rx≤0.20)'); axes[1].set_title('Controller Effect (narrow heater)')
axes[1].legend(); axes[1].grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{OUT}/gl2_controller_importance.png', dpi=150)
plt.close()

# ─── 4. Coupling alpha effect ─────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
coups = sorted(set(d['coupling'] for d in data))
for ax, metric, title in zip(axes, ['confinement', 'wall_flux'],
                              ['Confinement', 'Wall Flux']):
    for h in ['constant', 'event_driven']:
        means = []
        for a in coups:
            vals = [d[metric] for d in ok
                    if d['heater']==h and d['coupling']==a and d['heat_rx']<=0.20]
            means.append(np.mean(vals) if vals else 0)
        ax.plot(coups, means, 'o-', label=h, linewidth=2)
    ax.set_xlabel('coupling_alpha'); ax.set_ylabel(title)
    ax.set_title(f'{title} vs Coupling (narrow heater)'); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/gl2_coupling_effect.png', dpi=150)
plt.close()

# ─── 5. Gain effect ─────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
gain_vals = sorted(set(d['ctrl_gain'] for d in data))
for ax, metric, title in zip(axes, ['confinement', 'effort'],
                              ['Confinement', 'Control Effort']):
    for c in ['aniso_aware', 'event_triggered', 'proportional']:
        means = []
        for g in gain_vals:
            vals = [d[metric] for d in ok
                    if d['controller']==c and d['ctrl_gain']==g
                    and d['heat_rx']<=0.20 and d['heater']=='constant']
            means.append(np.mean(vals) if vals else 0)
        ax.plot(gain_vals, means, 'o-', label=c)
    ax.set_xlabel('ctrl_gain'); ax.set_ylabel(title)
    ax.set_title(f'{title} vs Gain (constant heater, rx≤0.20)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/gl2_gain_effect.png', dpi=150)
plt.close()

# ─── 6. Heater importance: ANOVA-like variance decomposition ─────
from itertools import product

factors = {
    'heater': sorted(set(d['heater'] for d in ok)),
    'controller': sorted(set(d['controller'] for d in ok)),
    'heat_rx': sorted(set(d['heat_rx'] for d in ok)),
    'ctrl_gain': sorted(set(d['ctrl_gain'] for d in ok)),
    'coupling': sorted(set(d['coupling'] for d in ok)),
}
metric = 'confinement'
grand_mean = np.mean([d[metric] for d in ok])
var_explained = {}
for factor_name, levels in factors.items():
    group_means = []
    for lev in levels:
        vals = [d[metric] for d in ok if d[factor_name]==lev]
        if vals:
            group_means.append((np.mean(vals), len(vals)))
    ssb = sum(n * (gm - grand_mean)**2 for gm, n in group_means)
    var_explained[factor_name] = ssb

total_var = sum(var_explained.values())
fig, ax = plt.subplots(figsize=(8, 5))
names = list(var_explained.keys())
pcts = [100*var_explained[n]/max(total_var,1e-8) for n in names]
bars = ax.barh(names, pcts, color=['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6'])
for bar, pct in zip(bars, pcts):
    ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2, f'{pct:.1f}%',
            va='center', fontsize=11)
ax.set_xlabel('% Variance in Confinement Explained')
ax.set_title('Factor Importance for Confinement')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f'{OUT}/gl2_factor_importance.png', dpi=150)
plt.close()

# ─── 7. Top configurations ─────
print('\n' + '='*100)
print('TOP 20 CONFIGURATIONS BY COMPOSITE SCORE')
print('='*100)
ok_sorted = sorted(ok, key=lambda d: -d['score'])
print(f'{"Controller":<20s} {"Heater":<14s} {"rx":>5s} {"gain":>5s} {"coup":>5s} '
      f'{"Conf":>7s} {"Ban":>7s} {"WF":>10s} {"Eff":>7s} {"Dis":>4s} {"Score":>8s}')
print('-'*100)
for d in ok_sorted[:20]:
    print(f'{d["controller"]:<20s} {d["heater"]:<14s} {d["heat_rx"]:5.2f} {d["ctrl_gain"]:5.1f} '
          f'{d["coupling"]:5.1f} {d["confinement"]:7.2f} {d["barrier_aniso"]:7.2f} '
          f'{d["wall_flux"]:10.4f} {d["effort"]:7.3f} {d["disruptions"]:4d} {d["score"]:8.1f}')

# Which heater is most common in top 50?
print('\nTop 50 breakdown:')
top50 = ok_sorted[:50]
for key in ['heater', 'controller', 'heat_rx', 'ctrl_gain', 'coupling']:
    counts = defaultdict(int)
    for d in top50:
        counts[d[key]] += 1
    print(f'  {key}: {dict(sorted(counts.items(), key=lambda x: -x[1]))}')

print('\nSaved all figures to figures/')
