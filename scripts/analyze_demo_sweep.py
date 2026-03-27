#!/usr/bin/env python3
"""Analyze demo sweep results to find interesting parameter combos."""
import pandas as pd
import sys

df = pd.read_csv("data/demo_sweep.csv")
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print()

# Basic stats per heater
print("=" * 70)
print("SUMMARY BY HEATER TYPE")
print("=" * 70)
for ht in df['heater'].unique():
    sub = df[df['heater'] == ht]
    stable = sub[sub['collapsed'] == 0]
    hollow = sub[sub['hollow'] == 1]
    print(f"\n{ht}: {len(sub)} runs, {len(stable)} stable ({100*len(stable)/len(sub):.0f}%), "
          f"{len(hollow)} hollow ({100*len(hollow)/max(len(sub),1):.0f}%)")
    if len(stable) > 0:
        print(f"  center_E:  {stable['center_E'].min():.3f} - {stable['center_E'].max():.3f}")
        print(f"  edge_E:    {stable['edge_E'].min():.3f} - {stable['edge_E'].max():.3f}")
        print(f"  conf:      {stable['confinement'].min():.2f} - {stable['confinement'].max():.2f}")
        print(f"  barrier_a: {stable['barrier_aniso'].min():.3f} - {stable['barrier_aniso'].max():.3f}")
        print(f"  wall_flux: {stable['wall_flux'].min():.5f} - {stable['wall_flux'].max():.5f}")

# Find most interesting cases
print("\n" + "=" * 70)
print("INTERESTING CASES")
print("=" * 70)

stable = df[df['collapsed'] == 0].copy()

# 1. Strongest hollow profile (edge >> center)
print("\n--- Strongest hollow profiles (edge_E / center_E) ---")
stable['hollow_ratio'] = stable['edge_E'] / stable['center_E'].clip(lower=0.01)
top_hollow = stable.nlargest(10, 'hollow_ratio')
for _, r in top_hollow.iterrows():
    print(f"  {r['heater']:14s} pw={r['power']:.1f} rx={r['radius']:.2f} k={r['grad_kappa']:.0f} "
          f"trig={r['trigger']:.1f} del={r['obs_delay']:.1f} resp={r['resp_tau']:.1f} | "
          f"cE={r['center_E']:.3f} eE={r['edge_E']:.3f} ratio={r['hollow_ratio']:.1f} "
          f"ban={r['barrier_aniso']:.3f}")

# 2. Strongest barrier anisotropy (ring)
print("\n--- Strongest barrier anisotropy (ring formation) ---")
top_ring = stable.nlargest(10, 'barrier_aniso')
for _, r in top_ring.iterrows():
    print(f"  {r['heater']:14s} pw={r['power']:.1f} rx={r['radius']:.2f} k={r['grad_kappa']:.0f} "
          f"trig={r['trigger']:.1f} del={r['obs_delay']:.1f} resp={r['resp_tau']:.1f} | "
          f"cE={r['center_E']:.3f} eE={r['edge_E']:.3f} ban={r['barrier_aniso']:.3f} "
          f"conf={r['confinement']:.2f}")

# 3. Best confinement (center >> edge)
print("\n--- Best confinement (center_E / edge_E) ---")
confined = stable[stable['center_E'] > 0.1]
top_conf = confined.nlargest(10, 'confinement')
for _, r in top_conf.iterrows():
    print(f"  {r['heater']:14s} pw={r['power']:.1f} rx={r['radius']:.2f} k={r['grad_kappa']:.0f} "
          f"trig={r['trigger']:.1f} del={r['obs_delay']:.1f} resp={r['resp_tau']:.1f} | "
          f"cE={r['center_E']:.3f} eE={r['edge_E']:.3f} conf={r['confinement']:.2f} "
          f"ban={r['barrier_aniso']:.3f}")

# 4. High total energy (active system, not dead)
print("\n--- Highest total energy (most active) ---")
top_E = stable.nlargest(10, 'total_E')
for _, r in top_E.iterrows():
    print(f"  {r['heater']:14s} pw={r['power']:.1f} rx={r['radius']:.2f} k={r['grad_kappa']:.0f} "
          f"trig={r['trigger']:.1f} del={r['obs_delay']:.1f} resp={r['resp_tau']:.1f} | "
          f"totE={r['total_E']:.1f} cE={r['center_E']:.3f} eE={r['edge_E']:.3f} "
          f"ban={r['barrier_aniso']:.3f}")

# 5. Event-driven with disruptions (dramatic)
print("\n--- Event-driven with disruptions (dramatic but surviving) ---")
evt_dis = stable[(stable['heater'] == 'event_driven') & (stable['disruptions'] > 0)]
evt_dis_sorted = evt_dis.sort_values('disruptions', ascending=False).head(10)
for _, r in evt_dis_sorted.iterrows():
    print(f"  {r['heater']:14s} pw={r['power']:.1f} rx={r['radius']:.2f} k={r['grad_kappa']:.0f} "
          f"trig={r['trigger']:.1f} del={r['obs_delay']:.1f} resp={r['resp_tau']:.1f} | "
          f"dis={r['disruptions']:.0f} cE={r['center_E']:.3f} eE={r['edge_E']:.3f} "
          f"ban={r['barrier_aniso']:.3f}")

# 6. Constant heater that survives vs collapses
print("\n--- Constant heater stability boundary ---")
const = df[df['heater'] == 'constant']
const_stable = const[const['collapsed'] == 0]
const_dead = const[const['collapsed'] == 1]
print(f"  Stable: {len(const_stable)}, Collapsed: {len(const_dead)}")
if len(const_stable) > 0:
    print("  Stable params:")
    for _, r in const_stable.head(5).iterrows():
        print(f"    pw={r['power']:.1f} rx={r['radius']:.2f} k={r['grad_kappa']:.0f} | "
              f"cE={r['center_E']:.3f} eE={r['edge_E']:.3f} conf={r['confinement']:.2f}")
if len(const_dead) > 0:
    print("  Collapsed params:")
    for _, r in const_dead.head(5).iterrows():
        print(f"    pw={r['power']:.1f} rx={r['radius']:.2f} k={r['grad_kappa']:.0f}")

# 7. Summary table: best per heater type
print("\n" + "=" * 70)
print("BEST DEMO CANDIDATE PER HEATER TYPE")
print("=" * 70)
for ht in ['constant', 'event_driven', 'pulsed', 'target']:
    sub = stable[stable['heater'] == ht]
    if len(sub) == 0:
        print(f"\n{ht}: NO STABLE RUNS")
        continue
    # Pick by barrier_aniso (most visually interesting)
    best = sub.loc[sub['barrier_aniso'].idxmax()]
    print(f"\n{ht} (best barrier):")
    print(f"  power={best['power']:.1f}  radius={best['radius']:.2f}  grad_kappa={best['grad_kappa']:.0f}")
    if ht == 'event_driven':
        print(f"  trigger={best['trigger']:.1f}  obs_delay={best['obs_delay']:.1f}  resp_tau={best['resp_tau']:.1f}")
    print(f"  center_E={best['center_E']:.3f}  edge_E={best['edge_E']:.3f}  "
          f"confinement={best['confinement']:.2f}")
    print(f"  barrier_aniso={best['barrier_aniso']:.3f}  wall_flux={best['wall_flux']:.5f}  "
          f"hollow={best['hollow']}")
