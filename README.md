# connectivity-sim

CUDA Monte Carlo simulation of a coarse-grained anisotropic medium driven by a fixed acceptance operator. The current build models a magnetised, ionised plasma cord in a cylindrical tube with ohmic heating, an external axial magnetic field, and a passive wall sink, but the underlying operator framework is substrate-independent.

This repository accompanies the preprint *Retraction and operator-level reformulation* (in `paper/v2.tex`); see [Status](#status) for what is and is not claimed.

## What it is

A lattice-based Kawasaki Monte Carlo on a regular 3D grid (2D fallback when `Nz=1`). Each cell holds three local state fields:

- `E` — internal energy
- `m` — mass density
- `q` — charge density (derived from `E`, `m` via a model ionisation function)

Pairs of neighbouring cells exchange these fields stochastically through a single fused acceptance form. The acceptance is **not** derived from a free-energy functional; it is **chosen** as

```
P_accept = l0 + exp(pi_arg)        (clamped to [0, 1])
```

with `pi_arg` a sum of independent operator contributions described below. The parameter `l0` is an irreducible structural floor: a minimum acceptance probability that cannot be suppressed regardless of local conditions.

## Operator structure

For every accepted pair-hop, `pi_arg` aggregates contributions from each enabled physical channel:

```
pi_arg = ΔS_thermal           # 2-T local detailed balance for δE flow
       + Pcx · δM              # centrifugal pressure-like drive (toroidal)
       + ohmic · dm_q / δM     # V·J ohmic field-work on charged mass
       + larmor · dm_q / δM    # B-field deflection (Larmor suppression)
       + ampere · dm_q / δM    # local J·J Ampere force projection
       + l0 · s_scaled         # operator-level transport floor
```

Individual channels can be turned off by setting the corresponding parameter to zero.

### Thermal acceptance

The thermal driver uses the two-temperature local-detailed-balance form:

```
ΔS_thermal = δE · (m_b / E_b  −  m_a / E_a)
```

In normalised model units (k_B = 1, mass_ref = 1) this is dimensionless, gives canonical Metropolis behaviour at uniform temperature (ΔS = 0 ⇒ symmetric acceptance), and the correct heat-flow direction when local temperatures `T = E / (m · c_heat)` differ between the two cells. Each cell is treated as its own local reservoir; no single global temperature is assumed or used.

### Edge connectivity

The per-edge isotropic scalar `s_scaled` modulates the operator floor and source-asymmetric transport bias:

```
s_scaled = κ · E_a · m_a · dist_fac · (α_m · Δm_rel  +  α_e · ΔE_rel)
Δm_rel   = (m_a − m_b) / (m_a + m_b)
ΔE_rel   = (E_a − E_b) / (E_a + E_b)
```

`s_scaled` is computed on-the-fly per pair; the symmetric tensor `S` exposed to the visualisation panel is a separate diagnostic field reconstructed for display.

### Kawasaki pair maps

Each step uses a precomputed **pair map**: every cell stores a direction code 0..26 indicating which of its 26 Moore neighbours is its partner for that step (code 13 = unpaired). Multiple maps are stacked in a pool and the host picks one per step at random (with a random z-shift if z is periodic). This guarantees no race conditions: only the "owner" of each pair performs the exchange, and walls partner at most one bulk cell per slot.

### Coulomb and ionisation gates

Before any 3D mass/charge hop is even considered for Boltzmann acceptance, a hard physical-feasibility gate is applied:

```
E_a_post = E_a − de + j_half − dQ_loss
E_b_post = E_b + de + dQ_loss + j_half
require: E_a_post ≥ 0  AND  E_b_post ≥ 0
```

If the move would drive either side's energy negative after accounting for the field-work term `j_half = ½ · V_local · dm_q` and Joule loss `dQ_loss = (dm_q)² / R_edge`, it is rejected outright. This is the operator analog of "the system cannot pay for the move" — the same role as the bond-break gate in the sister `condense-sim` project.

## Geometry, boundary, drives

- **Cylindrical tube**: `is_wall` mask defined by `r_perp > wall_radius`; periodic in z when `wall_z_periodic = 1`.
- **Wall thermostat**: each step, every wall cell is clamped to a fixed `E_wall = m_w · c · T_bath`, then participates in pair exchange as a passive sink that absorbs both energy and charge.
- **Ohmic heating**: a Gaussian heater profile in (x,y) injects energy each step into bulk cells.
- **External axial field** `B_ext`: enters `pi_arg` through `larmor` and `ampere` terms; suppresses cross-field hops.
- **Loop voltage** `V_loop`: drives charge along z via the ohmic `Pv = V_local · ê_z` term in `pi_arg`.

## Step structure

```
1. update_delayed_S          # low-pass smoothing of S diagnostic (no state effect)
2. prepare_step              # heater injection, wall E-clamp, copy state -> _buf
3. mass_sum (pre)            # double-precision Σm_buf snapshot for FP-fix
4. decay_j_accum             # EMA decay of accumulated current j_acc
5. exchange                  # Kawasaki pair sweep with fused acceptance
6. mass_sum (post)           # post-step Σm_buf
7. apply_mass_fp_fix         # rescale non-wall m_buf by ref/post ratio
8. fill_J_from_charge_accum  # snapshot J ← j_acc for next step's pi_B reads
9. tensor_step               # diagnostic S(E,m) reconstruction for GUI
10. swap_buffers             # state <- _buf
11. readback (every Nth)     # D2H copy for GUI / metrics
```

## Status

This is research code. The `paper/v2.tex` preprint explicitly retracts several claims from an earlier version of this work, in particular:

- The simulation is a coarse-grained operator on a lattice; not a self-consistent MHD solver.
- Connections to the Planck era of cosmology (asserted in the earlier version) are withdrawn.
- The earlier "magnetic helicity" reading of `S` is withdrawn; `S` is just a per-edge connectivity scalar with the structure shown above.
- A patent application made on the basis of the earlier version was withdrawn.

What the current code reliably demonstrates:

- **Filamentary structures** form spontaneously at sufficient B_ext and current.
- **Anisotropic transport** parallel-vs-cross-field develops with the right qualitative direction.
- **Heating-profile-leakage**: narrow heating profiles relative to the cord radius cause edge collapse and disruption-like termination events; broad profiles do not. See `scripts/plot_heating_hero.py` and `paper/figures/heating_hero.png`.
- **`l0`-limited fluctuation amplitude**: turbulent fluctuation amplitude scales as `√l0`, consistent with `l0` acting as the structural floor.

The model has not been validated quantitatively against any experimental plasma device or against any first-principles MHD solver, and no such validation is planned within this project.

## Building

Requirements: CUDA 12+, CMake 3.20+, OpenGL, X11/Wayland.

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

All dependencies (yaml-cpp, GLFW, Dear ImGui, ImPlot) are fetched automatically via CMake FetchContent. The default GPU architecture is `sm_89` (Ada Lovelace / RTX 4060 Ti); adjust `CUDA_ARCHITECTURES` in `CMakeLists.txt` for other GPUs.

A convenience script at the repository root rebuilds and launches the GUI on the default config:

```bash
./rebuild_and_run.sh
```

## Running

### Interactive GUI

```bash
./build/aniso_gpu configs/best_run.yaml
```

Provides z-slice heatmaps for E, m, q, J, and the diagnostic S-tensor (magnitude, dominant direction, condensetropy), plus time-series plots for total/center/edge energy, wall flux, and confinement metrics. All material constants and drive parameters are hot-tunable via ImGui sliders.

### Headless sweeps

```bash
./build/aniso_sweep_heating  [Nxy=96] [Nz=32] [steps=6000] [log_every=50]
```

Sweeps the heater profile width across a fixed B_ext + V_loop preset and emits per-step CSV to stdout. Used to produce the heating-hero figure in `paper/v2.tex`.

```bash
./build/aniso_sweep_3d
./build/aniso_sweep_gl
```

Older sweep variants kept for reproducibility of earlier scans.

## Project structure

```
include/aniso/gpu/        GPU headers (sim_params, kernels, math utilities)
src/gpu/                  CUDA kernels (k_init, k_prepare, k_exchange, k_tensor, ...)
src/gpu_main.cu           GUI application
src/gpu_sweep_*.cu        Headless sweep binaries (heating, 3d, gl)
configs/                  YAML parameter presets
scripts/                  Python analysis and plotting
paper/                    LaTeX preprint (v2: retraction + operator-level reformulation)
```

## Related work

A separate, leaner build of the same operator framework — applied to solid/liquid/gas phase emergence rather than plasma — is in active development under the name `condense-sim`. Both projects share the `P_accept = l0 + exp(pi_arg)` core and the per-edge `s_scaled` formulation; they differ in which physical channels populate `pi_arg` and which feasibility gates are enforced (Coulomb gate here vs bond-break mass-gate there).

## License

See `LICENSE` in the repository root.
