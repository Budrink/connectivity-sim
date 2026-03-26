# Connectivity Tensor Transport Simulation

Real-time CUDA simulation of energy transport in an anisotropic medium governed by a **connectivity tensor** `G = S + Ω`.

The model replaces traditional MHD equations with a qualitatively simpler framework: a **symmetric connectivity tensor** `S` determines transport probabilities between lattice cells via stochastic Monte Carlo dynamics, while an **antisymmetric component** `Ω` (vorticity/magnetic confinement, planned) will control directed transport.

## Physics

### Core idea

Any continuous medium is characterized by two quantities:
- **Connectivity tensor** `G_ik` — the geometry of connections between regions
- **Resolution limit** `l` — minimum distinguishable scale, intrinsically tied to `G`

Energy transport is modeled as a fully stochastic Monte Carlo process on a 2D lattice:

1. For each edge `(i, j)`: compute energy difference `dE = E_j - E_i`
2. Transfer probability `p = 2 · S_avg · dt`, where `S_avg` is the S-tensor projection onto the edge direction
3. Transfer amount `δE = u₁ · dE` (random fraction), accepted if `u₂ ≤ p` (dice roll)
4. If `p ≥ 1`: cells merge — instant equalization `flux = dE/2`

### S-dynamics

The symmetric tensor `S` relaxes to a natural state determined by local energy:

```
S_natural = (E / E_ref) · I + κ · (∇E ⊗ ∇E) / E_ref²
```

- **Magnitude** scales with energy density `E` (hotter → more connected)
- **Anisotropy direction** follows the energy gradient `∇E`
- Relaxation timescale `τ` controls how fast `S` responds to energy changes

### Key emergent phenomena

- **Hollow temperature profiles**: heating the center causes energy to spread outward and accumulate at boundaries — analogous to tokamak density peaking inversions
- **Event-driven heater stability**: feedback-based heating (triggered by local anisotropy) is the only stable regime without magnetic confinement (`Ω`)
- **Heater response delay stabilization**: realistic observation delay and response time improve confinement

## Building

Requirements: CUDA 12+, CMake 3.20+, OpenGL, X11/Wayland.

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

All dependencies (yaml-cpp, GLFW, Dear ImGui, ImPlot) are fetched automatically via CMake FetchContent.

GPU architecture is set to `sm_89` (Ada Lovelace / RTX 4060 Ti). Adjust `CUDA_ARCHITECTURES` in CMakeLists.txt for other GPUs.

## Running

### Interactive GUI

```bash
./aniso_gpu ../configs/best_run.yaml
```

Real-time visualization of energy density, S-tensor anisotropy, eigenvalues, and radial profiles. Full parameter control via ImGui sliders.

### Headless parameter sweeps

```bash
./aniso_sweep_gl    # outputs CSV to stdout
```

Sweep results are analyzed with Python scripts in `scripts/`.

## Project structure

```
include/aniso/gpu/    GPU headers (sim_params, kernels, math utilities)
src/gpu/              CUDA kernel implementations (transport, S-dynamics)
src/gpu_main.cu       GUI application
src/gpu_sweep_gl.cu   Headless sweep binary
configs/              YAML parameter files
scripts/              Python analysis & plotting
```

## Status

Current model uses only the symmetric component `S`. The antisymmetric component `Ω` (magnetic field / vorticity) — which will enable the controller to redirect energy flows — is planned as the next major addition.
