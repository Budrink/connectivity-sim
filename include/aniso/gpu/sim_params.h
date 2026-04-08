#pragma once

// Plain C struct — usable from both host and device code.
// Passed by value to CUDA kernels (fits in constant memory).

#ifdef __cplusplus
extern "C" {
#endif

enum HeaterType {
    HEAT_CONSTANT       = 0,
    HEAT_PULSED         = 1,
    HEAT_EVENT_DRIVEN   = 2,
    HEAT_ANISO_AWARE    = 3,
    HEAT_TARGET         = 4,
    HEAT_BEAM_ARRAY     = 5
};

struct SimParams {
    int   Nx, Ny, Nz;
    float dt;

    // --- S eigenvalue bounds ---
    float eig_lo;         // minimum eigenvalue (prevents anti-diffusion)
    float eig_hi;         // safety cap (prevents float overflow)

    // --- Resolution scaling ---
    float l0;             // base cell resolution scale
    float res_alpha;      // S-to-transport power: T_eig = l0 * S_eig^(alpha/2)

    // --- Heater ---
    int   heater_type;
    float heater_power;
    float heater_period;
    float heater_duty;
    float heater_trigger;
    float heater_E_target;
    float heater_k_heat;

    // --- Heater dynamics ---
    float heater_response_tau;
    float heater_obs_delay;

    // --- Heating spatial profile (Gaussian) ---
    float heat_cx, heat_cy, heat_cz;
    float heat_rx, heat_ry, heat_rz;
    float heat_peak;

    // --- S dynamics: S_nat = (E/E_ref)·I + κ·(∇E⊗∇E)/E_ref² ---
    float grad_kappa;        // gradient anisotropy coupling strength
    float grad_tau;          // legacy / unused: S = S_tgt every step (no τ relaxation)
    float grad_E_ref;        // reference energy scale

    // --- Wall ---
    float wall_radius;
    float tube_length;       // physical z-extent (Nz cells span this)
    int   wall_z_periodic;   // periodic boundary condition in z
    int   num_pair_maps;     // Kawasaki pair-map pool (CPU-built); random slot per step

    // --- Initial conditions ---
    float g_noise_init;

    // --- RNG ---
    unsigned long long seed;

    // --- Simulation time ---
    float t;
    unsigned int step_count;

    // --- Plasma medium (mass field) ---
    float cord_radius;       // initial cord radius (fraction of wall_radius)
    float cord_mass;         // initial mass per cell inside cord
    float m0;                // half-saturation for acceptance: alpha = m/(m+m0)
    float m_ref;             // reference mass for tensor scaling
    float alpha_m;           // mass power in S_nat: S *= (m/m_ref)^alpha_m
    // 1: after Kawasaki exchange, rescale plasma m,q so Σm matches post-prepare (float drift).
    int   mass_fp_fix;

    // --- Wall thermal model ---
    float wall_cooling;      // dE_wall/dt -= wall_cooling * E_wall
    float wall_E_max;        // melt threshold
    // Exchange into wall: transport acceptance = 1 ("zero wall resistance"); energy and
    // resistive charge heating are deposited into wall_E scaled by this (short-circuit / KZ).
    float wall_sink_E_gain;
    // Effective partner mass on plasma–wall energy edges (Metropolis mass factor vs wma).
    float wall_edge_mass;
    // Plasma radiation loss dE/dt = rad_alpha * m * E^2 (E = cell energy); summed energy
    // deposited uniformly across wall cells (see n_wall_cells). 0 = off.
    float rad_alpha;
    int   n_wall_cells;      // set at GpuGrid::init from is_wall count (not loaded from YAML)

    // --- Disruption ---
    float beta_limit;        // thermal quench if center_E > beta_limit (0 = off)

    // --- Equilibrium (real tokamak geometry) ---
    int   use_equilibrium;
    float chi_parallel;
    float chi_perp;

    // --- Self-consistent B-field (charge transport + vector Poisson) ---
    float V_loop;             // loop voltage: Pv = V_loop * (ê·ẑ) on exchange edge (unit dir in x,y,z phys)
    float spitzer_exp;        // legacy YAML; unused (kept for config compatibility)
    float Bz_ext;             // external axial (toroidal) field magnitude
    int   poisson_iters;      // SOR iterations per field update
    int   field_update_every; // recompute B from j_acc every N steps (0 = never)
    float sor_omega;          // SOR over-relaxation factor (1.0–1.9)
    float inv_aspect_ratio;   // ε: C_mid l→r; C_mid=C0(1+t·ε·C0), t=x along +x (inner at i=0)
    float cent_C0;             // C0: inner C_mid; mass hop P∝p_base(1+Pc), Pc=C_mid·evx

    // --- Charge field (3D): q ~ mass scale; MC hop → j_acc → J → ∇²A = -J → B = ∇×A + B_ext ---
    float charge_mass_scale;  // q = scale * m at init
    float charge_R0;        // R = R0 / s_scaled; hop P uses s_scaled/R0; kernel floors R0 at 0.01 (no near-zero R)
    // Dimensionless; J from j_acc uses face areas in normalized xy (0..1) and tube_length in z:
    //   Jz  *= scale / (fe*dt*dx^2),   Jx,Jy *= scale / (fe*dt*dx*dz),  dx=1/(max(Nx,Ny)-1), dz=tube_length/Nz
    float charge_j_scale;

    // --- Beam array heating (used when heater_type == HEAT_BEAM_ARRAY) ---
    int   n_beams;            // number of heating beams
    float beam_sigma_r;       // radial Gaussian width (normalized coords)
    float beam_sigma_z;       // z Gaussian width (normalized coords)
    float beam_power;         // power per beam
    float beam_r0;            // radial placement from center (normalized, ~0.15)

    // --- Absorption modulation (all heater types) ---
    float heat_E_abs;         // Q *= 1/(1 + E/E_abs); 0 = disabled
};

#ifdef __cplusplus
}
#endif
