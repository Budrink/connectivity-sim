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
    float grad_tau;          // relaxation time: dS/dt = -(S - S_nat)/tau
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

    // --- Wall thermal model ---
    float wall_cooling;      // dE_wall/dt -= wall_cooling * E_wall
    float wall_E_max;        // melt threshold
    // Exchange into wall: transport acceptance = 1 ("zero wall resistance"); energy and
    // resistive charge heating are deposited into wall_E scaled by this (short-circuit / KZ).
    float wall_sink_E_gain;

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
    float field_kappa;        // fk in apply_field_congruence: parallel/perp eigenvalue ratio = 1+fk
    float inv_aspect_ratio;   // ε = a/R₀; centrifugal mass drift ~ ε·m/Bz on +x edges in k_exchange

    // --- Charge field (3D): q ~ mass scale; MC hop → j_acc → J → ∇²A = -J → B = ∇×A + B_ext ---
    float charge_mass_scale;  // q = scale * m at init
    float charge_R0;        // R = R0 / Sab (Sab = edge transport scalar); calibrates resistive scale
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
