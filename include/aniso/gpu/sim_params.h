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

    // --- Initial conditions ---
    float g_noise_init;

    // --- RNG ---
    unsigned long long seed;

    // --- Simulation time ---
    float t;
    unsigned int step_count;

    // --- Radiation (nonlinear energy loss) ---
    float gamma_rad;         // P_rad = gamma_rad * E^rad_exp
    float rad_exp;

    // --- Wall thermal model ---
    float wall_cooling;      // dE_wall/dt -= wall_cooling * E_wall
    float wall_E_max;        // melt threshold

    // --- Disruption ---
    float beta_limit;        // thermal quench if center_E > beta_limit (0 = off)

    // --- External Omega (antisymmetric transport: drift perpendicular to grad E) ---
    float omega_base;        // uniform Omega magnitude (signed: +CCW, -CW)
    float omega_r_power;     // radial profile: w(r) = omega_base * (r/wall_r)^omega_r_power

    // --- Equilibrium (real tokamak geometry) ---
    int   use_equilibrium;
    float chi_parallel;
    float chi_perp;

    // --- Self-consistent B-field (Ohm + Ampere) ---
    float V_loop;             // loop voltage driving current; Jz = V_loop * E^spitzer
    float spitzer_exp;        // J_z ~ E^exp (1.5 = Spitzer conductivity)
    float Bz_ext;             // external axial (toroidal) field magnitude
    int   poisson_iters;      // SOR iterations per field update
    int   field_update_every; // recompute B-field every N steps (0 = never)
    float sor_omega;          // SOR over-relaxation factor (1.0–1.9)
    float field_kappa;        // congruence field suppression strength
    float beta_scale;         // β degradation: fk_local = fk / (1 + E/(B²·beta_scale))
    float inv_aspect_ratio;   // ε = a/R₀; centrifugal drift v = ε·E/Bz outward (+x)

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
