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
    HEAT_TARGET         = 4
};

struct SimParams {
    int   Nx, Ny;
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
    float heat_cx, heat_cy;
    float heat_rx, heat_ry;
    float heat_peak;

    // --- S dynamics: S_nat = (E/E_ref)·I + κ·(∇E⊗∇E)/E_ref² ---
    float grad_kappa;        // gradient anisotropy coupling strength
    float grad_tau;          // relaxation time: dS/dt = -(S - S_nat)/tau
    float grad_E_ref;        // reference energy scale

    // --- Wall ---
    float wall_radius;

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
};

#ifdef __cplusplus
}
#endif
