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

    // --- Edge transport scale: S_iso = grad_kappa*Ea*ma; Sab = S_iso*dist_fac*exp(-αm*Δm/Σm - αe*ΔE/ΣE); s_scaled=l0*(1+Sab) ---
    float grad_kappa;        // κ in S_iso = κ * E * m (computed per pair in k_exchange)
    float grad_tau;          // legacy / unused
    float grad_E_ref;        // reference energy (metrics / legacy readback)

    // --- Wall ---
    float wall_radius;
    float tube_length;       // physical z-extent (Nz cells span this)
    int   wall_z_periodic;   // periodic boundary condition in z
    int   num_pair_maps;     // = Nz (clamped): Kawasaki pair-map pool; random slot per step

    // --- Initial conditions ---
    float g_noise_init;

    // --- RNG ---
    unsigned long long seed;

    // --- Simulation time ---
    float t;
    unsigned int step_count;

    // --- Plasma medium (mass field) ---
    // cord_radius * wall_radius = outer R (hard support for random cord mass); init only
    float cord_radius;
    float cord_profile_frac; // unused (YAML compat; was Gaussian σ/R)
    float cord_mass;         // peak m on axis (r=0)
    // Mean axis in normalized grid coords [0,1] (same as heat_cx/cy); each z-slice shifts by cord_xy_wander
    float cord_cx, cord_cy;
    // Multiplicative noise: m *= max(0, 1 + cord_mass_noise * N(0,1)) per cell; 0 = off
    float cord_mass_noise;
    // Axis wander in xy vs z: smoothed N(0,1) along k, amplitude in normalized coords (~0.05 typical)
    float cord_xy_wander;
    float m0;                // half-saturation for acceptance: alpha = m/(m+m0)
    float m_ref;             // reference mass (legacy / metrics)
    float alpha_m;           // α_m in exp(-α_m*|ma-mb|/(ma+mb) - …) on edge Sab
    float alpha_e;           // α_e in exp(… - α_e*|Ea-Eb|/(Ea+Eb)) on edge Sab
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

    // --- Loop-voltage drive (signed Gaussian profile in xy) ---
    //   V(x,y) = V_loop_offset + V_loop_amp · exp(-½·((x-cx)²/rx² + (y-cy)²/ry²))
    // On each edge Pv = V_local · (ê·ẑ), V_local = ½(V_profile[a] + V_profile[b]).
    // Sign matters: V>0 → forward drive and ohmic heating j_half = ½·Pv·dm_q;
    //                V<0 → counter-drive, j_half<0 (physical counter-EMF cooling).
    // V_loop_offset may be negative → Gaussian tails dip below zero, producing
    // a radially signed V profile (source of ExB shear at the zero crossing).
    float V_loop_amp;           // peak amplitude at the centre (may be <0 to invert)
    float V_loop_offset;        // baseline (may be <0 → counter-driven outer ring)
    float V_loop_cx, V_loop_cy; // centre in normalized [0,1] grid coords
    float V_loop_rx, V_loop_ry; // Gaussian widths in normalized [0,1]

    // --- Magnetic field & current snapshot ---
    float spitzer_exp;          // legacy YAML key; unused (kept for config compatibility)
    float Bz_ext;               // external axial (toroidal) field magnitude — the only B
                                // field in the model; self-consistent Poisson B is removed,
                                // currents interact locally via the Ampere term in Pi_B.
    int   field_update_every;   // snapshot j_acc → J every N steps (0 = never).
                                // Pi_B reads Jx/Jy/Jz inside edge_pi_B_mid.
    int   j_smooth_window;      // EMA window (steps) for j_acc: decay = 1 − 1/W, hop·(1/W).
                                // Decoupled from field_update_every so we can suppress
                                // thermal noise with a wide window without rebuilding J
                                // too often.
    float inv_aspect_ratio;     // ε: C_mid grows linearly in +x; also tapers local_Bz.
    float cent_C0;              // inner C_mid; Pcx = C_mid·ê_x on the unit edge.
    float cent_bias_cterm;      // unused (YAML compat); Cbias uses Pcx·mshift_avg in k_exchange.

    // --- Charge transport (3D) ---
    // Each step  j_acc ← j_acc·(1−n) + hop·n,  n = 1/W;  J = snapshot of j_acc.
    //
    // Pi_B (magnetic contribution to pi_arg, applied per Kawasaki edge):
    //     Pi_B = ( −|B_ext×ê|  +  k·F·ê / (dm_q² + ε) ) · f_a,
    //     k = charge_j_scale,  f_a = dm_q/δM.
    //
    // Term 1: Larmor suppression of transverse hops in B_ext.
    // Term 2: Ampere force between two neighbouring current elements (double cross
    //   product, BAC−CAB identity, with the Biot–Savart convention r̂ = b→a = −ê):
    //     F·ê = [ (J_a·J_b) − (J_a·ê)(J_b·ê) ] / |r|².
    //   Parallel currents on a transverse edge → F·ê > 0  (currents attract).
    //   The 1/dm_q² rescaling cancels the bilinear-in-J factor (locally J ~ C·dm_q)
    //   so the term enters pi_arg with the same "per-dm_q" weight as Larmor.
    //   Global ⟨|J|²⟩ rescalings were tried and broke pinch directionality; the
    //   local form is preserved. See edge_pi_B_mid in kernels.cu.
    float ionization_k;         // Q = f(E,M)·M with f = tanh(k·E/(2M)); see ionization_f.
    float charge_mass_scale;    // legacy; cord init now uses ionization_f(E,m,k)·m directly.
    float charge_R0;            // R_edge = R0 / s_scaled; kernel floors R0 at 0.01.
    float charge_j_scale;       // k in the local Ampere term; calibration constant.

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
