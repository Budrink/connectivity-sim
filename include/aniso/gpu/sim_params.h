#pragma once

// Plain C struct — usable from both host and device code.
// Passed by value to CUDA kernels (fits in constant memory).

#ifdef __cplusplus
extern "C" {
#endif

enum ControllerType {
    CTRL_PROPORTIONAL   = 0,
    CTRL_ANISO_AWARE    = 1,
    CTRL_PULSED         = 2,
    CTRL_EVENT_TRIGGERED = 3,
    CTRL_PID            = 4
};

enum HeaterType {
    HEAT_CONSTANT       = 0,
    HEAT_PULSED         = 1,
    HEAT_EVENT_DRIVEN   = 2,
    HEAT_ANISO_AWARE    = 3,
    HEAT_TARGET         = 4
};

enum GResponseType {
    GRESP_RELAX_ANISO      = 0,
    GRESP_RELAX_ENERGY     = 1,
    GRESP_MELT             = 2,
    GRESP_LANDAU_ENERGY    = 3,
    GRESP_SHEAR_BIFURCATION = 4,
    GRESP_GRADIENT          = 5
};

struct SimParams {
    int   Nx, Ny;
    float dt;

    // --- Transport ---
    float D_E;              // energy diffusion coefficient (through S^{-1})
    float D_x;              // state diffusion coefficient
    float C_adv;            // advection strength (omega transport)
    float gamma_diss;       // energy dissipation rate
    float eta_ctrl;         // control effort -> heat coupling

    // --- Symmetric tensor S dynamics ---
    float tau_0;            // base relaxation time
    float kappa_aniso;      // anisotropy slows relaxation
    float kappa_shear;      // |grad omega|^2 suppresses anisotropy (barrier!)
    float noise_S;          // S noise amplitude (scaled by sqrt(E))
    float eig_lo, eig_hi;  // eigenvalue clamp bounds

    // --- Vorticity omega dynamics ---
    float kappa_baro;       // baroclinic source: nabla^2 E -> omega
    float visc_omega;       // linear damping of omega
    float diff_omega;       // spatial diffusion of omega
    float noise_omega;      // omega noise amplitude (scaled by sqrt(E))

    // --- Resolution (observation quality tied to S) ---
    float l0;               // base resolution scale
    float alpha;            // resolution-metric coupling exponent

    // --- Observer ---
    float sigma_G;          // S estimation noise scale
    float E_noise_beta;     // energy-dependent noise amplification

    // --- Controller ---
    int   controller_type;
    float ctrl_gain;
    float ctrl_u_max;
    float ctrl_period;      // for pulsed
    float ctrl_duty;        // for pulsed
    float ctrl_trigger;     // for event_triggered
    float ctrl_hysteresis;  // for event_triggered
    float ctrl_anticipation;// for event_triggered
    float ctrl_ki, ctrl_kd; // for PID

    // --- Heater ---
    int   heater_type;
    float heater_power;
    float heater_period;
    float heater_duty;
    float heater_trigger;
    float heater_hysteresis;
    float heater_E_target;
    float heater_k_heat;

    // --- Heating spatial profile (Gaussian) ---
    float heat_cx, heat_cy;
    float heat_rx, heat_ry;
    float heat_peak;

    // --- G response model ---
    int   g_response_type;
    float landau_nu;        // cubic saturation for Landau model

    // --- Wall ---
    float wall_radius;

    // --- Initial conditions ---
    float g_noise_init;

    // --- RNG ---
    unsigned long long seed;

    // --- Simulation time (managed on host, passed for heater/controller) ---
    float t;
    unsigned int step_count;

    // --- Coupling (control -> S drive) ---
    float coupling_alpha;   // drive strength
    float coupling_gamma;   // drive nonlinearity exponent

    // --- Target energy ---
    float E_target;

    // --- Radiation (nonlinear energy loss) ---
    float gamma_rad;         // Bremsstrahlung-like: P_rad = gamma_rad * E^rad_exp
    float rad_exp;           // radiation exponent (1.5 = Bremsstrahlung, 1.0 = linear)

    // --- Wall thermal model ---
    float wall_cooling;      // cooling rate: dE_wall/dt -= wall_cooling * E_wall
    float wall_E_max;        // melt threshold: E_wall > wall_E_max → hard disruption

    // --- β-limit (pressure limit) ---
    float beta_limit;        // thermal quench if center_E > beta_limit (0 = off)
    float thermal_quench_frac; // fraction of E dumped to wall on quench (0.8 = 80%)

    // --- ELM / disruption physics ---
    float elm_gradient_crit; // critical |∇E| above which ballooning instability fires
    float elm_drive;         // amplitude of ELM burst vorticity injection
    float elm_energy_frac;   // fraction of local E ejected to wall during ELM

    // --- Controller delay (observation lag) ---
    float ctrl_delay_tau;    // time constant for S observation delay (0 = instant)

    // --- Shear-bifurcation GL model (GRESP_SHEAR_BIFURCATION) ---
    float shear_crit;        // |∇E|² threshold for L→H transition
    float S_turb;            // turbulent-state eigenvalue (low S → high transport)
    float S_barrier;         // barrier-state eigenvalue (high S → low transport)
    float gl_rate;           // bifurcation transition rate
    float gl_diffS;          // spatial diffusion of S (Ginzburg term ∝ ∇²S)
    float l_crit;            // resolution limit: barrier degrades when l > l_crit
    float ctrl_l_suppress;   // how much control effort reduces effective l

    // --- First-principles gradient model (GRESP_GRADIENT) ---
    float grad_S0;           // isotropic base conductivity
    float grad_kappa;        // gradient coupling: S_nat = S0*I + kappa*(∇E⊗∇E)/E_ref²
    float grad_tau;          // relaxation time: dS/dt = -(S - S_nat)/tau
    float grad_E_ref;        // reference energy scale for gradient coupling

    // --- Equilibrium (real tokamak geometry) ---
    int   use_equilibrium;   // 0 = circular wall, 1 = GEQDSK equilibrium
    float chi_parallel;      // parallel transport coefficient (along B)
    float chi_perp;          // perpendicular transport coefficient (across B)
    float q_elm_scale;       // ELM threshold scaling: grad_crit * (q_local / q_elm_scale)
};

#ifdef __cplusplus
}
#endif
