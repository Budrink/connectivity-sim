#pragma once

#include "sim_params.h"
#include <cuda_runtime.h>

// SoA field pointers — all arrays have size Nx*Ny
struct GridFieldsPtrs {
    // State (double-buffered)
    float *x0, *x1;
    float *x0_buf, *x1_buf;

    // Energy (double-buffered)
    float *E, *E_buf;

    // Symmetric tensor S — 3 independent components (double-buffered)
    float *s00, *s01, *s11;
    float *s00_buf, *s01_buf, *s11_buf;

    // Vorticity (double-buffered)
    float *omega, *omega_buf;

    // Control output
    float *u0, *u1;

    // PID per-cell state
    float *pid_int0, *pid_int1;
    float *pid_prev0, *pid_prev1;

    // Event-triggered per-cell state
    float *evt_prev_xn;
    unsigned char *evt_active;

    // Static fields
    float *heat_profile;
    unsigned char *is_wall;

    // Per-cell wall flux (energy lost to wall neighbors this step)
    float *wall_flux;

    // Wall thermal state (only wall cells meaningful)
    float *wall_E;

    // Delayed S observation (exponential smoothing for controller)
    float *s00_obs, *s01_obs, *s11_obs;

    // Equilibrium data (static, set once at init)
    float *eq_psi_norm;  // normalized poloidal flux (0=axis, 1=LCFS)
    float *eq_q;         // local safety factor
    float *eq_bR;        // poloidal B_R component (unit vec)
    float *eq_bZ;        // poloidal B_Z component (unit vec)

    // Readback buffers (device-side staging for D2H copy)
    float *rb_E;
    float *rb_aniso;
    float *rb_omega;
    float *rb_effort;
    float *rb_wall_flux;
    float *rb_aniso_angle;   // principal direction of S anisotropy [0, π)
    float *rb_gradE_sq;      // |∇E|² — GL bifurcation control parameter
    float *rb_fisher_min;    // min Fisher eigenvalue — observation quality
    float *rb_psi_norm;      // equilibrium psi_norm for display
};

// ---- Global metrics (reduced on GPU) ----
struct GlobalMetrics {
    float total_E;
    float center_E;
    float edge_E;
    float mean_aniso;
    float barrier_aniso;
    float total_wall_flux;
    float mean_effort;
    float mean_omega_abs;
    float mean_x_norm;
    float max_x_norm;
    float mean_shear;       // mean |grad omega|^2 in barrier region
    float confinement;      // center_E / edge_E
    float fusion_margin;    // center_E / E_target
    float mean_fisher_min;  // mean of min(Fisher eigenvalue) — observability
    float mean_fisher_ctrl; // mean Fisher in controlled region (center)
    float total_ctrl_effort;// sum |u| in controlled region

    // Wall thermal state
    float max_wall_E;       // peak wall temperature
    float mean_wall_E;      // average wall temperature
    float total_radiation;  // total radiated power this step
    int   n_wall;

    // Disruption indicators
    int   n_interior;
    int   n_center;
    int   n_edge;
    int   n_barrier;
};

// ---- Kernel launch wrappers ----

void launch_init_fields(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s = 0);
void launch_observe_and_control(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s = 0);
void launch_transport_step(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s = 0);
void launch_tensor_step(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s = 0);
void launch_update_delayed_S(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s = 0);
void launch_readback(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s = 0);
void launch_compute_metrics(GridFieldsPtrs& f, const SimParams& p,
                            GlobalMetrics* d_out, cudaStream_t s = 0);
