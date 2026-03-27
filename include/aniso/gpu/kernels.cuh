#pragma once

#include "sim_params.h"
#include <cuda_runtime.h>

// SoA field pointers — all arrays have size Nx*Ny
struct GridFieldsPtrs {
    // Energy (double-buffered)
    float *E, *E_buf;

    // Symmetric tensor S — 3 independent components (double-buffered)
    float *s00, *s01, *s11;
    float *s00_buf, *s01_buf, *s11_buf;

    // Heater current power (double-buffered, exponential smoothing)
    float *hpow, *hpow_buf;

    // Delayed S observation (exponential smoothing for event-driven heater)
    float *s00_obs, *s01_obs, *s11_obs;

    // Static fields
    float *heat_profile;
    unsigned char *is_wall;

    // Per-cell wall flux (energy lost to wall neighbors this step)
    float *wall_flux;

    // Wall thermal state (only wall cells meaningful)
    float *wall_E;

    // Equilibrium data (static, set once at init)
    float *eq_psi_norm;  // normalized poloidal flux (0=axis, 1=LCFS)
    float *eq_q;         // local safety factor
    float *eq_bR;        // poloidal B_R component (unit vec)
    float *eq_bZ;        // poloidal B_Z component (unit vec)

    // Readback buffers (device-side staging for D2H copy)
    float *rb_E;
    float *rb_aniso;
    float *rb_aniso_angle;
    float *rb_wall_flux;
    float *rb_gradE_sq;
    float *rb_psi_norm;
};

// ---- Global metrics (reduced on GPU) ----
struct GlobalMetrics {
    float total_E;
    float center_E;
    float edge_E;
    float mean_aniso;
    float barrier_aniso;
    float total_wall_flux;
    float confinement;
    float total_radiation;

    // Wall thermal state
    float max_wall_E;
    float mean_wall_E;
    int   n_wall;

    // Region counts
    int   n_interior;
    int   n_center;
    int   n_edge;
    int   n_barrier;
};

// ---- Kernel launch wrappers ----

void launch_init_fields(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s = 0);
void launch_update_delayed_S(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s = 0);
void launch_transport_step(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s = 0);
void launch_tensor_step(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s = 0);
void launch_readback(GridFieldsPtrs& f, const SimParams& p, cudaStream_t s = 0);
void launch_compute_metrics(GridFieldsPtrs& f, const SimParams& p,
                            GlobalMetrics* d_out, cudaStream_t s = 0);
